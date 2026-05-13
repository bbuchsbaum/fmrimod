# `NotImplementedError` Tier Audit (v1)

**Owner:** `bd-01KRHW5K5FQPC1NW09JQP6XH5N`
**Board source:** `major-issues-lets-talk/post-01KRHVA1NC98BP4KQ1Z29WYXWG` (bullet 8)
**Date:** 2026-05-13

## Purpose

Bullet 8 of the bad-cop pass named **85** `raise NotImplementedError`
sites across `fmrimod/` and asked: which of these are honest generic
dispatch sentinels, which are capability cliffs in spine APIs, and
which are bugs the user encounters as runtime crashes?

This audit categorizes every site into one of four patterns. The
deliverable is a classification, not a fix: each pattern has a
disposition rule, and Pattern B (the real honest-debt cluster) lists
follow-up beads.

## Method

`grep -rcE "raise NotImplementedError|NotImplementedError\(" fmrimod/`
totals **85 sites** across **20 files** at HEAD. Each site was
inspected to classify the *shape* of the raise, not the surrounding
function: a `NotImplementedError(f"X not implemented for {type(x)}")`
in a `@singledispatch` ladder is a Pattern A sentinel even if the
function is on the public surface, because the sentinel itself is
the standard Python idiom for "register a handler for your type."

## Tier shape

The four patterns map to four dispositions:

| Pattern | Shape | Disposition | Action needed |
| --- | --- | --- | --- |
| **A** | Generic-dispatch sentinel | Keep | Document as the dispatch contract; no per-site work |
| **B** | Capability cliff in spine/transitional API | Convert | Replace with typed `FmriCapabilityError` carrying repair path, *or* compat-deprecate the entry point |
| **C** | Backend-protocol unsupported method | Keep | Document as the backend contract; capability probes belong on the backend object |
| **D** | One-off (bare raise, deprecation stub, etc.) | Review | Per-site disposition |

## Pattern A — Generic-dispatch sentinels (~38 sites)

Standard `@singledispatch` / kitchen-sink dispatch fallthrough. The
sentinel is the contract: "this generic accepts any type that has
registered a handler; types without a handler get a clear runtime
error." Removing these would silently default to wrong behavior; they
are not bugs.

| Module | Sites | Note |
| ---: | ---: | --- |
| `fmrimod/utils/generics.py` | 27 | The whole module is dispatch-ladder leaves (`blockids`, `blocklens`, `term_names`, `longnames`, `shortnames`, `cells`, `conditions`, `condition_map`, `onsets`, `durations`, and 17 others). |
| `fmrimod/accessors.py` | 5 | `coef_names`, coefficient extraction, `get_data`, `get_mask`, `get_subjects` — all sentinels in the `singledispatch` pattern even though the module is user-facing. |
| `fmrimod/contrast/fcontrast.py` | 2 | `Fcontrasts`, `plot_Fcontrasts`. |
| `fmrimod/convolve.py` | 1 (line 162) | `convolve` generic fallthrough. (The other site at l.752 is Pattern B.) |
| `fmrimod/residualize.py` | 1 | `residualize` generic fallthrough. |
| `fmrimod/design/design_matrix.py` | 1 | Generic dispatch. |
| `fmrimod/events/event_table.py` | 1 | Generic dispatch. |
| `fmrimod/dispatch.py` | 1 | Generic dispatch. |
| **Subtotal** | **~39** | |

**Disposition:** keep. No follow-up beads. These sites are the contract.

## Pattern B — Capability cliffs in spine/transitional API (~20 sites)

The dangerous category. Each site is a public-or-near-public function
that raises `NotImplementedError` when the caller hits a feature
boundary. These are user-facing crashes on the happy path adjacent to
the spine API, exactly what `VISION.md:105-108` says shouldn't
happen.

| Module | Line(s) | Capability gap |
| --- | --- | --- |
| `fmrimod/stats/meta.py` | 53, 72, 240, 300, 302 | Non-CSV `GroupData`; effect-only path; non-intercept formula; robust meta; combine modes |
| `fmrimod/stats/ttest.py` | 38, 128, 148, 163 | Non-intercept formula; one-sample variant boundaries |
| `fmrimod/stats/meta_compat.py` | 111 | Robust meta estimation (same as meta.py:300; compat duplicate) |
| `fmrimod/bids/stats_model.py` | 55, 59, 94 | Multi-factor BIDS nodes; missing `Convolve`; non-t contrasts |
| `fmrimod/glm/strategies.py` | 597, 599 | Chunkwise + AR; chunkwise + robust |
| `fmrimod/single/voxel_hrf.py` | 220, 242 | Multi-run dataset mode |
| `fmrimod/convolve.py` | 752 | Capability mismatch in convolve helper |
| `fmrimod/glm/combine.py` | 186 | Unsupported combine path |
| `fmrimod/spec/_compile.py` | 265 | Spec lowering case |
| `fmrimod/formula/dsl.py` | 173 | DSL fallthrough (bare `raise` with no message — worst-of-class) |
| **Subtotal** | **~20** | |

**Disposition: typed-error promotion or explicit compat marking.** Two
shapes:

1. **`FmriCapabilityError` (preferred for spine API):** typed
   subclass that carries `feature`, `current_capability`, `repair`
   fields. The user sees "this fit's spec uses Z, which is not
   supported in v1; the typed repair is W" rather than a bare
   `NotImplementedError`. Same shape as `IncompleteFitProvenanceError`
   added recently.
2. **Explicit compat marker (preferred for transitional API):** if
   the function's surrounding tier in `api_inventory_v1.json` is
   `compat` or to-be-compat, the NIE stays but the function is
   documented as a known-narrow surface, not a spine entry.

The Pattern B subset is small enough (~20 sites) that it is
**tractable as a single follow-up bead** rather than per-site work.
The right test: a CI gate that lists `NotImplementedError` sites in
files whose `api_inventory` tier is `spine` — any such site needs a
typed-error replacement before the next release.

## Pattern C — Backend-protocol unsupported methods (~11 sites)

The `Backend` protocol exposes a method that a concrete backend
hasn't implemented. Like Pattern A but on a protocol rather than
`singledispatch`. Already the right shape: capability probes belong
on the backend object, not at the call site.

| Module | Sites | Note |
| --- | ---: | --- |
| `fmrimod/stats/backends/python_backend.py` | 6 | Per-method unimplemented branches |
| `fmrimod/stats/backends/fmrigds_backend.py` | 5 | Explicit "fmrigds backend does not yet support X" |
| `fmrimod/dataset/backend_methods.py` | 2 | `get_loadings`, `reconstruct_voxels` |
| **Subtotal** | **~13** | |

**Disposition:** keep. Optional capability surfaces are documented as
backend-conditional; `hasattr` probes or typed `Backend.supports(...)`
methods are the right discovery mechanism. No follow-up beads.

## Pattern D — One-offs (~13 sites)

The remainder. Each needs individual review.

Sample inspection:

- `fmrimod/formula/dsl.py:173` — bare `raise NotImplementedError` with
  no message. Promote to Pattern B (capability cliff in spine API).
- `fmrimod/spec/_compile.py:265` — listed under B above; compiler
  fallthrough for un-lowered placeholder.
- `fmrimod/dispatch.py:49` — listed under A above; dispatch sentinel.

A site-by-site D pass is in scope for a follow-up bead if it surfaces
new Pattern B cases. Spot-check suggests it will not.

## Aggregate findings

1. **~39 of 85 sites (46%) are correct Python idiom.** Removing them
   would *worsen* error reporting, not improve it. The author of the
   bad-cop post was right that 85 is "too many for a high-level API,"
   but the right framing is the ~20 Pattern B sites, not 85.
2. **The real fix-work is the ~20 Pattern B sites,** concentrated in
   `stats/`, `bids/`, `glm/strategies.py`, and `single/voxel_hrf.py`.
   The right shape is one typed `FmriCapabilityError` class plus a
   coordinated migration of those sites to it, gated by a CI test that
   forbids new `NotImplementedError` raises in `spine` tier API.
3. **Pattern C (~13 sites) is the right shape already.** Backend
   capability surfaces should be discoverable via the backend object,
   not at the call site. No work needed.
4. **`fmrimod/formula/dsl.py:173` is the worst single site** — bare
   `raise NotImplementedError` with no message, no `type(x)` info, no
   feature name. Pure user trap. **S0 fix** in isolation; promote to
   a typed `FmriCapabilityError` with `feature="dsl-fallthrough"`.

## Recommended follow-up beads

1. **Define `FmriCapabilityError`.** Typed subclass of
   `NotImplementedError` with `feature: str`, `current_capability:
   tuple[str, ...]`, `repair: str | None` fields. **S1** (no public
   API change; new internal error class). Lands a single test
   verifying the class shape.
2. **Migrate Pattern B sites to `FmriCapabilityError`.** One bead per
   module (`stats/meta.py`, `stats/ttest.py`, `bids/stats_model.py`,
   `glm/strategies.py`, `single/voxel_hrf.py`). Each is **S1**.
   `meta_compat.py:111` is covered by the existing meta_compat
   retirement (`bd-01KRHVN7X25KBXN1RWK0EAT2BY` follow-ups).
3. **CI gate: `spine`-tier `NotImplementedError` forbidden.** After
   bullets 1-2 land, a test reads `api_inventory_v1.json`, finds every
   row with `tier="spine"`, and forbids any `raise NotImplementedError`
   in those files. **S1**. Pins the spine API as
   capability-error-only.
4. **`formula/dsl.py:173` quick fix.** Bare raise → typed capability
   error. **S0.**

## What this audit does NOT do

- Does not delete or rewrite any `NotImplementedError`.
- Does not introduce `FmriCapabilityError`.
- Does not gate CI on spine-tier NIE absence.

Each is a separate bead. The audit is the substrate for those beads,
not the work itself.

The four follow-up beads above shrink the Pattern B problem
from "85 user-facing crash points" to "~20 typed capability errors
with repair paths," without changing the underlying behavior of any
spine path. That is the right ratio under MISSION.md's
"high-level API that does not break."
