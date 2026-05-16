# Scoped-mypy vs. full-strict gap — inventory v1

Audit substrate for **bd-01KRNN0H73CCYGFJSJ30JPVFTW** (reconcile the
typed-API epic's scoped gate with the authoritative full-strict
`[tool.mypy]` config). Inventory-before-action: this document is the
checked-in measurement; no retirement/refactor/config change is made
here.

> **DECISION ENACTED (2026-05-16):** the steward chose full-strict as
> the authoritative gate. The canonical rule now lives in
> [`decisions/0002-full-strict-is-the-authoritative-type-gate.md`](decisions/0002-full-strict-is-the-authoritative-type-gate.md);
> the residue is an exiting caveat in
> [`type_gate_caveats_v1.md`](type_gate_caveats_v1.md) owned by
> `bd-01KRRM1DB1CP8E5CEA2Q1Z49Z5` (EventModel/HRFProtocol seam) and
> `bd-01KRRM1DZH80R9SPHDH6QG6ZGR` (CovariateEvent taxonomy). This
> document is now the historical audit substrate; the
> §"Recommendation" section below is superseded by decision 0002 and
> retained only as the reasoning record.

> **STATUS (2026-05-15): mechanical reduction COMPLETE — full-strict
> 746 → 100 (646 cleared, ~87%, 104 files), full non-rpy2 suite
> 3007 passed / 0 failed.** The §"746 errors classified" and
> §"Verified findings" sections below are the *session-start*
> measurement and triage, preserved as the audit record. The current
> state and the crystallized residue the steward decision now turns on
> are in §"Post-reduction state (2026-05-15)" near the end. The
> mechanical + boundary-cast buckets are no longer a backlog — they
> are emptied; only steward-reserved / in-flight / documented
> scoped-strict divergence remains.

## The two gates

- **Epic gate (what ~38 typed-API beads were verified against):**
  per-subpackage `python3.9 -m mypy fmrimod/<pkg>/ --follow-imports=skip`.
  As of HEAD this is **0 errors across every top-level subpackage**,
  verified by a `find fmrimod -maxdepth 1 -type d` sweep (not a curated
  list — a curated list previously masked top-level `backends/`).
- **Authoritative config (`pyproject.toml [tool.mypy]`):** full-tree
  `strict = true`, `warn_return_any = true`, no `follow_imports`
  override. Running `python3.9 -m mypy fmrimod/` (project config):
  **746 errors in 105 files** *at session start*; **100 errors in
  18 files** after the mechanical reduction (see §Post-reduction).

The scoped gate is a proxy. `--follow-imports=skip` makes every
cross-module symbol opaque (`Any`), which both *hides* real
cross-module mismatches and *inverts* cast-redundancy (a cast scoped
mypy demands, full-strict flags redundant, and single-file vs
subpackage scope disagree again). The gap is structural, not a
backlog of the same recipe.

## 746 errors classified (session-start measurement, 2026-05-15)

| Bucket | Codes (count) | Nature |
| --- | --- | --- |
| **Mechanical** (~282) | `type-arg` 132, `no-any-return` 92, `no-untyped-def` 20, `unused-ignore` 14, `redundant-cast` 13, `no-untyped-call` 9, `var-annotated` 2 | Same recipe used across the 38 scoped beads: parametrize generics, coerce/cast returns, annotate, drop stale ignores. No judgment. |
| **Boundary-cast** (~433) | `arg-type` 194, `attr-defined` 115, `assignment` 36, `index` 30, `operator` 26, `call-overload` 18, `union-attr` 14 | Mostly `object`-typed duck params that should be `Any`, `**kwargs: object` re-unpacks, Optional-narrowing asserts. Mechanical *but* a minority are real cross-module type mismatches hiding in the same bucket — each needs a glance, not a blind cast. |
| **Judgment / likely-real** (~41) | `override` 9, `return-value` 8, `void` 8, `call-arg` 4, `misc` 3, `str`/`list-item`/`dict-item` 2 each, `abstract` 2, `func-returns-value` 1 | Real design questions or latent bugs. Must NOT be cast away. |

Top files: `glm/strategies.py` 165, `design/event_model.py` 93,
`convolve.py` 93, `glm/solver.py` 56, `glm/engines/sketch.py` 32,
`dataset/bids_h5.py` 30, `ar/whitening.py` 28, `hrf_dispatch.py` 26,
`dispatch.py` 26, `events/variable.py` 25.

## Verified findings (real, with red checks)

1. **`glm/compat.py:490` — `fmri_rlm()` was entirely broken
   [FIXED this commit].** `replace(base.robust, enabled=True)` —
   `RobustOptions.enabled` is a read-only `@property`
   (`type is not False`), not a field, so `dataclasses.replace`
   raised `TypeError` on *every* call. Zero prior test coverage.
   Fixed to `replace(base.robust, type="huber")` (documented
   default); regression test added
   (`test_fmri_rlm_enables_robust_without_replace_typeerror`,
   proven to fail on the pre-fix code).

2. **`condition_basis.py:61`, `design/event_model.py:2058` —
   `Cannot instantiate abstract class "EventModel"` [VERIFIED
   ARTIFACT, not a bug].** Runtime check:
   `EventModel.__abstractmethods__` is empty and `add_contrast`
   is present — `EventModel` is fully instantiable. mypy sees an
   abstract `add_contrast` via a Protocol/base it declares
   conformance to while the concrete class provides it. A
   type-declaration/Protocol-shape question for the gate
   decision, not a latent bug.

3. **`utils/event_utils.py:257` — `Unexpected keyword argument
   "name"/"onsets" for "EventProtocol"` [VERIFIED ARTIFACT, not a
   bug].** The line is `type(event)(name=..., onsets=...)` where
   `event` is `EventProtocol`-typed, so mypy infers
   `type[EventProtocol]` and rejects the kwargs (Protocols declare
   no such `__init__`). At runtime `type(event)` is the *concrete*
   event class (EventFactor/Variable/…), which accepts them; the 19
   `event_util` tests pass. Same Protocol-typed-variable artifact
   family as #2.

4. **`convolve.py:815` — `Too many arguments for "convolve"`
   [VERIFIED ARTIFACT, not a bug].** `convolve` is a
   `@singledispatch` generic; `_convolve_list` is its
   `@convolve.register(list)` handler and recurses
   `convolve(event, ...)` per element. mypy type-checks that
   inner call against the *base* generic signature
   (4 positional + `**kwargs`), not knowing singledispatch routes
   real event instances to registered handlers with compatible
   signatures. Runtime check:
   `convolve([EventVariable, EventVariable], sampling_rate=1.0,
   total_duration=20.0)` returns correctly. An earlier provisional
   "possible arity bug" note here was wrong — it came from a repro
   that used a fake event type, which singledispatch could not
   route, falling to the base signature. Corrected in place per
   board rule (wrong claims fixed as part of the work).

5. **`glm/fmri_lm.py:783` — `Missing positional argument "x" in
   call to "contrast_weights"` [likely dispatch artifact].**
   `contrast_weights` is a singledispatch generic; the bound-method
   call shape confuses strict resolution. Same family as #2/#4 —
   probably not a runtime bug but not yet runtime-verified.

6. **`regressors.py:150` — `"add" of "set" does not return a value`
   [NOT a bug].** `[i for i in xs if not (i in seen or seen.add(i))]`
   is the standard dedup-preserving-order idiom and is correct at
   runtime. Strict flags the `set.add()`-in-boolean idiom; the fix
   is cosmetic (rewrite the idiom), not a behavior change.

7. **Second real bug [FIXED, bd-01KRNPWCVX59Q4QW027XGW7SGT].**
   Surfaced *because* #1's fix unblocked the path: `robust/irls.py`
   had a half-applied fix — a broken `X_w = X_r * w_sqrt`
   `(n,p)*(n,V)` line (with its own "won't work" comment) executed
   *before* the correct row-weight path that was already written
   below it, so the runwise robust IRLS refit raised
   `ValueError: operands could not be broadcast together`. fmri_rlm
   (and any runwise `RobustOptions(type=…)` fit) crashed the moment
   it reached the refit — masked only because #1 prevented
   fmri_rlm from getting that far, and direct runwise-robust paths
   were untested. Fix: removed the dead broken line and the
   redundant overwritten `Y_w` assignment, leaving the intended
   row-mean-WLS approximation (`w_row = mean(w_sqrt, axis=1)`
   applied to both X and Y). Regression test
   `test_fmri_rlm_completes_runwise_robust_irls_fit` (parametrized
   1- and 2-run) added; proven to FAIL on the pre-fix `irls.py`
   (git-stash check) and pass after, asserting correct-shape finite
   betas — not cheap-passable by xfail/skip/stub.

   **Revised yield:** the full-strict pass directly found **one**
   real bug (#1); fixing it then *unblocked* a **second** real bug
   (#7) that no static check could have surfaced while #1 masked
   the path. Two real bugs total, both in the robust-GLM compat
   surface, both now fixed + regression-tested. This strengthens
   the steward input: the robust-fitting compat path was
   end-to-end broken and wholly untested; the gate question aside,
   that surface now has its first real coverage.

## Post-reduction state (2026-05-15)

The mechanical + boundary-cast buckets were worked to completion in a
per-file rhythm: **104 files cleared, full-strict 746 → 100 (646
cleared, ~87%)**, the scoped `--follow-imports=skip` gate green at
every commit, both inventory artifacts (`internal_any_audit.json`,
`api_inventory_v1.json`) regenerated per commit, every commit
error-file-set-diffed so nothing regressed into already-clean files.
Final full non-rpy2 suite: **3007 passed / 0 failed**.

A structural improvement, not just suppression: typing
`BaseEvent.__post_init__` and `CacheMixin.__init__` in `base.py`
(commit 1e46615) **permanently removed the `no-untyped-call` leg from
the entire scoped/strict divergence cluster** — under full-strict the
subclass `self.__post_init__()` / `super().__init__()` calls are now
typed. That leg is *resolved*, not reserved; the cluster comments in
the events/ + covariate subclasses were reconciled (49ad609) so the
documented divergence no longer over-claims.

### The remaining 100 errors — none mechanically fixable

| Class | Files / counts | Why reserved |
| --- | --- | --- |
| **Steward-reserved EventModel divergence** (72) | `design/event_model.py` 60, `hrf_dispatch.py` 12 | The singledispatch/Protocol-vs-abstract artifact (findings #2/#4/#5 family, runtime-verified NOT bugs). Resolving needs the gate choice + possibly a typed EventModel protocol seam — S2, GOVERNANCE §7/§9. |
| **In-flight per CLAUDE.md** (9) | `spec/_compile.py` 7, `dataset/adapters/neuroim_adapter.py` 2 | Moving targets; CLAUDE.md says treat as such until they land thematically. Not touched. |
| **Documented scoped-required-cast sub-cluster** (~8) | `events/basis.py:160`, `group/dataset.py:113/118`, `utils/misc.py`, `regressor/core.py`, `group/ops.py`, `glm/fmri_lm.py:428`, `dispatch.py` | Each verified *empirically*: removing the cast turns the **scoped** gate red (`no-any-return`, opaque-Any base under `--follow-imports=skip`); full-strict flags the same cast `redundant-cast`. No annotation satisfies both gates. Documented inline + on the bead; the cast stays because scoped is the epic gate. |
| **Documented `[misc]` unused-ignore cluster** (~7) | `events/term|variable|matrix|factor|basis.py`, `covariate.py` (CovariateTerm/CovariateEvent), 1 in `event_model.py` | Under scoped the opaque-Any base *requires* the `# type: ignore[misc]` on `class X(Base)`; full-strict resolves the base and flags it `unused-ignore`. Same irreducible either/or. |
| **`covariate.py` event_type taxonomy** (1 `override`) | `covariate.py` | `CovariateEvent.event_type` returns `"covariate"`, outside `EventType = Literal["categorical","continuous","basis","matrix"]`. A genuine LSP/taxonomy *design* question (extend the Literal vs. CovariateEvent is not a true EventType), independent of the gate choice. |
| **PEP 562 limitation** (1 `no-untyped-def`) | `fmrimod/__init__.py` | A module `__getattr__` cannot be cleanly annotated under strict: `-> Any` propagates `Any?` into every `from fmrimod import X` importer and reintroduces errors in clean modules (verified: glm/compat.py, dataset/compat.py); `-> object` breaks callable re-exports. The lone `no-untyped-def` is strictly cheaper than that cascade; documented inline. |

### One regression introduced and fixed during the reduction

`events/factor.py` commit 10f3938 rewrote `from_dataframe`'s
`values=df[value_col].values` to `values=np.asarray(...)`, silently
stripping a pandas `Categorical`'s declared level order and breaking
semantic-contrast categorical-ordering parity (4 tests). Caught by a
robust `git bisect` (skip-on-collection-error oracle; the naive
bisect mislocalized), root-caused, fixed in 6631eb5 with
`cast(Any, ...)` (runtime-identity, unlike `np.asarray`). Recorded
because it sharpens the steward input: `np.asarray(...)` is **not**
a type-inert substitute on pandas `.values` — a lesson the gate
decision's tooling guidance should carry. This was a
mechanical-work-introduced regression, not a full-strict *find*; the
direct real-bug yield remains #1/#7 (two bugs, robust-GLM compat).

## Recommendation (SUPERSEDED by decision 0002 — retained as reasoning record)

> The steward enacted **full-strict as authoritative** on 2026-05-16.
> The canonical rule is
> [`decisions/0002-full-strict-is-the-authoritative-type-gate.md`](decisions/0002-full-strict-is-the-authoritative-type-gate.md).
> The text below is the pre-decision reasoning, kept for the audit
> trail; it is no longer the operative rule.

The decision is *which gate is authoritative*, and it is reserved
(adding `follow_imports` to `pyproject` weakens the committed strict
contract — a §10/§8 config change). This inventory supports either
path:

- **If scoped stays the gate:** add a `CAVEATS.md` row documenting
  the scoped/strict divergence with this file as the substrate, and
  teach the audit harness that scoped-green ≠ strict-green so the
  epic does not over-claim "typed".
- **If full-strict becomes the gate:** the ~715 mechanical +
  boundary-cast errors are *already cleared* (see §Post-reduction —
  this is no longer a future cost). What remains is the ~19
  documented scoped/strict divergence residue + the 72
  EventModel/hrf_dispatch artifact errors; under a full-strict gate
  those casts/ignores flip from required to redundant and can be
  removed, but the EventModel artifact cluster still needs the typed
  protocol-seam call (S2). The judgment items were triaged to
  completion (two real bugs, #1/#7, fixed).

## Verified real-bug yield (the number that matters for the decision)

Triage of the judgment/likely-real items found **two real
runtime bugs**, both now fixed + regression-tested, both in the
robust-GLM compat surface: #1 (`fmri_rlm` config TypeError) which
the full-strict pass found directly, and #7 (runwise robust IRLS
broadcast) which fixing #1 then *unblocked* — #7 could not have
been surfaced by any static check while #1 masked the path.
Findings #2, #3 and #4 were runtime-verified as
**singledispatch/Protocol typing artifacts, not bugs**
(`EventModel` is instantiable; `type(event)(...)` on a
Protocol-typed var hits the concrete class at runtime;
`convolve(list, total_duration=...)` works with real events). #5
is the same artifact family (low-suspicion, not separately
verified — the pattern is well-established). #6 is a correct
idiom flagged cosmetically. Judgment triage is complete: no
unverified judgment items remain and no open follow-up bead
remains (#7 closed).

Implication for the gate decision: full-strict's incremental
*direct* bug-finding value over the scoped gate was ~1 real bug
per ~750 errors — the rest mechanical hygiene plus
dispatch/Protocol artifacts `--follow-imports=skip` structurally
cannot surface. But the decision input is stronger than that
ratio suggests: the one bug full-strict found gated a *second*
real bug behind it, and both sat in an end-to-end-broken,
wholly-untested public robust-fitting path. Full-strict is not a
large hidden-bug reservoir here, yet the cluster it did expose
was a complete dead public surface. The steward call is whether
that justifies adopting full-strict as the gate vs. documenting
the scoped/strict divergence as a caveat; either way the
robust-GLM compat surface now has its first real coverage and no
work is left dangling outside the gate decision itself.
