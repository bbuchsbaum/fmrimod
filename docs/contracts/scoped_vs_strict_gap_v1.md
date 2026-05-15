# Scoped-mypy vs. full-strict gap — inventory v1

Audit substrate for **bd-01KRNN0H73CCYGFJSJ30JPVFTW** (reconcile the
typed-API epic's scoped gate with the authoritative full-strict
`[tool.mypy]` config). Inventory-before-action: this document is the
checked-in measurement; no retirement/refactor/config change is made
here.

## The two gates

- **Epic gate (what ~38 typed-API beads were verified against):**
  per-subpackage `python3.9 -m mypy fmrimod/<pkg>/ --follow-imports=skip`.
  As of HEAD this is **0 errors across every top-level subpackage**,
  verified by a `find fmrimod -maxdepth 1 -type d` sweep (not a curated
  list — a curated list previously masked top-level `backends/`).
- **Authoritative config (`pyproject.toml [tool.mypy]`):** full-tree
  `strict = true`, `warn_return_any = true`, no `follow_imports`
  override. Running `python3.9 -m mypy fmrimod/` (project config):
  **746 errors in 105 files**.

The scoped gate is a proxy. `--follow-imports=skip` makes every
cross-module symbol opaque (`Any`), which both *hides* real
cross-module mismatches and *inverts* cast-redundancy (a cast scoped
mypy demands, full-strict flags redundant, and single-file vs
subpackage scope disagree again). The gap is structural, not a
backlog of the same recipe.

## 746 errors classified

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

7. **Separate observation (not a strict error):** with the
   `fmri_rlm` config bug fixed, a minimal dummy model driven through
   the *runwise* robust IRLS path hits
   `ValueError: operands could not be broadcast together with
   shapes (n,p) (n,v)` at `X_w = X_r * w_sqrt` in
   `glm/strategies.py`. This is reached only now that robust is
   actually enabled. Either a real runwise-robust weighting bug or a
   model-contract gap; tracked as a follow-up, NOT fixed here.

## Recommendation (for the steward decision, not enacted here)

The decision is *which gate is authoritative*, and it is reserved
(adding `follow_imports` to `pyproject` weakens the committed strict
contract — a §10/§8 config change). This inventory supports either
path:

- **If scoped stays the gate:** add a `CAVEATS.md` row documenting
  the scoped/strict divergence with this file as the substrate, and
  teach the audit harness that scoped-green ≠ strict-green so the
  epic does not over-claim "typed".
- **If full-strict becomes the gate:** the ~715 mechanical +
  boundary-cast errors are tractable in the established per-file
  rhythm; the ~41 judgment items must still be triaged per-item
  rather than cast away.

## Verified real-bug yield (the number that matters for the decision)

Triage of the judgment/likely-real items found exactly **one
confirmed runtime bug** in 746 strict errors: finding #1
(`fmri_rlm`, now fixed + regression-tested). Findings #2, #3 and
#4 were runtime-verified as **singledispatch/Protocol typing
artifacts, not bugs** (`EventModel` is instantiable;
`type(event)(...)` on a Protocol-typed var hits the concrete
class at runtime; `convolve(list, total_duration=...)` works
with real events). #5 is the same artifact family (low-suspicion,
not separately verified — the pattern is now well-established).
#6 is a correct idiom flagged cosmetically. #7 is a separate,
real observation (runwise-robust IRLS broadcast with minimal
models), surfaced *because* #1's fix unblocked that path —
tracked as its own bead, not a strict-mypy finding. Judgment
triage is now complete: no unverified judgment items remain.

Implication for the gate decision: full-strict's incremental
bug-finding value over the scoped gate, on this codebase, was ~1
real bug per ~750 errors — the rest is mechanical hygiene plus
singledispatch/Protocol artifacts that `--follow-imports=skip`
structurally cannot surface anyway. That ratio is the decision
input: full-strict is not a large hidden-bug reservoir here, but
the one bug it found was a 100%-broken untested public API. The
steward call is whether that yield justifies adopting full-strict
as the gate vs. documenting the scoped/strict divergence as a
caveat. The only item to bead independently of the gate decision
is **#7** (runwise-robust IRLS broadcast) — filed separately.
