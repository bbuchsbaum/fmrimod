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
   `Cannot instantiate abstract class "EventModel" with abstract
   attribute "add_contrast"` [needs decision].** Either `EventModel`
   should concretely implement `add_contrast`, or the
   protocol/ABC declaring it abstract is mis-shaped. This is a
   four-stage-seam / protocol-shape question (§7/§9 steward
   territory), not a mechanical cast.

3. **`utils/event_utils.py:257` — `Unexpected keyword argument
   "name"/"onsets" for "EventProtocol"` [needs check].**
   `EventProtocol(...)` constructed with kwargs the protocol does
   not declare; either the protocol is under-specified or the call
   site targets the wrong type.

4. **`convolve.py:815` — `Too many arguments for "convolve"`
   [needs check].** Possible real arity bug or a singledispatch
   resolution artifact; not yet root-caused.

5. **`glm/fmri_lm.py:783` — `Missing positional argument "x" in
   call to "contrast_weights"` [likely dispatch artifact].**
   `contrast_weights` is a singledispatch generic; the bound-method
   call shape confuses strict resolution. Probably not a runtime
   bug but needs confirmation.

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
  rhythm; the ~31 judgment items (findings 2–5 above) must be
  triaged as their own beads first (regression/test/doc, or a
  protocol-shape decision note per §8), per the board-to-bead rule.

Either way, findings 2–5 and the runwise-robust broadcast (7) are
real and should be beaded independently of the gate decision.
