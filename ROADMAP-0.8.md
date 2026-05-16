# ROADMAP to 0.8

Status: active release contract.

Owner mote: `bd-01KRRVEP2RT4811CJCK9HQCCFW`

Policy steward: user-delegated to the active agent for this goal on
2026-05-16. Steward decisions in this file are scoped to getting a credible
0.8 release candidate; they do not redefine the 1.0 bar.

## Thesis

fmrimod 0.8 is the typed first-level seam preview. It should be release-quality
for a narrow claim:

```text
fmri_dataset -> fmri_lm(spec, dataset) -> contrast
```

The release demonstrates that authored statistical intent can travel through
typed Python objects into a fit and a contrast without falling back to notebook
local arrays as the user-facing story. It also records the current functional
connectivity answer: plain seed-target correlation is an explicit estimator,
while GLM-style seed modeling is expressed as `covariate(...)`, an identity
HRF term, inside `fmri_lm`.

0.8 is not a miniature 1.0. It is the stable public slice that makes the 1.0
proof-bundle work credible.

## Steward Policy Decisions

1. **0.8 is allowed to ship before the full 1.0 proof bundle.**
   The release claim is the typed first-level seam, not all flagship families.

2. **A live caveat can be 0.8-compatible when it is indexed and principled.**
   `dfres-n-minus-rank` remains live under
   `bd-01KRHTASRWPA5ZQNGV55BS6XFE`. It is not a 0.8 blocker because the
   current policy is explicit: fmrimod uses `n - rank` for rank-deficient
   designs and will not regress to Nilearn's `n - p` behavior. The 0.8
   requirement is that this remains visible in `docs/contracts/CAVEATS.md`.

3. **Connectivity support is narrow by design.**
   fmrimod should own the ergonomic pain point, not pretend to be a complete
   connectivity toolbox. Seed-target Pearson correlation can be a transparent
   stats helper. Seed-as-regressor connectivity belongs in the GLM seam via
   `covariate(...)`.

4. **Native group inference is present but not claimed as the 0.8 flagship.**
   Group support can appear in docs and tests, but the release claim remains
   first-level modeling plus contrast.

5. **The release is blocked by enforceable gates, not by broad cleanup.**
   API inventory, full proof-bundle polish, optional dataset backends, and
   exhaustive BIDS transform coverage move to 0.9/1.0 unless they break the
   typed first-level seam.

## 0.8 Claims

0.8 claims support for:

- typed model declarations through `fmrimod.spec`;
- `hrf(...)` terms that lower to design regressors;
- `covariate(...)` as an identity/no-op HRF species for sampled time courses;
- `fmri_lm(spec, dataset)` as the first-level modeling entry point;
- contrast evaluation from the fitted public object;
- seed-target correlation as an explicit, readable estimator helper;
- GLM-style seed modeling through `fmri_lm(covariate(...), dataset)`;
- owned, indexed caveats for deliberate divergences;
- focused tests and CI around the release surface.

## 0.8 Non-Claims

0.8 does not claim:

- all inherited R exports are ported;
- the full `fmri_dataset -> fmri_lm -> contrast -> group_fit` path is the
  release flagship;
- the 1.0 proof bundle is complete;
- every BIDS Stats Model transform is covered;
- every optional dataset backend is production-ready;
- public API stability for names outside the first-level seam;
- fmrimod owns general graph, atlas, or whole-connectome estimation.

## Blocking Motes

Resolved blockers (closed; no longer gating):

| Mote | Role |
| --- | --- |
| `bd-01KRRT0ZNM8VQJA6T1E7FV5RRD` | Land connectivity code with the internal API audit regenerated. *(closed)* |

Live blocker (open; gates a 0.8 *tag*, not the RC):

| Mote | Role |
| --- | --- |
| `bd-01KRRT0ZEM0K909QK4HZCHJ8XH` | Enforcing CI for pytest + ruff/Python-floor guard. Workflow is committed (`.github/workflows/release-0-8.yml`) but the mote is `blocked`: GitHub Actions is billing-blocked, so the gate has **never executed**. Tracked alongside `bd-01KRS5FT6MPY191ADEV6XXPXNF`. |

These motes are policy conditions or immediate follow-through:

| Mote | Role |
| --- | --- |
| `bd-01KRHTASRWPA5ZQNGV55BS6XFE` | Live dfres caveat owner; must remain indexed and principled. |
| `bd-01KRRVGG9NP5PBWYZ6Q27NCSWW` | Document the typed covariate/connectivity golden path. |
| `bd-01KRRVEXBNG3DFP5RSW1X6J0E6` | Revise the roadmap into the 0.8 -> 0.9 -> 1.0 ladder. |

## Evidence Gates

The 0.8 candidate must pass the focused release-surface gates:

```bash
.venv/bin/python -m pytest \
  tests/test_stats/test_connectivity.py \
  tests/test_spec/test_spec.py \
  tests/test_spec/test_serialize.py \
  tests/test_glm/test_fmri_lm_spec_dataset.py \
  tests/design/test_covariate.py \
  tests/design/test_covariate_trialwise_comprehensive.py -q
```

It must also pass caveat discipline checks:

```bash
.venv/bin/python -m pytest \
  cross_testing/test_caveats_index.py \
  tests/test_benchmarks/test_parity_proof_artifacts.py -q
```

Before tagging, the broader documented suite should pass:

```bash
.venv/bin/python -m pytest tests/ -k "not rpy2"
```

The CI mote (`bd-01KRRT0ZEM0K909QK4HZCHJ8XH`) defines the automated gate. The
workflow is committed but has **not executed** — GitHub Actions is
billing-blocked (`bd-01KRS5FT6MPY191ADEV6XXPXNF`). Under the Path B release
posture, a green `.venv` run of the documented suite is the **interim proof**
that advances the release *candidate*; a 0.8 *tag* still requires one genuinely
green run of `.github/workflows/release-0-8.yml` on the tip.

## Acceptance Checklist

- [x] Connectivity helper code is committed with regenerated internal API
      audit data.
- [x] `covariate(...)` is exported, serialized where source-free, rejected
      where inline data would be ambiguous, and lowered through the identity
      HRF path.
- [x] Functional connectivity docs show both the explicit correlation helper
      and the GLM route through `fmri_lm(covariate(...), dataset)`.
- [x] `docs/contracts/CAVEATS.md` names `dfres-n-minus-rank` with owner and
      exit criterion.
- [x] CI gate for release-relevant pytest + ruff/Python-floor is defined and
      committed (`.github/workflows/release-0-8.yml`).
- [ ] CI gate has executed green on the tip — **blocked**: GitHub Actions
      billing (`bd-01KRRT0ZEM0K909QK4HZCHJ8XH` / `bd-01KRS5FT6MPY191ADEV6XXPXNF`).
      RC advances on interim green `.venv` local proof.
- [x] `ROADMAP-1.0.md` no longer reads as the immediate next-release plan.
- [x] Focused 0.8 gates pass locally.
- [x] The broader `tests/ -k "not rpy2"` suite is either green or has a
      documented, release-approved blocker.

## Path After 0.8

0.9 is the proof-bundle candidate. It should promote the 0.8 seam into the
flagship proof artifacts, finish the canonical native group release path, and
make `python -m benchmarks.parity.release_1_0_bundle` a meaningful release
receipt again.

1.0 is the artifact-as-argument release. It ships when the proof bundle is
hard to dismiss, not when the package has accumulated enough surface area.
