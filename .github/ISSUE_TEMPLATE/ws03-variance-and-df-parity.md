---
name: WS03 Variance and df parity
about: Track residual variance and degrees-of-freedom parity for OLS and AR paths.
title: "[WS03] Variance and df parity"
labels: ["parity", "workstream", "ws03"]
assignees: []
---

## Objective
Verify sigma2 and df accounting across OLS and AR-corrected execution paths.

## Scope
- [ ] Add sigma2/df parity suites.
- [ ] Add contract tests for df behavior.

## Acceptance Criteria
- [ ] sigma2-corr >= 0.995.
- [ ] sigma2-MAE <= 0.02.
- [ ] df contracts pass.

## Artifacts
- [ ] Report path:

## CI Gate
- [ ] PR quick gate updated.
- [ ] Nightly strict gate updated.
