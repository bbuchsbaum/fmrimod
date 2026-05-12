---
name: WS01 Design matrix construction parity
about: Track design matrix parity work against fitlins/nilearn references.
title: "[WS01] Design matrix construction parity"
labels: ["parity", "workstream", "ws01"]
assignees: []
---

## Objective
Implement and validate design matrix parity (events, durations, amplitudes, HRF basis/derivatives, drifts/confounds).

## Scope
- [ ] Add fixture coverage for required design variants.
- [ ] Add parity comparator and artifact output.

## Acceptance Criteria
- [ ] Column-wise correlation >= 0.999 or documented transformed equivalence.
- [ ] Max absolute deviation within tolerance envelope.

## Artifacts
- [ ] Report path:
- [ ] Failing-case fixtures (if any):

## CI Gate
- [ ] PR quick gate updated.
- [ ] Nightly strict gate updated.
