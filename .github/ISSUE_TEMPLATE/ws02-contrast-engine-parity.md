---
name: WS02 Contrast engine parity
about: Track t/F contrast parity for effect, variance, and inference outputs.
title: "[WS02] Contrast engine parity"
labels: ["parity", "workstream", "ws02"]
assignees: []
---

## Objective
Validate t/F contrasts for single-column and multi-column cases.

## Scope
- [ ] Add contrast parity suites and fixtures.
- [ ] Emit JSON artifact with contrast metrics.

## Acceptance Criteria
- [ ] t-corr >= 0.995.
- [ ] p-MAE <= 0.035.

## Artifacts
- [ ] Report path:

## CI Gate
- [ ] PR quick gate updated.
- [ ] Nightly strict gate updated.
