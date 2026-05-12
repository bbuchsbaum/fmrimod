---
name: WS05 Censor/sample-mask parity
about: Track censor/sample-mask parity and boundary safety.
title: "[WS05] Censor/sample-mask parity"
labels: ["parity", "workstream", "ws05"]
assignees: []
---

## Objective
Verify censor handling and segmentation correctness across run boundaries.

## Scope
- [ ] Add censor-sensitive fixtures.
- [ ] Add leakage checks at boundaries and gaps.

## Acceptance Criteria
- [ ] All censor fixtures pass.
- [ ] No boundary leakage regressions.

## Artifacts
- [ ] Report path:

## CI Gate
- [ ] PR quick gate updated.
- [ ] Nightly strict gate updated.
