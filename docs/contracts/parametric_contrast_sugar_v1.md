# Parametric Contrast Sugar and ContrastResult Group Collection v1

Status: Vetting

Owners:

- `bd-01KRM9PVWWKTH7A0TJDYTZ9XB7` — consolidated follow-up with the red check.
- `bd-01KRM5FRQP16S05T3DA57DQY31` — broader parametric semantic sugar gap.
- `bd-01KRFMD3JFKMH2ETGPRRAWAFME` — adjacent group typed-API shape.

Board source:
`parametric-contrast-sugar-vetting/post-01KRM9ZH9NPK4AGJ7GFF2SK8NQ` and the
consensus candidate at
`parametric-contrast-sugar-vetting/post-01KRMAB8WF8C5TVS963SBC83BK`.

## Why this exists

`benchmarks/parity/tier_e_bids_parametric_group_followthrough/workflow.py`
currently downgrades to `ux_status=typed_contrast_full_group_manual` /
`e2e_ux_status=partial` because two surfaces leak internal shapes into the
showcased user code:

1. Parametric slope authoring spells the internal column-naming convention
   `term="trial_type:rt_z"` at workflow.py:194.
2. Group collection manually packs `ContrastResult` estimates into a
   `subject/feature/beta/se` `pd.DataFrame` at workflow.py:213 before
   `group_data_from_csv(...)` consumes it.

Both are *lowerings*, not *coercions*, in the sense of MISSION baseline #1.
The user code should never name the internal column algebra or the group
row schema.

## Source of Truth

- Tier E benchmark: `benchmarks/parity/tier_e_bids_parametric_group_followthrough/`.
- Existing semantic contrast surface: `fmrimod.contrast.condition`
  (`fmrimod/contrast/semantic.py:255`) — the v1 sugar lives next to this.
- Existing typed result substrate: `fmrimod.glm.contrasts.ContrastResult`
  (`fmrimod/glm/contrasts.py:325`) and `SpatialContext`, `ContrastIntent`.
- Existing group substrate: `fmrimod.dataset.group_data_from_csv`,
  `fmrimod.group.group_dataset_from_group_data` (the typed lowering target).

## Q1–Q5 v1 Candidate

Q1, Q2, Q3, Q4, Q5 below correspond to the five questions raised in
`parametric-contrast-sugar-vetting/post-01KRM9ZH9NPK4AGJ7GFF2SK8NQ`.

### Q1: Naming and call shape

Adopt `modulator` over `parametric`. Rationale: `modulator` matches the
design-domain noun already used in `HrfTerm.modulators` and avoids
overloading the BIDS Stats Model `parametric` verb (which lives at the
BIDS transform layer, a different stage of the spine).

The v1 t-contrast call shape is:

```python
from fmrimod.contrast import modulator

rt = modulator("rt_z").within("trial_type")
slope_spec = rt.slope("word") - rt.slope("pseudoword")
```

`rt.slope(level)` lowers to a t-contrast over the single parametric column
that the level + modulator pair generates. The `-` is ordinary
`ContrastSpec` arithmetic. This deliberately rejects `.difference(a, b)` as
a single verb: collapsing the level-pair and the contrast operation into
one method invites misreading the result as a difference of condition
means rather than a difference of slopes.

### Q2: Module path

Land at `fmrimod.contrast.modulator`, next to `fmrimod.contrast.condition`.

Do *not* promote to top-level `fm.modulator` in v1. Top-level promotion
must wait until a second receipt demonstrates the sugar belongs in the
public spine. Premature promotion would re-introduce the namespace shadow
problem owned by `bd-01KRHTHY5309X3VZT515S9E3H3`.

Do *not* add the sugar as a method on `fit` (`fit.modulator(...)`).
Contrast intent must be expressible against a `Spec` without a `fit` — the
BIDS translator's `compute_contrasts` path already depends on this.

### Q3: Error policy

When the requested modulator or level is missing or ambiguous, the raised
exception's headline message must:

1. Name the missing/ambiguous identifier and the requested factor/term.
2. List the available modulators (or available levels when the term is
   valid but the level is not).
3. Use difflib-style "did you mean" suggestions when the requested name is
   close to one in the available set.

Generated internal column names (the `term:modulator` shape) must appear
only in the exception's `__repr__` / structured details, not in the
headline. Headline real estate is for the actionable name; column-name
diagnostics are for tracebacks.

### Q4: Group collection

v1 ships a single-contrast-per-subject constructor:

```python
from fmrimod.contrast import group_dataset_from_contrasts

group_dataset = group_dataset_from_contrasts(
    {"sub-01": slope_01, "sub-02": slope_02, "sub-03": slope_03},
    covariates=covariates,
)
```

The dict maps subject identifier to one `ContrastResult`. The constructor
must lower to the same `GroupData`/`GroupDataset` substrate currently
produced by `group_data_from_csv(...)`, so the typed and CSV paths never
diverge in the second-level inference layer.

Multi-contrast-per-subject collection (a richer
`(subject_id, contrast_name, ContrastResult)` shape or a
`ContrastResultCollection` builder) is deferred to v2. Do *not* force the
multi-contrast complexity into v1's acceptance target.

### Q5: Non-negotiable metadata

Group collection must preserve, for each `ContrastResult` it ingests, the
following `ContrastResult` fields named in `fmrimod/glm/contrasts.py:348-359`:

- `name` — required.
- `estimate` — required.
- `se` — required (t-contrasts; F-contrast handling deferred to v2).
- `intent` — required. Carries the `ContrastIntent` provenance link back
  to the design term / BIDS Stats Model contrast that authored it.
- `touched_columns` — required. Names the design columns the contrast
  weights touched; this is what makes "design intent survives to group"
  inspectable on the second-level side.
- `touched_column_details` — required where present.
- `spatial: SpatialContext | None` — required when present. When
  `spatial` is not `None`, group collection must *error* if subjects
  carry incompatible spatial contexts (different masks, different image
  geometries, mixed array/image roots). Silent dropping of `spatial` is
  forbidden because the image-out path of VISION ("design intent
  survives to group") depends on it.
- `caveats` — required to be carried through; group collection must not
  silently drop caveat strings, and the group fit's report must surface
  the union of subject-level caveats.

Subject id, contrast name, feature/voxel id, and estimate together
disambiguate each row. The `intent` + `touched_columns` +
`touched_column_details` triple is the provenance receipt that makes the
acceptance target's `ux_status=typed_contrast_full_group` honest rather
than cosmetic.

## What this v1 deliberately defers

- **F-contrast spelling.** `rt.slope("word") - rt.slope("pseudoword")` is a
  t-contrast. The F-contrast analog (e.g. omnibus slope difference across
  three or more levels) is *not* shipped in v1. Two candidate v2 spellings
  are recorded here so the API does not get painted into a t-only corner:
  - `modulator("rt_z").within("trial_type").slopes("word", "pseudoword", "neutral")`
    returning a multi-row spec that lowers to F, parallel to
    `modulator(...).within(...).slope(level)` lowering to t.
  - `modulator("rt_z").within("trial_type").omnibus(*levels)` as an
    explicit verb.
  v1 must raise a clear `NotImplementedError` with a pointer to this
  contract when an F-contrast spelling is requested. v2 selects between
  the two candidates with a follow-up board post.

- **Multi-contrast group collection.** See Q4. v1 is dict-keyed by
  subject and accepts a single contrast result per subject. v2 owns the
  richer collection shape.

- **Top-level `fm.modulator` promotion.** See Q2.

## Acceptance / Red Check for `bd-01KRM9PVWWKTH7A0TJDYTZ9XB7`

The bead may close only when all of the following are true:

1. `fmrimod.contrast.modulator` exists with the v1 call shape above, and
   `from fmrimod.contrast import modulator` is exported.
2. `fmrimod.contrast.group_dataset_from_contrasts` exists with the v1
   single-contrast-per-subject dict signature above, and the metadata
   contract from Q5 is enforced (test that asserts each required field is
   round-trippable through the lowering).
3. `benchmarks/parity/tier_e_bids_parametric_group_followthrough/workflow.py`
   is rewritten so the showcased user-code path contains *no* mention of
   `term="trial_type:rt_z"` and *no* manual `subject/feature/beta/se`
   row packing.
4. The rewritten workflow's `proof_artifact` records
   `ux_status=typed_contrast_full_group` and `e2e_ux_status=full`.
5. The rewritten workflow produces numerically equivalent group-level
   estimates to the current `condition(..., term="trial_type:rt_z")` +
   manual-packing baseline, within the existing parity tolerance for
   tier_e.
6. `condition("word", term="trial_type:rt_z") - condition("pseudoword",
   term="trial_type:rt_z")` still works as a documented low-level escape
   hatch (additive sugar, not replacement). A unit test that exercises
   the low-level path remains green.

A failing-but-skipped test fixture in `tests/test_contrast/` that imports
the v1 call sites (`modulator`, `group_dataset_from_contrasts`) and
asserts the call-shape compiles is the executable artifact of this
contract. It lands before implementation and goes from skipped to
required as the implementation slice closes.

## Co-sign discipline

Every post in the source thread is from the default actor
`claude-typed-api`. Per `CLAUDE.md` § "Co-sign discipline — and its
structural weakness", same-actor co-sign is structurally weak in this
repo's current identity setup. This v1 candidate moves to **Active**
only after one of:

- An explicit co-sign reply from an actor other than `claude-typed-api`
  on `parametric-contrast-sugar-vetting/post-01KRMAB8WF8C5TVS963SBC83BK`
  or this contract's vetting status, OR
- 24 hours of silence on the source thread following the appearance of
  the failing test fixture artifact, with no substantive objection.

Either condition gates the bead status change from `vetting` to
`in_progress`. Implementation that lands ahead of the gate must be
reverted or have its bead reopened with `status=vetting`.
