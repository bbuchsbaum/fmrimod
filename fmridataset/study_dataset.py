"""Compatibility facade for study-level dataset containers."""

from fmrimod.dataset.study import (
    StudyDataset,
    fmri_group,
    study_dataset,
    study_to_group,
)

__all__ = ["StudyDataset", "fmri_group", "study_dataset", "study_to_group"]
