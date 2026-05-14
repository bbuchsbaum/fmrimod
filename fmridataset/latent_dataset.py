"""Compatibility re-exports for latent datasets."""

from fmrimod.dataset.latent import (
    LatentDataset,
    get_component_info,
    get_latent_scores,
    get_spatial_loadings,
    latent_dataset,
)

__all__ = [
    "LatentDataset",
    "get_component_info",
    "get_latent_scores",
    "get_spatial_loadings",
    "latent_dataset",
]
