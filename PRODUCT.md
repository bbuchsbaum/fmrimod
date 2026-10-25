# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

fMRI researchers and methods developers learning to specify, fit, and inspect
statistical models in Python. The audience is described in MISSION.md.

## Product Purpose

The documentation teaches the path from dataset and design to first-level
fit, contrast, and group analysis. Readers should be able to run the examples,
understand the modeling choices, and inspect the supporting evidence.

## Operating Context

Quarto renders executable Python tutorials and a curated quartodoc API
reference. The package supports Python 3.10 and newer. neuroim-python supplies
spatial containers. Existing source and rendered-artifact audits check tutorial
contracts, links, and image alternatives.

## Capabilities and Constraints

The package covers HRFs, event designs, GLMs, contrasts, single-trial estimates,
and group analyses. It consumes preprocessed data. Simulated examples and
reference comparisons support different claims and must be identified as such.
Active numerical differences remain documented in docs/contracts/CAVEATS.md.

## Brand Commitments

The user requested current documentation, prose edited with judgment, a full
visual redesign with less SaaS styling, and prominent scientific plots.
Preserve the fmrimod name, executable examples, and honest scientific scope.

## Evidence on Hand

The FIAC tutorial and brain montage, deterministic simulation tutorials,
benchmark reports, numerical tests, and source docstrings. Newer APIs include
smooth FIR HRF estimation and native group examination. Scientific figures
must come from executed code, with simulation identified in their captions.

## Documentation Direction

The user confirmed that the site should generally showcase model inspection
capabilities and chose direct implementation in Quarto with browser review.
