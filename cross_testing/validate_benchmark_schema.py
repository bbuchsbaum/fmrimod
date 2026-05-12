#!/usr/bin/env python
"""Validate benchmark artifacts against canonical JSON schema."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from jsonschema import validate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate benchmark JSON artifact against schema."
    )
    parser.add_argument(
        "--artifact",
        type=str,
        required=True,
        help="Path to benchmark artifact JSON.",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="cross_testing/schemas/core_parity_benchmark.schema.json",
        help="Path to schema JSON.",
    )
    args = parser.parse_args(argv)

    with open(args.schema, "r", encoding="utf-8") as fobj:
        schema = json.load(fobj)
    with open(args.artifact, "r", encoding="utf-8") as fobj:
        artifact = json.load(fobj)

    validate(instance=artifact, schema=schema)
    print(f"schema validation passed: {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
