#!/usr/bin/env python
"""Find words and phrases over-represented in generated prose.

Compares the n-gram frequencies of this repo's documentation against a
reference corpus of human-written prose, and reports what the generated side
uses far more often. The output is a *candidate* list for human review, not a
ban list: the two corpora differ in subject matter as well as in voice, so
domain nouns will always be over-represented for legitimate reasons.

The method is borrowed from Sam Paech's slop-forensics
(https://github.com/sam-paech/slop-forensics), minus the model-lineage and
fine-tuning machinery, which needs local weights and a GPU. Only the
diagnostic half transfers to prose written through an API model.

Why this rather than asking a model what sounds generated: a critic sharing
the writer's weights shares the writer's blind spots. Frequency against human
text is an outside measurement, so it can surface habits neither party would
think to look for.

Usage
-----
    python scripts/prose_corpus_diff.py \
        --reference ~/code/fmrireg ~/code/neuroim2 ~/code/fmrihrf \
        --target docs README.md \
        --top 40
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# Prose lives between the code. Strip anything that is not someone writing.
_FENCED_CODE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`]*`")
_YAML_HEADER = re.compile(r"\A---\n.*?\n---\n", re.S)
_DIV_FENCE = re.compile(r"^:::.*$", re.M)
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|\]\([^)]*\)")
_ROXYGEN = re.compile(r"^#'\s?", re.M)
_RD_TAG = re.compile(r"@\w+")
_NON_WORD = re.compile(r"[^a-z\s-]")
# Tables are data, not voice. Left in, an inventory of module names swamps
# every real signal -- the first run of this script returned "name type
# description" as a top bigram.
_TABLE_ROW = re.compile(r"^\s*\|.*$", re.M)
_TABLE_RULE = re.compile(r"^\s*[-: ]+\|[-| :]*$", re.M)
_INDENTED_BLOCK = re.compile(r"^(?: {4,}|\t).*$", re.M)

# Closed-class words carry no stylistic signal; they only add noise to the
# ratios. Deliberately short -- content words are the point.
_STOPWORDS = frozenset(
    """
a an the and or but if then than that this these those there here
is are was were be been being am do does did doing done
have has had having will would shall should can could may might must
of in on at to for from with without by as into onto over under
it its it's we you your our they them their he she his her i
not no nor so such only just also both each few more most other some
same too very s t don now which who whom what when where why how
one two three all any because before after above below up down out off
about against between through during again further once
""".split()  # noqa: SIM905  (readability: hand-edited word list)
)

# Words that are over-represented because this repo is about them, not
# because of how it is written. Extend as needed.
_DOMAIN = frozenset(
    """
fmrimod fmrireg fmrihrf fmridesign fmrilss fmriar fmrigds fmridataset
neuroim hrf hrfs glm ols ar1 bold fmri mri voxel voxels vox tr
regressor regressors design designs matrix matrices contrast contrasts
event events onset onsets duration durations basis bases spline splines
beta betas dataset datasets model models fit fits spec specs term terms
python r nilearn fitlins bids numpy pandas scipy api
convolution convolve convolved sampling frame block blocks run runs
subject subjects group trial trials condition conditions baseline drift
""".split()  # noqa: SIM905  (readability: hand-edited word list)
)


def _strip_to_prose(text: str, *, roxygen: bool) -> str:
    """Reduce a source document to the sentences a person wrote."""
    if roxygen:
        # Only roxygen comment bodies are prose in an .R file.
        text = "\n".join(
            line for line in text.splitlines() if line.lstrip().startswith("#'")
        )
        text = _ROXYGEN.sub("", text)
        text = _RD_TAG.sub(" ", text)
    text = _YAML_HEADER.sub("", text)
    text = _FENCED_CODE.sub(" ", text)
    text = _INDENTED_BLOCK.sub(" ", text)
    text = _TABLE_RULE.sub(" ", text)
    text = _TABLE_ROW.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _DIV_FENCE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    return text


def _tokens(text: str) -> list[str]:
    text = _NON_WORD.sub(" ", text.lower())
    return [w for w in text.split() if len(w) > 2 and w not in _STOPWORDS]


def _collect(paths: list[Path]) -> tuple[list[str], int, int]:
    """Return (tokens, n_files, n_words) over every prose file under *paths*."""
    tokens: list[str] = []
    n_files = 0
    for root in paths:
        if not root.exists():
            print(f"  ! missing, skipped: {root}", file=sys.stderr)
            continue
        candidates = (
            [root]
            if root.is_file()
            else [
                p for pat in ("*.qmd", "*.Rmd", "*.md", "*.R") for p in root.rglob(pat)
            ]
        )
        for path in candidates:
            parts = set(path.parts)
            if parts & {"_site", "_freeze", ".quarto", "vendor", "node_modules"}:
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            prose = _strip_to_prose(raw, roxygen=path.suffix == ".R")
            got = _tokens(prose)
            if got:
                tokens.extend(got)
                n_files += 1
    return tokens, n_files, len(tokens)


def _ngrams(tokens: list[str], n: int) -> Counter:
    if n == 1:
        return Counter(tokens)
    return Counter(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _over_represented(
    target: list[str], reference: list[str], n: int, *, top: int, min_count: int
) -> list[tuple[str, float, int, float, float]]:
    """Rank n-grams by how much more often the target uses them.

    Rate is per 10k tokens. A smoothing constant stands in for the reference
    rate of terms that never appear there, so a term absent from human prose
    does not produce an infinite ratio.
    """
    t_counts, r_counts = _ngrams(target, n), _ngrams(reference, n)
    t_total, r_total = max(len(target), 1), max(len(reference), 1)
    smoothing = 10_000 / r_total  # one hypothetical occurrence

    rows = []
    for gram, count in t_counts.items():
        if count < min_count:
            continue
        if any(w in _DOMAIN for w in gram.split()):
            continue
        t_rate = count * 10_000 / t_total
        r_rate = r_counts.get(gram, 0) * 10_000 / r_total
        ratio = t_rate / (r_rate if r_rate > 0 else smoothing)
        if ratio <= 1.0:
            continue
        rows.append((gram, ratio, count, t_rate, r_rate))

    # Rank by ratio weighted by how often it actually occurs: a term used
    # twice at 50x is less interesting than one used forty times at 8x.
    rows.sort(key=lambda r: r[1] * (r[2] ** 0.5), reverse=True)
    return rows[:top]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reference",
        nargs="+",
        type=Path,
        required=True,
        help="Directories/files of human-written prose (the baseline voice).",
    )
    ap.add_argument(
        "--target",
        nargs="+",
        type=Path,
        required=True,
        help="Directories/files of generated prose to inspect.",
    )
    ap.add_argument("--top", type=int, default=30, help="Rows per n-gram size.")
    ap.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Ignore n-grams occurring fewer than this many times in target.",
    )
    args = ap.parse_args(argv)

    print("Reading reference corpus (human-written):", file=sys.stderr)
    ref, ref_files, ref_words = _collect(args.reference)
    print("Reading target corpus (generated):", file=sys.stderr)
    tgt, tgt_files, tgt_words = _collect(args.target)

    if not ref or not tgt:
        print("error: one of the corpora is empty", file=sys.stderr)
        return 1

    print(f"\nreference : {ref_files:>4} files, {ref_words:>7,} prose tokens")
    print(f"target    : {tgt_files:>4} files, {tgt_words:>7,} prose tokens")
    print(
        "\nOver-represented in the generated prose. `ratio` is target rate over\n"
        "reference rate; rates are per 10k tokens. Domain terms are filtered,\n"
        "but subject matter still differs between corpora -- read, do not\n"
        "paste into a ban list.\n"
    )

    for n, label in ((1, "WORDS"), (2, "BIGRAMS"), (3, "TRIGRAMS")):
        rows = _over_represented(tgt, ref, n, top=args.top, min_count=args.min_count)
        print(f"== {label} " + "=" * (58 - len(label)))
        print(f"{'n-gram':<34}{'ratio':>8}{'count':>7}{'tgt/10k':>9}{'ref/10k':>9}")
        for gram, ratio, count, t_rate, r_rate in rows:
            shown = f"{ratio:>8.1f}" if r_rate > 0 else f"{'only':>8}"
            print(f"{gram:<34}{shown}{count:>7}{t_rate:>9.1f}{r_rate:>9.1f}")
        print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
