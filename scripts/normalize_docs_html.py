"""Remove Quarto 1.7's unused AMD dependency from static notebook pages.

Quarto classifies any HTML output, including a dataframe table, as a widget.
Its injected RequireJS loader conflicts with later UMD scripts. Keep the
loader when an inline script or a widget actually uses AMD; otherwise remove
only the known loader and its jQuery bridge. Table markup is left intact.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPTS = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
AMD_CALL = re.compile(r"\b(?:require|requirejs|define)\s*\(")
SCRIPT_SRC = re.compile(r'''\bsrc=["']([^"']+)["']''', re.IGNORECASE)
WIDGET = "application/vnd.jupyter.widget"


def normalize(text: str) -> str:
    scripts = SCRIPTS.findall(text)
    loader = [s for s in scripts if "cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/" in s]
    bridge = [s for s in scripts if "define('jquery', [],function() {return window.jQuery;})" in s]
    if not loader or WIDGET in text:
        return text
    if any(AMD_CALL.search(s) for s in scripts if s not in loader + bridge):
        return text
    # A custom external script may use AMD internally. Its source is outside
    # this normalizer's evidence, so retain the loader in that case too.
    for script in scripts:
        source = SCRIPT_SRC.search(script)
        if source is not None and script not in loader:
            url = source.group(1)
            if not (
                re.match(r"^(?:\.\./)*site_libs/", url)
                or url.startswith("https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/")
            ):
                return text
    for script in loader + bridge:
        text = text.replace(script, "")
    return text


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "docs/_site"
    count = 0
    for path in root.rglob("*.html"):
        original = path.read_text(encoding="utf-8")
        result = normalize(original)
        if result != original:
            path.write_text(result, encoding="utf-8")
            count += 1
    print(f"Static notebook HTML: removed unused AMD loader on {count} pages")


if __name__ == "__main__":
    main()
