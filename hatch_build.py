"""Hatch build hook for optional native ARMA C backend packaging."""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sysconfig

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Compile and bundle optional native ARMA shared library in wheels."""

    def initialize(self, version: str, build_data: dict) -> None:
        src = Path(self.root) / "fmrimod" / "ar" / "arma_filter.c"
        if not src.exists():
            return

        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        out = src.parent / f"_arma_filter{ext_suffix}"

        cc_raw = self.config.get("cc") or sysconfig.get_config_var("CC") or "cc"
        cc = shlex.split(cc_raw) or ["cc"]

        cmd = [*cc, "-O3", "-fPIC", "-shared", str(src), "-o", str(out)]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                self.app.display_warning(
                    "Native ARMA backend compilation failed; falling back to runtime "
                    "numba/scipy paths."
                )
                return
        except Exception:
            self.app.display_warning(
                "Native ARMA backend compilation failed; falling back to runtime "
                "numba/scipy paths."
            )
            return

        rel = out.relative_to(Path(self.root)).as_posix()
        artifacts = build_data.setdefault("artifacts", [])
        if rel not in artifacts:
            artifacts.append(rel)
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        self.app.display_info(f"Bundled native ARMA backend: {rel}")
