"""CLI option validation / help coverage and SPMG1 derivative accuracy."""
import contextlib
import io

import numpy as np
from scipy import stats

from fmrimod import cli
from fmrimod.hrf import HRF_SPMG1, deriv


def run(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.fmrihrf_cli(args)
    return rc, out.getvalue(), err.getvalue().strip()

for args in (["eval", "--hrf", "spmg1", "--nbasis", "x"],
             ["eval", "--hrf", "spmg1", "--span", "2,3"],
             ["eval", "--hrf", "spmg1", "--output="],
             ["eval", "--hrf", "spmg1", "--from=0", "--to=1", "--by=0.5", "--json"]):
    rc, out, err = run(args)
    print(args, "->", rc, err[:90] or out[:120].replace("\n", " "))
rc, out, _ = run(["help", "design"])
print("design help mentions --sparse/--json/--output/--method:",
      [k for k in ("--sparse", "--json", "--output", "--method", "--start-time", "--span") if k in out])

t = np.array([0.5, 2.5, 6.0, 15.0])
a = (stats.gamma.pdf(t, 6) * (5 / t - 1)) - 0.167 * stats.gamma.pdf(t, 16) * (15 / t - 1)
print("deriv(HRF_SPMG1) numeric vs analytic max|diff|:", np.max(np.abs(np.ravel(deriv(HRF_SPMG1, t)) - a)))
