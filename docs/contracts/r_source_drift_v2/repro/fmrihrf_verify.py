"""Empirical checks for fmrihrf v2 drift rows (HRF-xx).

Each block transcribes the fixed R formula/expectation (fmrihrf HEAD 5b63d6a)
and compares against fmrimod HEAD. Run with the fmrimod venv python.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import interpolate, stats

warnings.simplefilter("ignore")

from fmrimod.hrf import (  # noqa: E402
    HRF_MEXHAT,
    HRF_SPMG1,
    HRF_SPMG2,
    HRF_SPMG3,
    as_hrf,
    block_hrf,
    bspline_hrf,
    hrf_basis_lwu,
    hrf_tent_generator,
)
from fmrimod.hrf.functions import spm_canonical_legacy  # noqa: E402
from fmrimod.hrf.generators import weighted_generator  # noqa: E402
from fmrimod.regressor import regressor  # noqa: E402


def hdr(tag: str) -> None:
    print(f"\n=== {tag}")


# ---------------------------------------------------------------- HRF-01 block_hrf
hdr("HRF-01 block_hrf integrates every positive width (dff54b9)")

flat = as_hrf(lambda t: ((np.asarray(t) >= 0) & (np.asarray(t) <= 4)).astype(float), span=4.0)
linear = as_hrf(lambda t: np.where((np.asarray(t) >= 0) & (np.asarray(t) <= 4), np.asarray(t, float), 0.0), span=4.0)
for width in (1e-6, 0.05, 0.0999, 0.1, 0.1001):
    for prec in (0.1, 0.01):
        got = float(np.ravel(block_hrf(flat, width, prec)(np.array([1.0])))[0])
        got_lin = np.ravel(block_hrf(linear, width, prec)(np.array([1.0, 2.0])))
        exp_lin = np.array([1.0, 2.0]) * width - width**2 / 2
        print(f"width={width:<7} prec={prec:<5} flat: fmrimod={got:.6g} R={width:.6g}  "
              f"linear max|diff|={np.max(np.abs(got_lin - exp_lin)):.3g}")
below = float(np.ravel(block_hrf(flat, 0.1 - 1e-8, 0.1)(np.array([1.0])))[0])
above = float(np.ravel(block_hrf(flat, 0.1 + 1e-8, 0.1)(np.array([1.0])))[0])
print(f"continuity at width=precision: above-below fmrimod={above - below:.6g}  R=2e-8")

# ---------------------------------------------------------------- HRF-02 bspline
hdr("HRF-02 B-spline/tent basis zero at both ends (18d418f)")


def r_bspline_new(t, span=24.0, N=5, degree=3):
    """Transcription of R hrf_bspline() at HEAD: interior N of clamped N+2 basis."""
    t = np.asarray(t, float)
    n_int = N + 2 - (degree + 1)
    knots_int = np.linspace(0, span, n_int + 2)[1:-1] if n_int > 0 else np.array([])
    knots = np.concatenate([np.repeat(0.0, degree + 1), knots_int, np.repeat(span, degree + 1)])
    nfull = len(knots) - degree - 1
    out = np.zeros((len(t), nfull))
    inside = (t >= 0) & (t <= span)
    te = np.where(inside, t, 0.0)
    for i in range(nfull):
        c = np.zeros(nfull)
        c[i] = 1
        # splines::bs evaluates the right boundary as the limit from the left
        v = interpolate.BSpline(knots, c, degree, extrapolate=True)(np.minimum(te, span - 1e-12))
        out[:, i] = v
    out[~inside] = 0
    return out[:, 1:-1]


t = np.arange(0, 24.0001, 0.1)
for deg in (1, 2, 3):
    B = bspline_hrf(t, n_basis=6, degree=deg, span=24.0)
    R = r_bspline_new(t, N=6, degree=deg)
    print(f"degree={deg}: fmrimod row(t=0)={np.round(B[0], 3).tolist()} row(t=24)={np.round(B[-1], 3).tolist()}; "
          f"R row(t=24)={np.round(R[-1], 3).tolist()}; max|fmrimod-R|={np.max(np.abs(B - R)):.3f}")
tent = hrf_tent_generator(nbasis=5, span=24)
tt = np.arange(0, 24.001, 0.01)
peaks_fm = tt[np.argmax(tent(tt), axis=0)]
peaks_r = tt[np.argmax(r_bspline_new(tt, N=5, degree=1), axis=0)]
print(f"tent N=5 peaks: fmrimod={np.round(peaks_fm, 2).tolist()}  R={np.round(peaks_r, 2).tolist()}")
# step at end of span in a regressor built from the default bspline basis
from fmrimod.hrf import HRF_BSPLINE  # noqa: E402

reg = regressor([0.0], hrf=HRF_BSPLINE)
g = np.arange(23, 25.0001, 0.1)
y = reg.evaluate(g, precision=0.1)
print(f"regressor(0, HRF_BSPLINE) max|diff| over 23..25 s: fmrimod={np.max(np.abs(np.diff(y, axis=0))):.3f} (R test requires < 0.05)")
try:
    b1 = bspline_hrf(t, n_basis=1, degree=3)
    print(f"bspline n_basis=1 degree=3: fmrimod returns shape {b1.shape}, max={b1.max():.3f} (R: error 'at least 2')")
except Exception as e:  # noqa: BLE001
    print("bspline n_basis=1:", type(e).__name__, e)

# ---------------------------------------------------------------- HRF-03 hrf_weighted
hdr("HRF-03 hrf_weighted constant method uses every weight (18d418f)")
h = weighted_generator([1, 2, 3, 0], width=6, method="constant")
pts = np.array([1.49, 1.5, 2.99, 3, 4.49, 4.5])
print(f"width=6 weights=[1,2,3,0] at {pts.tolist()}: fmrimod={np.round(h(pts), 3).tolist()}  R=[1,2,2,3,3,0]; span fmrimod={h.span} R=6")
h2 = weighted_generator([0.1, 0.3, 1, 1, 0.3, 0.1], width=10)
tt = np.arange(0, 12.0001, 0.01)
yy = h2(tt)
sel = (tt >= 8.34) & (tt < 10)
print(f"weights 6 over width 10: fmrimod values on [8.34,10) unique={np.unique(np.round(yy[sel], 4)).tolist()} (R: all 0.1); "
      f"fmrimod h(10)={float(h2(np.array([10.0]))[0])} (R: 0)")
h3 = weighted_generator([1, 2, 2, 1], times=[4, 6, 8, 10])
print(f"times=[4,6,8,10] weights=[1,2,2,1] at [5,7,9,11,12]: fmrimod={np.round(h3(np.array([5, 7, 9, 11, 12.0])), 3).tolist()} "
      f"R(raw)=[1,2,2,1,0]; span fmrimod={h3.span} R=12")

# ---------------------------------------------------------------- HRF-04 mexhat
hdr("HRF-04 Mexican-hat normalising constant (7e5f7ec)")
r_const = 2 / (math.sqrt(3 * 2) * math.pi ** 0.25)
fm_peak = float(np.ravel(HRF_MEXHAT(np.array([6.0])))[0])
print(f"HRF_MEXHAT(6): fmrimod={fm_peak:.6f} R={r_const:.6f} ratio R/fmrimod={r_const / fm_peak:.6f} (sqrt(2)*pi^-1/8={math.sqrt(2) * math.pi ** -0.125:.6f})")
dt = 1e-3
tt = np.arange(6 - 24, 6 + 24, dt)
mu = 30.0
tt = np.arange(mu - 24, mu + 24, dt)
from fmrimod.hrf.library import MexhatHRF  # noqa: E402

l2 = float(np.sum(MexhatHRF(mean=mu, sd=2.0)(tt) ** 2) * dt)
print(f"L2 norm^2 of MexhatHRF(mean=30, sd=2): fmrimod={l2:.6f} R=1")
print(f"causal mask: HRF_MEXHAT(-3) fmrimod={float(np.ravel(HRF_MEXHAT(np.array([-3.0])))[0]):.4g} (R .causal -> 0)")
bm = block_hrf(HRF_MEXHAT, 4.0)
print(f"block_hrf(HRF_MEXHAT, 4)(1.0): includes pre-onset mass? fmrimod={float(np.ravel(bm(np.array([1.0])))[0]):.5f}; "
      "(causal R reference computed in fmrihrf_extra.py)")

# ---------------------------------------------------------------- HRF-05 conv window edge
hdr("HRF-05 conv block at the start of the fine window keeps its mass (b901bca)")
box = as_hrf(lambda t: np.where((np.asarray(t) >= 0) & (np.asarray(t) <= 1), 1.0, 0.0), span=1.0, name="box")
reg = regressor([0.0], hrf=box, duration=0.5)
at_edge = reg.evaluate(np.array([1.0, 2.0, 3.0]), precision=0.1)
inside = reg.evaluate(np.array([0.5, 1.0, 2.0, 3.0]), precision=0.1)[1:]
print(f"grid [1,2,3]: fmrimod={np.round(at_edge, 4).tolist()}  grid [0.5,1,2,3][1:]: {np.round(inside, 4).tolist()}  R: both [0.5, 0, 0]")
at_edge = reg.evaluate(np.array([1.05, 2.05]), precision=0.1)
inside = reg.evaluate(np.array([0.55, 1.05, 2.05]), precision=0.1)[1:]
print(f"grid [1.05,2.05]: fmrimod={np.round(at_edge, 4).tolist()} vs {np.round(inside, 4).tolist()} (R: equal)")

# ---------------------------------------------------------------- HRF-06 SPMG
hdr("HRF-06 SPMG basis at R HEAD (daca7ba, d19ba00) vs fmrimod default SPMG")
t = np.arange(0, 32.0001, 0.01)
r_spmg1 = stats.gamma.pdf(t, 6) - stats.gamma.pdf(t, 16) / 6
fm1 = HRF_SPMG1(t)
print(f"SPMG1 max|fmrimod - R_new| = {np.max(np.abs(fm1 - r_spmg1)):.3e} (peak {r_spmg1.max():.4f}); "
      f"R_old(A1=.0833) peak = {np.max(np.exp(-t) * (0.0833 * t**5 - 1.274527e-13 * t**15)):.4f}")
r_disp = ((stats.gamma.pdf(t, 6) - stats.gamma.pdf(t, 16) / 6)
          - (stats.gamma.pdf(t, 6 / 1.01, scale=1.01) - stats.gamma.pdf(t, 16) / 6)) / 0.01
fm3 = HRF_SPMG3(t)
print(f"SPMG3 col3 (dispersion) max|fmrimod - R_new| = {np.max(np.abs(fm3[:, 2] - r_disp)):.3e} (max|R| {np.max(np.abs(r_disp)):.4f})")
A1, C = 1 / 120, 1 / (6 * math.factorial(15))
r_dt = np.exp(-t) * (A1 * t**4 * (5 - t) - C * t**14 * (15 - t))
fm2 = HRF_SPMG2(t)[:, 1]
cor = np.corrcoef(fm2, r_dt)[0, 1]
beta = float(np.dot(fm2, r_dt) / np.dot(r_dt, r_dt))
print(f"SPMG2 col2: fmrimod (h(delay)-h(delay+0.1))/0.1 vs R analytic dh/dt: corr={cor:.4f}, LS scale={beta:.3f}, "
      f"max|fm - scale*R|={np.max(np.abs(fm2 - beta * r_dt)):.3e}")
nil_like = (np.interp(t, t, r_spmg1) - np.interp(t - 0.1, t, r_spmg1, left=0)) / 0.1
print(f"  (Nilearn onset-shift FD vs R dh/dt corr={np.corrcoef(nil_like, r_dt)[0, 1]:.4f}; vs fmrimod corr={np.corrcoef(nil_like, fm2)[0, 1]:.4f})")

hdr("HRF-07 legacy spm_canonical_legacy undershoot normaliser with P2 != 15 (d19ba00)")
t = np.arange(0, 40.0001, 0.25)
for p2 in (12.0, 15.0, 16.5):
    fm = spm_canonical_legacy(t, p2=p2)
    rr = np.exp(-t) * (0.0833 * t**5 - t**p2 / (6 * math.gamma(p2 + 1)))
    print(f"p2={p2}: max|fmrimod - fixedR(A1=.0833)| = {np.max(np.abs(fm - rr)):.4e} (peak {np.max(np.abs(rr)):.3f})")

# ---------------------------------------------------------------- HRF-08 CLI design run spill
hdr("HRF-08 CLI `design` convolves across runs (a8556ff / #62)")
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

from fmrimod import cli  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    ev = os.path.join(d, "ev.csv")
    with open(ev, "w") as fh:
        fh.write("onset,condition,block\n230,a,1\n")
    out = os.path.join(d, "out.json")
    with contextlib.redirect_stdout(io.StringIO()):
        rc = cli.fmrihrf_cli(["design", "--events", ev, "--blocklens", "120,120", "--tr", "2", "--json", "--output", out]) \
            if True else None
    rows = json.load(open(out))
    vals = np.array([r["a"] for r in rows], float)
    print(f"rc={rc}; run-1 nonzero rows={int(np.sum(vals[:120] != 0))}; run-2 nonzero rows={int(np.sum(vals[120:] != 0))}, "
          f"max|run2|={np.max(np.abs(vals[120:])):.4g} (R: run 2 identically 0)")
    # missing explicitly named duration column
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            rc2 = cli.fmrihrf_cli(["design", "--events", ev, "--blocklens", "120,120", "--tr", "2", "--duration", "dur_s",
                            "--output", out])
        except SystemExit as e:  # noqa: PERF203
            rc2 = e.code
    print(f"--duration dur_s with no dur_s column: fmrimod exit={rc2} (R: exit 1, 'Missing required column(s)')")

# ---------------------------------------------------------------- HRF-09 empty regressors
hdr("HRF-09 regressors without events (b7ce497, 304f1fb, 6212569)")
reg = regressor([1.0, 2.0], hrf=[HRF_SPMG3], amplitude=0.0)
single = regressor([1.0, 2.0], hrf=HRF_SPMG3, amplitude=0.0)
print(f"list-HRF all-dropped: nbasis fmrimod={reg.nbasis} (R 3); span fmrimod={reg.span} vs single {single.span}; "
      f"evaluate shape={reg.evaluate(np.arange(6.0)).shape} (R (6,3))")
from fmrimod.regressor import regressor_set  # noqa: E402

try:
    rs = regressor_set(np.array([]), np.array([], dtype=str))
    m = rs.evaluate(np.arange(6.0))
    print(f"regressor_set with no levels: shape {m.shape}")
except Exception as e:  # noqa: BLE001
    print(f"regressor_set with no levels -> evaluate raises {type(e).__name__}: {e} (R: 6x0 matrix)")
empty = regressor([1.0, 2.0], amplitude=0.0)
tni, vni = empty.neural_input(start=5, resolution=1)
print(f"neural_input empty regressor start=5: time range fmrimod=({tni[0]}, {tni[-1]}) (R: 5..15)")

# ---------------------------------------------------------------- HRF-10 neural_input
hdr("HRF-10 neural_input is the conv stimulus density (ca84508)")
pts = regressor([3.0, 7.1, 12.62], amplitude=[1.0, -2.0, 0.5])
tm, ni = pts.neural_input(start=0, end=20, resolution=0.25)
print(f"point events: sum(ni)*res fmrimod={np.sum(ni) * 0.25:.4f} (R: sum(amp)={1 - 2 + 0.5}); "
      f"value at t=3 fmrimod={ni[np.isclose(tm, 3)][0] if np.any(np.isclose(tm, 3)) else 'n/a'} (R: 1/res=4)")
short = regressor([2.05, 5.6], duration=[0.1, 0.07], amplitude=[2.0, 3.0])
tm, ni = short.neural_input(start=0, end=10, resolution=0.5)
print(f"sub-resolution blocks: mass fmrimod={np.sum(ni) * 0.5:.4f} (R: {2 * 0.1 + 3 * 0.07:.4f})")
reg = regressor([2.0, 6.0], duration=[2.0, 1.0], amplitude=[1.0, 3.0])
tm, ni = reg.neural_input(start=0, end=10, resolution=1)
print(f"blocks res=1: fmrimod={ni.tolist()}  R=[0,0,0.5,1,0.5,0,1.5,1.5,0,0,0]")
tm, ni = reg.neural_input(start=0, end=10, resolution=0.33)
print(f"time grid spacing res=0.33, end=10: fmrimod step={tm[1] - tm[0]:.5f} (R seq(by=0.33) step=0.33)")

# ---------------------------------------------------------------- HRF-11 sampling_frame
hdr("HRF-11 sampling_frame fractional blocklens (dbcb920)")
from fmrimod.sampling import SamplingFrame  # noqa: E402

try:
    sf = SamplingFrame(blocklens=[100, 10.5], tr=2)
    print(f"SamplingFrame(blocklens=[100, 10.5]) accepted -> blocklens={sf.blocklens.tolist()} (R: error 'whole numbers')")
except Exception as e:  # noqa: BLE001
    print("raises", type(e).__name__, e)

# ---------------------------------------------------------------- HRF-12 LWU basis bounds
hdr("HRF-12 hrf_basis_lwu at validated bounds (5489369)")
for theta in ([6, 2.5, 0.0], [6, 2.5, 1.5], [6, 0.050001, 0.35]):
    try:
        b = hrf_basis_lwu(theta, np.arange(0, 20.0, 1.0))
        print(f"theta0={theta}: ok shape {b.shape}")
    except Exception as e:  # noqa: BLE001
        print(f"theta0={theta}: {type(e).__name__}: {e} (R: works, one-sided derivative)")

# ---------------------------------------------------------------- misc contracts
hdr("misc contracts")
r = regressor([1.0], summate=None)  # type: ignore[arg-type]
print(f"regressor(summate=None) accepted: summate={r.summate!r} (R rejects summate=NA)")
from fmrimod.hrf import gen_hrf, get_hrf, make_hrf  # noqa: E402

try:
    g = gen_hrf(HRF_SPMG1, width=-1)
    print(f"gen_hrf(HRF_SPMG1, width=-1) accepted silently -> {type(g).__name__} (R getHRF: error 'width > 0')")
except Exception as e:  # noqa: BLE001
    print("gen_hrf width=-1:", type(e).__name__, e)
for nm in ("bspline", "fir", "spmg3"):
    print(f"make_hrf({nm!r}).nbasis = {make_hrf(nm).nbasis}")
try:
    m2 = make_hrf(lambda t: np.column_stack([stats.norm.pdf(t, 5), stats.norm.pdf(t, 7)]))  # type: ignore[arg-type]
    print(f"make_hrf(two-column function).nbasis = {m2.nbasis} (R: 2)")
except Exception as e:  # noqa: BLE001
    print("make_hrf(callable):", type(e).__name__, e)
try:
    print(f"get_hrf('lwu') -> {type(get_hrf('lwu')).__name__}, nbasis={get_hrf('lwu').nbasis}")
except Exception as e:  # noqa: BLE001
    print("get_hrf('lwu'):", type(e).__name__, e)
from fmrimod.hrf import gen_hrf_lagged, hrf_lagged  # noqa: E402

t = np.arange(0, 30.0001, 0.5)
s = gen_hrf_lagged(HRF_SPMG1, [0, 2, 4])
ref = np.column_stack([HRF_SPMG1(t), HRF_SPMG1(t - 2), HRF_SPMG1(t - 4)])
print(f"gen_hrf_lagged vector: nbasis={s.nbasis}, max|diff|={np.max(np.abs(s(t) - ref)):.2e}; SPMG2 x2 lags nbasis={gen_hrf_lagged(HRF_SPMG2, [1, 2]).nbasis}")
try:
    z = hrf_lagged(HRF_SPMG1, 5, normalize=True)(np.arange(4.0))
    print(f"hrf_lagged(lag=5, normalize=True)(0:3) = {np.round(z, 6).tolist()} (R: zeros)")
except Exception as e:  # noqa: BLE001
    print("hrf_lagged normalize:", type(e).__name__, e)
from fmrimod.hrf import list_available_hrfs  # noqa: E402

info = list_available_hrfs(details=True)
print("aliases flagged:", [d["name"] for d in info if d["is_alias"]][:20])
from fmrimod.hrf import penalty_matrix  # noqa: E402
from fmrimod.hrf.generators import daguerre_generator  # noqa: E402

print("daguerre penalty nb=1:", penalty_matrix(daguerre_generator(n_basis=1)).shape)
print("bspline empty grid:", bspline_hrf(np.array([]), n_basis=5).shape, "tent empty:", hrf_tent_generator(nbasis=4)(np.array([])).shape)
