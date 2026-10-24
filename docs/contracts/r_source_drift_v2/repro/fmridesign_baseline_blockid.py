"""DES: BaselineTerm.get_block_matrix(blockid=) vs R 95042d4 (each active column
once; global intercept not duplicated; global constant drift present for every run)."""
import warnings

from fmrimod.baseline import baseline_model
from fmrimod.sampling import SamplingFrame

warnings.simplefilter("ignore")
sf = SamplingFrame(blocklens=[10, 10, 10], tr=2.0)
bm = baseline_model(sframe=sf, basis="poly", degree=1, intercept="global")
for name, term in bm.terms.items():
    if term is None:
        continue
    print(name, term.varname, "colind=", term.colind)
    for bid in ([1, 2, 3], 2, [2, 3]):
        sub = term.get_block_matrix(bid)
        print(f"   blockid={bid}: shape={sub.shape} cols={list(sub.columns)}")

bm2 = baseline_model(sframe=sf, basis="constant", degree=1, intercept="global")
for name, term in bm2.terms.items():
    if term is None:
        continue
    print("constant/global", name, "colind=", term.colind)
    for bid in (1, 2, [1, 2, 3]):
        sub = term.get_block_matrix(bid)
        print(f"   blockid={bid}: shape={sub.shape}")
