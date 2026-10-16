"""Dense parameters needed to match each looped model (paper §5.1, Appendix D and Table 8).

Reads configs/runs.csv, counts each model's parameters by building it, and interpolates the
dense ladder: log parameters against log perplexity (FineWeb-Edu) or against accuracy (iGSM).
Also fits the saturating power law log ppl = E + A N^-alpha of Appendix D to the dense models,
in non-embedding parameters N, and extrapolates the 20x2 full-placement model.

    python scripts/param_ratios.py
"""
import contextlib, csv, io, math, os, re, sys

import numpy as np
import torch
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from loop_attnres.nanogpt.model import GPT, GPTConfig
from loop_attnres.model import LoopedGPT, LoopedGPTConfig

RUNS = {r["id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "configs", "runs.csv")))}


def flags(run):
    f = dict(x[2:].split("=", 1) for x in run["command"].split()[3:])
    out = {}
    for k, v in f.items():
        out[k] = {"True": True, "False": False}.get(v, v)
        if isinstance(out[k], str) and re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
        elif isinstance(out[k], str) and re.fullmatch(r"-?\d*\.\d+(e-?\d+)?", v):
            out[k] = float(v)
    return out


def n_params(run, embedding=True):
    """Total parameters (the tied embedding counted once), as reported in the paper."""
    f = flags(run)
    f.setdefault("loop_residual", "attnres")         # the trainer's default
    igsm = f.get("dataset") == "igsm"
    base = dict(n_layer=f["n_layer"], n_head=8, n_embd=512, bias=False, dropout=0.0,
                block_size=f.get("block_size", 512), vocab_size=2094 if igsm else 50304)
    with torch.device("meta"), contextlib.redirect_stdout(io.StringIO()):
        if f.get("n_loops", 0) == 0:
            m = GPT(GPTConfig(**base))
        else:
            keys = LoopedGPTConfig.__dataclass_fields__
            m = LoopedGPT(LoopedGPTConfig(**base, **{k: v for k, v in f.items() if k in keys and k not in base}))
    n = sum(p.numel() for p in m.parameters())
    if not embedding:
        n -= m.transformer.wte.weight.numel() + m.transformer.wpe.weight.numel()
    return n / 1e6


def interp(x_ladder, y_ladder, y):
    """Interpolate x at y on a monotone ladder (x in log space); None outside it."""
    xs, ys = np.log(np.array(x_ladder)), np.array(y_ladder)
    order = np.argsort(ys)
    if not (ys.min() <= y <= ys.max()):
        return None
    return float(np.exp(np.interp(y, ys[order], xs[order])))


# ---- FineWeb-Edu: interpolate log params against log perplexity over the nine dense models ----
dense = ["dense-2x1", "dense-4x1", "dense-5x1", "dense-8x1", "dense-10x1", "dense-14x1",
         "dense-20x1", "dense-28x1", "dense-40x1"]
dp = [n_params(RUNS[d]) for d in dense]
dl = [math.log(float(RUNS[d]["val_ppl"])) for d in dense]
print("FineWeb-Edu, dense parameters needed (interpolated)")
for m in ["loop-5x8", "full-4x10", "full-5x8", "full-10x4", "full-20x2"]:
    p, ppl = n_params(RUNS[m]), float(RUNS[m]["val_ppl"])
    need = interp(dp, dl, math.log(ppl))
    print(f"  {m:12s} {p:6.1f}M  ppl {ppl:.2f}  " +
          (f"needs {need:6.1f}M  ratio {need / p:.2f}x" if need else "outside the ladder"))

# ---- Appendix D: saturating power law in non-embedding parameters ----
N = np.array([n_params(RUNS[d], embedding=False) for d in dense])
y = np.array(dl)
law = lambda n, E, A, a: E + A * n ** (-a)
(E, A, a), _ = curve_fit(law, N, y, p0=(3.0, 3.0, 0.4), maxfev=20000)
(_, _, a5), _ = curve_fit(law, N[-5:], y[-5:], p0=(E, A, a), maxfev=20000)
m = "full-20x2"
target = math.log(float(RUNS[m]["val_ppl"]))
n_need = (A / (target - E)) ** (1 / a)                       # non-embedding parameters needed
emb = n_params(RUNS[m]) - n_params(RUNS[m], embedding=False)
print(f"\nAppendix D power law: alpha {a:.2f} (largest five: {a5:.2f}); {m} needs "
      f"~{n_need + emb:.0f}M total, {(n_need + emb) / n_params(RUNS[m]):.1f}x its own")
# straight-line continuation through the 28- and 40-layer models
slope = (dl[-1] - dl[-2]) / (math.log(dp[-1]) - math.log(dp[-2]))
lin = math.exp(math.log(dp[-1]) + (target - dl[-1]) / slope)
print(f"  straight-line continuation of the ladder: {lin / n_params(RUNS[m]):.1f}x")

# ---- iGSM (Table 8): interpolate log params against accuracy over the four dense models ----
ig_dense = ["igsm-dense-5", "igsm-dense-10", "igsm-dense-20", "igsm-dense-40"]
ip = [n_params(RUNS[d]) for d in ig_dense]
ia = [float(RUNS[d]["igsm_accuracy"]) for d in ig_dense]
print("\niGSM, dense parameters needed (Table 8)")
for m in ["igsm-full-T2", "igsm-loop-T4", "igsm-identity-T8", "igsm-identity-T16", "igsm-full-T4",
          "igsm-loop-T8", "igsm-full-T8", "igsm-loop-T16", "igsm-full-T16"]:
    p, acc = n_params(RUNS[m]), float(RUNS[m]["igsm_accuracy"])
    need = interp(ip, ia, acc)
    print(f"  {m:18s} {p:5.1f}M  {acc:5.1f}%  " +
          (f"needs {need:6.1f}M  ratio {need / p:.2f}x" if need else f"above the ladder (>{ip[-1] / p:.2f}x)"))
