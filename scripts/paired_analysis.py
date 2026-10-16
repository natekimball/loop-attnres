"""Paired comparisons of the iGSM models (paper §5.7, Tables 9-10 and Appendix C).

Every model is scored on the same 1,000 15-operation problems, so differences are paired. The
paper treats a difference as a result only when McNemar's |z| > 1.96 and its size is well outside
the run-to-run spread of about 3.4 points. Growth of the carry's effect with depth is tested with a 95%
paired bootstrap interval on D = carry(T_b) - carry(T_a).

    python scripts/paired_analysis.py          # reads results/igsm/<model>.items.jsonl
"""
import glob, json, math, os, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(model):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "igsm", f"{model}.items.jsonl"))]
    return {r["i"]: r["correct"] for r in rows if r["bucket"] == "15"}


def mcnemar(a, b):
    """Paired difference b - a in points, the discordant counts, and McNemar's z."""
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
    b10 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    disc = b01 + b10
    return 100.0 * (b01 - b10) / len(keys), (b01 - b10) / math.sqrt(disc) if disc else 0.0


def boot_ci(vals, iters=20000, seed=1234):
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return 100.0 * means[int(0.025 * iters)], 100.0 * means[int(0.975 * iters)]


M = {os.path.basename(p)[:-len(".items.jsonl")]: None
     for p in glob.glob(os.path.join(ROOT, "results", "igsm", "*.items.jsonl"))}
M = {m: load(m) for m in M}
acc = lambda m: 100.0 * sum(M[m].values()) / len(M[m])

print("Accuracy (Table 11)")
for m in sorted(M):
    print(f"  {m:24s} {acc(m):5.1f}")

print("\nEqual compute (Table 9): best looped model vs dense model, and vs the identity carry")
for apps, dense, best, ident in [(10, "igsm-dense-10", "igsm-full-T2", "igsm-identity-T2"),
                                 (20, "igsm-dense-20", "igsm-full-T4", "igsm-identity-T4"),
                                 (40, "igsm-dense-40", "igsm-full-T8", "igsm-identity-T8")]:
    d, z = mcnemar(M[dense], M[best])
    print(f"  {apps:2d} block applications: {best} - {dense} = {d:+.1f} (z {z:+.2f}); "
          f"cost of tying {acc(dense) - acc(ident):.1f}, recovered {acc(best) - acc(ident):.1f}")

print("\nDense Attention Residuals (40 layers)")
for m in ["igsm-dense-40", "igsm-full-T8", "igsm-full-T16"]:
    d, z = mcnemar(M[m], M["igsm-dense-attnres-40"]) if m == "igsm-dense-40" else mcnemar(M["igsm-dense-attnres-40"], M[m])
    lab = "dense AttnRes - dense" if m == "igsm-dense-40" else f"{m} - dense AttnRes"
    print(f"  {lab:32s} {d:+.1f} (z {z:+.2f})")

print("\nCarry and placement by depth (Table 10)")
for T in (2, 4, 8, 16):
    d, z = mcnemar(M[f"igsm-identity-T{T}"], M[f"igsm-loop-T{T}"])
    line = f"  T={T:2d}  carry {d:+5.1f} (z {z:+.2f})"
    d, z = mcnemar(M[f"igsm-loop-T{T}"], M[f"igsm-full-T{T}"])
    print(line + f"   placement {d:+5.1f} (z {z:+.2f})")
base = sum(acc(m) for m in ["igsm-dense-5", "igsm-dense-5-seed2", "igsm-dense-5-seed3"]) / 3
print(f"  T= 1  placement: full {acc('igsm-full-T1'):.1f} vs the mean of three 5-layer seeds {base:.1f} "
      f"= {acc('igsm-full-T1') - base:+.1f} (second seed {acc('igsm-full-T1-seed2') - base:+.1f})")

print("\nGrowth of the carry's effect (pre-registered: 95% bootstrap interval excludes 0)")
c = {T: [M[f"igsm-loop-T{T}"][k] - M[f"igsm-identity-T{T}"][k] for k in range(1000)] for T in (4, 8, 16)}
for a, b in [(4, 16), (4, 8), (8, 16)]:
    D = [x - y for x, y in zip(c[b], c[a])]
    lo, hi = boot_ci(D)
    print(f"  carry(T={b}) - carry(T={a}) = {100.0 * sum(D) / len(D):+.1f}, 95% CI [{lo:+.1f}, {hi:+.1f}]")
for arm in ("identity", "loop", "full"):
    d, z = mcnemar(M[f"igsm-{arm}-T8"], M[f"igsm-{arm}-T16"])
    print(f"  {arm:8s} T=16 - T=8 = {d:+.1f} (z {z:+.2f})")
d, z = mcnemar(M["igsm-identity-T16"], M["igsm-full-T16"])
print(f"  full T=16 - identity T=16 = {d:+.1f} (z {z:+.2f})")
