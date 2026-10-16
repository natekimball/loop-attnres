"""Figures 1 and 2 of the paper, from configs/runs.csv.

    python scripts/make_figures.py      # writes figures/fig1_tying_grid.pdf, figures/fig2_reasoning.pdf
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
R = {r["id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "configs", "runs.csv")))}
ppl = lambda i: float(R[i]["val_ppl"])
accu = lambda i: float(R[i]["igsm_accuracy"])

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold",
})
C = {"dense": "#8c8c8c", "identity": "#4c72b0", "attnres": "#c44e52",
     "full": "#55a868", "coda": "#8172b2", "accent": "#dd8452"}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


TVALS = [20, 10, 8, 4, 2, 1]                 # grid points, most tied first
dense    = [ppl(f"dense-{s}") for s in ("2x1", "4x1", "5x1", "10x1", "20x1", "40x1")]
identity = [ppl(f"identity-{s}") for s in ("2x20", "4x10", "5x8", "10x4", "20x2")] + [None]
attnres  = [ppl(f"loop-{s}") for s in ("2x20", "4x10", "5x8", "10x4", "20x2")] + [None]
fullatt  = [ppl(f"full-{s}") for s in ("2x20", "4x10", "5x8", "10x4", "20x2", "40x1")]

# ---- Fig 1: the tying grid (a) and the carry's value against tying (b) ------------------
# One figure, two panels with the SAME x axis: iterations T on a log scale (L = 40 / T), so a
# point in (b) sits directly under the cell of (a) it is derived from.
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.9))
fig.subplots_adjust(wspace=0.30)
TX = TVALS
def tx_axis(a):
    a.set_xscale("log"); a.set_xticks([1, 2, 4, 8, 10, 20])
    a.set_xticklabels(["1\n40×1", "2\n20×2", "4\n10×4", "8\n5×8", "10\n4×10", "20\n2×20"], fontsize=7)
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # T=8 and T=10 sit close on a log axis: align their labels away from each other
    labs = a.get_xticklabels(); labs[3].set_ha("right"); labs[4].set_ha("left")
    a.set_xlim(0.85, 24)
    a.set_xlabel("iterations T\nlayers × iterations", linespacing=1.4)
ax.plot(TX, dense, "o-", color=C["dense"], label="dense", lw=1.6, ms=4.5)
# matched-compute reference: the 40-layer dense model. The gap from the identity carry to this
# line is the cost of tying at equal compute; the carry closes part of it (Sec. 5.2).
ax.axhline(dense[-1], color=C["dense"], ls="--", lw=0.9, zorder=0)
ax.text(23, dense[-1] - 0.3, "40-layer dense (same compute)", fontsize=6.5, color=C["dense"], ha="right", va="top")
# At 40x1 under loop placement both carries pass the single state through, so that model is the
# dense one: draw both lines into the dense point, with markers only on runs that exist.
for ys, mk, key, lab, lw, ms in [(identity, "s-", "identity", "identity carry", 1.6, 4.5),
                                 (attnres, "^-", "attnres", "attention, loop", 1.8, 5)]:
    ts = [t for t, v in zip(TX, ys) if v]
    ax.plot(ts + [1], [v for v in ys if v] + [dense[-1]], mk, color=C[key], label=lab,
            lw=lw, ms=ms, markevery=list(range(len(ts))))
ax.plot(TX, fullatt, "D-", color=C["full"], label="attention, full", lw=1.8, ms=4.5)
tx_axis(ax)
ax.set_ylabel("validation perplexity")
ax.set_title("(a) Perplexity vs. tying", fontsize=9.5)
ax.legend(loc="upper left", fontsize=8)

# (b) benefit over the identity carry; each perplexity rounded first, as in Tables 2 and 7.
# At T=1 the identity model is the dense one; loop placement is 0 there by construction.
ident = [i if i else d for i, d in zip(identity, dense)]
loopd = [round(i, 2) - round(a, 2) if a else 0.0 for i, a in zip(ident, attnres)]
fulld = [round(i, 2) - round(f, 2) for i, f in zip(ident, fullatt)]
ax2.plot(TX, loopd, "^-", color=C["attnres"], label="attention, loop", lw=1.8, ms=5,
         markevery=[k for k, a in enumerate(attnres) if a])
ax2.plot(TX, fulld, "D-", color=C["full"], label="attention, full", lw=1.8, ms=4.5)
ax2.axhspan(0, 0.18, color="0.85", zorder=0)
ax2.text(22, 0.04, "noise threshold", fontsize=7, color="0.35", ha="right")
tx_axis(ax2)
ax2.set_ylim(bottom=-0.1)
ax2.set_ylabel("identity − carry  (perplexity)")
ax2.set_title("(b) Gain over the identity carry", fontsize=9.5)
# no legend here: the same colors and markers as panel (a) (one shared legend)
save(fig, "fig1_tying_grid")

# ---- Fig 2: iGSM reasoning -- accuracy against compute (a) and against parameters (b) ------
# Greedy exact-answer accuracy (%) on the same 1,000 problems (Appendix C). A 5-layer block looped
# T times costs 5T block applications; the identity model at T=1 IS the 5-layer dense model.
IG_T = [1, 2, 4, 8, 16]
IG_DENSE = {L: accu(f"igsm-dense-{L}") for L in (5, 10, 20, 40)}   # layers -> accuracy
IG_DENSE_P = {5: 17.9, 10: 33.6, 20: 65.1, 40: 128.0}            # layers -> params (M)
IG_ID = [accu("igsm-dense-5")] + [accu(f"igsm-identity-T{t}") for t in (2, 4, 8, 16)]
IG_LOOP = [accu(f"igsm-loop-T{t}") for t in (2, 4, 8, 16)]
IG_FULL = [accu(f"igsm-full-T{t}") for t in (1, 2, 4, 8, 16)]
IG_AFD = (40, 128.3, accu("igsm-dense-attnres-40"))                # dense Attention Residuals
IG_D80 = (80, 252.8, accu("igsm-dense-80"))   # 80-layer dense: overfits (Appendix C)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.9))
fig.subplots_adjust(wspace=0.30)
apps = [5 * t for t in IG_T]
ax.plot(list(IG_DENSE), list(IG_DENSE.values()), "o-", color=C["dense"], label="dense", lw=1.6, ms=4.5)
ax.plot(apps, IG_ID, "s-", color=C["identity"], label="identity carry", lw=1.6, ms=4.5)
ax.plot(apps[1:], IG_LOOP, "^-", color=C["attnres"], label="attention, loop", lw=1.8, ms=5)
ax.plot(apps, IG_FULL, "D-", color=C["full"], label="attention, full", lw=1.8, ms=4.5)
ax.plot([IG_AFD[0]], [IG_AFD[2]], "*", color=C["dense"], ms=10, mec="k", mew=0.5,
        label="dense, AttnRes")
# the 80-layer dense model overfits: dashed, hollow, so it reads as off the ladder's trend
ax.plot([40, IG_D80[0]], [IG_DENSE[40], IG_D80[2]], "--", color=C["dense"], lw=1.1)
ax.plot([IG_D80[0]], [IG_D80[2]], "o", color=C["dense"], mfc="white", ms=4.5, mew=1.2)
ax.annotate("overfits", (IG_D80[0], IG_D80[2]), xytext=(-4, -11), textcoords="offset points",
            fontsize=6.5, color=C["dense"], ha="right")
ax.set_xscale("log"); ax.set_xticks(apps)
ax.set_xticklabels([f"{a}\n{t}" for a, t in zip(apps, IG_T)], fontsize=7)
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlim(4.3, 92)
ax.set_xlabel("block applications\niterations T (looped)", linespacing=1.4)
ax.set_ylabel("accuracy (%)")
ax.set_title("(a) Accuracy vs. compute", fontsize=9.5)
ax.legend(loc="lower right", fontsize=7)

# (b) all looped models have 17.9M (identity) or 18.1M (attention) parameters, so they share one
# x position; to keep that column readable, plot the best model of each carry, with a dotted guide
# across to the dense ladder (the dense size it matches, Table 8).
ax2.plot(list(IG_DENSE_P.values()), list(IG_DENSE.values()), "o-", color=C["dense"], lw=1.6, ms=4.5)
ax2.plot([IG_AFD[1]], [IG_AFD[2]], "*", color=C["dense"], ms=10, mec="k", mew=0.5)
ax2.plot([128.0, IG_D80[1]], [IG_DENSE[40], IG_D80[2]], "--", color=C["dense"], lw=1.1)
ax2.plot([IG_D80[1]], [IG_D80[2]], "o", color=C["dense"], mfc="white", ms=4.5, mew=1.2)
for p_, v, mk, key, lab, dy in [(18.1, IG_FULL[-1], "D", "full", "full, T=16", 4),
                                (18.1, IG_LOOP[-1], "^", "attnres", "loop, T=16", 4),
                                (17.9, IG_ID[-1], "s", "identity", "identity, T=16", -9)]:
    ax2.axhline(v, xmin=0.0, xmax=1.0, color=C[key], lw=0.8, ls=":", zorder=1)
    hollow = False
    ax2.plot([p_], [v], mk, color=C[key], ms=7 if hollow else 5, mfc="white" if hollow else C[key],
             mew=1.3, zorder=2 if hollow else 4)
    ax2.annotate(lab, (p_, v), xytext=(7, dy), textcoords="offset points", fontsize=7, color=C[key])
ax2.set_xscale("log"); ax2.set_xticks([17.9, 33.6, 65.1, 128.0, 252.8])
ax2.set_xticklabels(["17.9", "33.6", "65.1", "128", "253"], fontsize=7)
ax2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax2.set_xlim(14, 320)
ax2.set_xlabel("parameters (M)")
ax2.set_ylabel("accuracy (%)")
ax2.set_title("(b) Accuracy vs. parameters", fontsize=9.5)
for a in (ax, ax2):
    a.set_ylim(25, 90)
save(fig, "fig2_reasoning")

