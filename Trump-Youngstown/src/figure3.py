"""Main figure: the divided inheritance of Jackson's concurrence.

A  Delta by case - within each case, how much more the majority leans on
   Jackson's three categories, relative to how much the dissent leans on his
   structural warning.  Signed quantity, so a diverging pair with a neutral
   zero line; Trump is the one mark that carries a label because it is the
   claim of the figure.
B  The placebo null - the same statistic computed from 4,000 random
   partitions of Jackson's paragraphs into groups of the same sizes.
"""

import json
import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.use("Agg")
HERE = os.path.dirname(__file__)
PROC = os.path.join(HERE, "..", "data", "proc")
OUT = os.path.join(HERE, "..", "out")

BLUE, RED = "#2a78d6", "#c93f3e"
INK, INK2, MUTED, GRID = "#15171c", "#4a4f5a", "#8a8e98", "#e6e6e2"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": "#d5d5d0", "axes.linewidth": .8,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

SHORT = {
    "Nixon v. Administrator of General Services": "Nixon v. GSA",
    "Department of the Navy v. Egan": "Navy v. Egan",
    "Free Enterprise Fund v. PCAOB": "Free Enterprise Fund",
    "Clinton v. City of New York": "Clinton v. New York",
    "Trump v. United States": "Trump v. United States",
    "Seila Law LLC v. CFPB": "Seila Law",
    "Trump v. Mazars USA": "Trump v. Mazars",
}


def main():
    W = json.load(open(os.path.join(PROC, "within.json")))
    RB = json.load(open(os.path.join(PROC, "within_robust.json")))
    recs = sorted(W["recs"], key=lambda r: r["delta"])

    fig = plt.figure(figsize=(10.4, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1], wspace=.42,
                          left=.215, right=.965, top=.745, bottom=.155)

    # ---------------- Panel A ----------------
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(recs))
    for i, r in enumerate(recs):
        d = r["delta"]
        trump = r["case"] == "trump_us_2024"
        col = RED if d > 0 else BLUE
        ax.plot([0, d], [i, i], color=col, lw=2.2 if trump else 1.5,
                alpha=1 if trump else .55, solid_capstyle="round", zorder=2)
        ax.plot([d], [i], "o", ms=9 if trump else 6, color=col,
                mec="white", mew=1.5 if trump else 1.1, zorder=3)
    ax.axvline(0, color="#b9b9b3", lw=1, zorder=1)
    ax.axvline(W["mean"], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(W["mean"], len(recs) - .1, f"  corpus mean {W['mean']:+.1f}",
            fontsize=7, color=MUTED, va="bottom")

    labels = [f"{SHORT.get(r['name'], r['name'])}  {r['year']}" for r in recs]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.6)
    for t, r in zip(ax.get_yticklabels(), recs):
        if r["case"] == "trump_us_2024":
            t.set_fontweight("bold")
            t.set_color(INK)
    ax.set_ylim(-.8, len(recs) - .2)
    ax.set_xlabel("Δ   majority's pull toward the categories, minus the dissent's",
                  fontsize=8)
    ax.grid(axis="x", color=GRID, lw=.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("A   Who keeps which half of Jackson's concurrence",
                 loc="left", fontsize=10, fontweight="bold", pad=42)
    ax.text(0, 1.075,
            "Each case compares its own majority against its own dissent, so subject matter,\n"
            "era and vocabulary cancel.  Red = the majority leans to the three categories.",
            transform=ax.transAxes, fontsize=7, color=MUTED, linespacing=1.5, va="bottom")

    # ---------------- Panel B ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    mu, sd = RB["placebo_null_mean"], RB["placebo_null_sd"]
    xs = np.linspace(mu - 3.6 * sd, mu + 3.6 * sd, 400)
    dens = np.exp(-.5 * ((xs - mu) / sd) ** 2)
    ax2.fill_between(xs, dens, color="#dfe3e8", zorder=2)
    ax2.plot(xs, dens, color="#b6bcc6", lw=1, zorder=3)

    for val, lab, col, p in ((W["mean"], "corpus mean", "#7d2b34", RB["placebo_p_mean"]),
                             (W["trump_delta"], "Trump", RED, RB["placebo_p_trump"])):
        ax2.axvline(val, color=col, lw=2, zorder=4)
        ax2.annotate(f"{lab}\n{val:+.1f}\np = {p:.3f}", (val, 1.02),
                     xytext=(6, 0), textcoords="offset points",
                     fontsize=7.5, color=col, fontweight="bold",
                     va="top", ha="left", linespacing=1.45)
    ax2.set_xlim(mu - 3.6 * sd, max(W["trump_delta"] + 9, mu + 3.6 * sd))
    ax2.set_ylim(0, 1.30)
    ax2.set_yticks([])
    ax2.set_xlabel("Δ under random partitions of Jackson's text", fontsize=8)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(length=0)
    ax2.set_title("B   Placebo test", loc="left", fontsize=10,
                  fontweight="bold", pad=42)
    ax2.text(0, 1.075,
             "Jackson's paragraphs split at random into same-sized\n"
             "groups, 4,000 times.  The real split is not arbitrary.",
             transform=ax2.transAxes, fontsize=7, color=MUTED, linespacing=1.5, va="bottom")

    fig.text(.215, .028,
             "n = 20 presidential-power cases, 1977–2024, in which the Court divided and both opinions run to at least ten paragraphs.  "
             "Δ is a difference-in-differences in\ncosine similarity to the 38 main-text paragraphs of Jackson's concurrence, double-centred within each case.  "
             "Sign test p = .006; Wilcoxon p = .002.",
             fontsize=6.5, color=MUTED, linespacing=1.6)

    fig.savefig(os.path.join(OUT, "figure3_inheritance.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "figure3_inheritance.pdf"))
    print("wrote figure3_inheritance.{png,pdf}")


if __name__ == "__main__":
    main()
