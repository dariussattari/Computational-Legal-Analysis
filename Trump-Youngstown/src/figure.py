"""Figures for the paper.

Figure 1 - the finding: relative engagement with each section of Jackson's
           concurrence, plus the bootstrapped majority/dissent contrasts.
Figure 2 - the limits: what the same text supports and does not support,
           as leave-one-case-out AUC for three targets.

Diverging data (signed affinity) gets the blue<->red diverging pair with a
neutral gray midpoint; the AUC panel is a single-series magnitude chart, so it
carries no legend and is direct-labelled.
"""

import json
import os
import sys

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

mpl.use("Agg")
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from embed import Embedder
from jackson import JACKSON_MAIN_END, SECTIONS, section_of, strip_quotes

PROC = os.path.join(HERE, "..", "data", "proc")
OUT = os.path.join(HERE, "..", "out")

BLUE, GRAY, RED = "#2a78d6", "#f0efec", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
CMAP = LinearSegmentedColormap.from_list("cr", [BLUE, GRAY, RED])
ORDER = ["Roberts", "Thomas", "Barrett", "Sotomayor", "Jackson"]
ROLE = {"Roberts": "majority", "Thomas": "concurrence", "Barrett": "concurrence",
        "Sotomayor": "dissent", "Jackson": "dissent"}

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": "#d8d7d2", "axes.linewidth": .8,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def affinity_matrix():
    rows = [json.loads(l) for l in open(os.path.join(PROC, "paragraphs.jsonl"))]
    V = np.load(os.path.join(PROC, "embeddings.npy"))
    idx = [i for i, r in enumerate(rows) if not r["is_footnote"] and r["n_words"] >= 25]
    jk = [i for i in idx if rows[i]["case_key"] == "youngstown_1952"
          and rows[i]["author"] == "Jackson"][:JACKSON_MAIN_END]
    J = V[jk] / np.linalg.norm(V[jk], axis=1, keepdims=True)
    lab = [section_of(k) for k in range(len(jk))]
    tr = [i for i in idx if rows[i]["case_key"] == "trump_us_2024"]
    by = {}
    for i in tr:
        by.setdefault(rows[i]["author"], []).append(i)
    S = []
    for a in ORDER:
        M = V[by[a]]
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        S.append((M @ J.T).mean(0))
    S = np.vstack(S)
    R = S - S.mean(1, keepdims=True) - S.mean(0, keepdims=True) + S.mean()
    names = [n for n, _ in SECTIONS]
    G = np.zeros((len(ORDER), len(names)))
    for c, n in enumerate(names):
        cols = [k for k in range(len(jk)) if lab[k] == n]
        G[:, c] = R[:, cols].mean(1) * 1000
    counts = {a: len(by[a]) for a in ORDER}
    return G, names, counts


def fig1():
    G, names, counts = affinity_matrix()
    fig = plt.figure(figsize=(10.6, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=.50,
                          left=.105, right=.955, top=.775, bottom=.255)

    # ---- panel A: heatmap ----
    ax = fig.add_subplot(gs[0, 0])
    lim = np.abs(G).max()
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    ax.imshow(G, cmap=CMAP, norm=norm, aspect="auto")
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            v = G[i, j]
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > .62 * lim else INK)
    ax.set_xticks(range(len(names)))
    SHORT = ["Framework\npreamble", "Zone 1\nmaximum", "Zone 2\ntwilight", "Zone 3\nlowest ebb",
             "Applying\nthe zones", "Rejecting\nArt. II", "Inherent /\nemergency", "Structural\nwarning"]
    ax.set_xticklabels(SHORT, fontsize=7, linespacing=1.3)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([f"{a}\n{ROLE[a]}, n={counts[a]}" for a in ORDER], fontsize=7.5,
                       linespacing=1.3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.set_title("A   Relative engagement with Jackson's concurrence",
                 loc="left", fontsize=10, fontweight="bold", pad=26)
    ax.text(0, 1.045, "red = leans on this passage more than the other opinions do; blue = less\n"
                      "cosine × 1000, double-centred",
            transform=ax.transAxes, fontsize=6.9, color=MUTED, linespacing=1.45,
            va="bottom")

    # ---- panel B: bootstrap contrasts ----
    ax2 = fig.add_subplot(gs[0, 1])
    boot = json.load(open(os.path.join(PROC, "contrasts.json")))
    labels = ["Zone machinery\n(zones 1-3)", "Structural warning\n(concentrated power)"]
    obs = [boot["c2"]["obs"], boot["c1"]["obs"]]
    lo = [boot["c2"]["lo"], boot["c1"]["lo"]]
    hi = [boot["c2"]["hi"], boot["c1"]["hi"]]
    ys = [1, 0]
    for y, o, l, h in zip(ys, obs, lo, hi):
        col = RED if o > 0 else BLUE
        ax2.plot([l, h], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax2.plot([o], [y], "o", ms=9, color=col, mec="white", mew=1.6, zorder=3)
        ax2.annotate(f"{o:+.1f}", (o, y), textcoords="offset points", xytext=(0, 13),
                     ha="center", fontsize=8, color=INK, fontweight="bold")
    ax2.axvline(0, color="#c9c8c2", lw=1, zorder=1)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(labels, fontsize=7.5, linespacing=1.3)
    ax2.set_ylim(-.6, 1.6)
    ax2.set_xlabel("majority − dissents  (cosine × 1000)", fontsize=7.5)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(length=0)
    ax2.set_title("B   Majority vs. dissents", loc="left",
                  fontsize=10, fontweight="bold", pad=26)
    ax2.text(0, 1.045, "dot = observed difference\nbar = 95% bootstrap CI (4,000 draws)",
             transform=ax2.transAxes, fontsize=6.9, color=MUTED, linespacing=1.45,
             va="bottom")

    fig.text(.105, .035,
             "Trump v. United States, 603 U.S. 593 (2024), scored against the 38 main-text paragraphs of Jackson's concurrence in\n"
             "Youngstown, 343 U.S. 579, 634-655 (1952). Embeddings: bge-small-en-v1.5. Double-centring removes hub paragraphs and\n"
             "each opinion's overall similarity to Youngstown. Signs are unchanged when quoted spans are deleted.",
             fontsize=6.5, color=MUTED, linespacing=1.6)
    fig.savefig(os.path.join(OUT, "figure1_engagement.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "figure1_engagement.pdf"))
    print("wrote figure1_engagement.{png,pdf}")


def fig2():
    data = json.load(open(os.path.join(PROC, "diagnostics.json")))
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    fig.subplots_adjust(left=.30, right=.95, top=.74, bottom=.26)
    labels = ["Doctrinal stance\nexpansive vs. restrictive",
              "Rhetorical role\nmajority vs. dissent",
              "Period\npre- vs. post-2000"]
    vals = [data["stance"], data["role"], data["era"]]
    cols = [MUTED, BLUE, BLUE]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=.52, color=cols, zorder=3)
    ax.axvline(.5, color="#c9c8c2", lw=1, zorder=2)
    ax.set_ylim(-.75, 2.5)
    ax.text(.5, -.68, "chance", fontsize=7, color=MUTED, ha="center", va="bottom")
    for yy, v in zip(y, vals):
        ax.annotate(f"{v:.2f}", (v, yy), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5, linespacing=1.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("leave-one-case-out AUC", fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("What the text can and cannot predict", loc="left",
                 fontsize=9.5, fontweight="bold", pad=16)
    ax.text(0, 1.10, "same features, same protocol, three targets",
            transform=ax.transAxes, fontsize=7, color=MUTED)
    fig.savefig(os.path.join(OUT, "figure2_limits.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "figure2_limits.pdf"))
    print("wrote figure2_limits.{png,pdf}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1()
    fig2()
