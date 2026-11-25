"""Strict-F1-only version of 01_headline_scores.png for both datasets side by side:
left = 9 pk-individual papers, right = 9 pk-summary papers. Each panel lists its own
complete runs ordered by strict F1 (the run sets differ between the datasets). Values and
run labels come from the two 20261031-cascade-qwen38 snapshots; style (tokens, palette,
rounded bars, header/footer helpers) is copied from their figures/make_figures_cascade.py.

pk-individual: f1_strict_macro over the runs with all 9 papers (same selection as its
01_headline_scores.png, simple_gpt4o_20260908 duplicate excluded). pk-summary:
f1_strict_macro_missing_as_zero (its 01 convention: a missing paper counts as 0, the two
incomplete skills runs are drawn faded and labelled).
"""
import os, textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

ROOT = "/fs/scratch/PCON0100/feng1426/projects/playground/tabular-data-benchmark"
PKI = f"{ROOT}/pk-individual/benchmark_results/20261031-cascade-qwen38"
PKS = f"{ROOT}/pk-summary/benchmark_results/20261031-cascade-qwen38"
OUT = os.path.dirname(os.path.abspath(__file__))
NAME = "01_f1_headline_pk_individual_vs_pk_summary.png"

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
MODE = {"Multi-agent": "#2a78d6", "Pipeline": "#eb6834", "Skills": "#1baf7a", "Simple": "#eda100", "Cascade": "#e87ba4"}
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "font.size": 10})

RUNS_PKI = {
 "multiagent_gpt54": ("Multi-agent  gpt-5.4 (B)", "Multi-agent"),
 "multiagent_gpt4o_gpt54": ("Multi-agent  gpt-4o + gpt-5.4 (A)", "Multi-agent"),
 "multiagent_qwen36_pertable": ("Multi-agent  qwen3.6 per-table", "Multi-agent"),
 "multiagent_gemma4": ("Multi-agent  gemma4:31b", "Multi-agent"),
 "multiagent_qwen38": ("Multi-agent  qwen3.8", "Multi-agent"),
 "cascade_qwen38": ("Cascade  qwen3.8", "Cascade"),
 "skills_qwen36": ("Skills  qwen3.6", "Skills"),
 "skills_qwen38_aug26": ("Skills  qwen3.8 (26 Aug run)", "Skills"),
 "skills_qwen38": ("Skills  qwen3.8 (2 Oct run)", "Skills"),
 "pipeline_qwen36_final": ("Pipeline  qwen3.6", "Pipeline"),
 "pipeline_gpt4o": ("Pipeline  gpt-4o", "Pipeline"),
 "pipeline_gemma4": ("Pipeline  gemma4:31b", "Pipeline"),
 "pipeline_gpt54": ("Pipeline  gpt-5.4", "Pipeline"),
 "pipeline_qwen38": ("Pipeline  qwen3.8", "Pipeline"),
 "simple_gpt4o": ("Simple  gpt-4o", "Simple"),
 "simple_qwen36": ("Simple  qwen3.6", "Simple"),
 "simple_gpt54": ("Simple  gpt-5.4", "Simple"),
 "simple_qwen38": ("Simple  qwen3.8", "Simple"),
 "simple_gemma4": ("Simple  gemma4:31b", "Simple"),
}
RUNS_PKS = {
 "multiagent_gpt54": ("Multi-agent  gpt-5.4", "Multi-agent"),
 "multiagent_gpt4o_gpt54": ("Multi-agent  gpt-4o + gpt-5.4", "Multi-agent"),
 "multiagent_qwen36": ("Multi-agent  qwen3.6", "Multi-agent"),
 "multiagent_gemma4": ("Multi-agent  gemma4:31b", "Multi-agent"),
 "multiagent_qwen38": ("Multi-agent  qwen3.8", "Multi-agent"),
 "cascade_qwen38": ("Cascade  qwen3.8", "Cascade"),
 "skills_qwen36": ("Skills  qwen3.6  (8 of 9 papers)", "Skills"),
 "skills_qwen38": ("Skills  qwen3.8", "Skills"),
 "skills_gemma4": ("Skills  gemma4:31b  (8 of 9 papers)", "Skills"),
 "pipeline_qwen36": ("Pipeline  qwen3.6", "Pipeline"),
 "pipeline_gpt4o": ("Pipeline  gpt-4o", "Pipeline"),
 "pipeline_gemma4": ("Pipeline  gemma4:31b", "Pipeline"),
 "pipeline_qwen38": ("Pipeline  qwen3.8", "Pipeline"),
 "pipeline_gpt54": ("Pipeline  gpt-5.4", "Pipeline"),
 "simple_gpt4o": ("Simple  gpt-4o", "Simple"),
 "simple_qwen36": ("Simple  qwen3.6", "Simple"),
 "simple_gemma4": ("Simple  gemma4:31b", "Simple"),
 "simple_qwen38": ("Simple  qwen3.8", "Simple"),
 "simple_gpt54": ("Simple  gpt-5.4", "Simple"),
}
S_PKI = pd.read_csv(f"{PKI}/scores_summary.csv").set_index("set")
S_PKS = pd.read_csv(f"{PKS}/scores_summary.csv").set_index("set")
keys_pki = [k for k in RUNS_PKI if S_PKI.loc[k, "papers_with_output"] == 9]
keys_pki.sort(key=lambda k: -S_PKI.loc[k, "f1_strict_macro"])
keys_pks = sorted(RUNS_PKS, key=lambda k: -S_PKS.loc[k, "f1_strict_macro_missing_as_zero"])
incomplete_pks = [k for k in RUNS_PKS if S_PKS.loc[k, "papers_with_output"] < 9]

def rbar(ax, y, val, h, color, radius_px=4, alpha=1.0):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); bb = ax.get_window_extent()
    px_x, px_y = bb.width / abs(x1 - x0), bb.height / abs(y1 - y0)
    r = radius_px * (200 / 96) / px_x
    ax.add_patch(FancyBboxPatch((-3 * r, y - h / 2), val + 3 * r, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=px_x / px_y, fc=color, ec="none", lw=0, clip_on=True, alpha=alpha))
def style_bar_axes(ax):
    ax.set_xlim(0, 1); ax.set_xticks([0, .2, .4, .6, .8, 1]); ax.set_xticklabels([f"{t:.1f}" for t in [0, .2, .4, .6, .8, 1]], fontsize=9.5, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0); ax.axvline(0, color=AXIS, lw=0.9, zorder=1)

n = max(len(keys_pki), len(keys_pks)); pitch = 0.5
W = 16.0; top_in, bot_in = 2.1, 1.45
H = top_in + n * pitch + bot_in
fig = plt.figure(figsize=(W, H)); axh = (H - top_in - bot_in) / H
# two panels, each with its own label column to its left
panels = [(fig.add_axes([0.215, bot_in / H, 0.235, axh]), keys_pki, RUNS_PKI, S_PKI, "f1_strict_macro", [],
           "pk-individual: F1, strict (0-1), 9 papers"),
          (fig.add_axes([0.715, bot_in / H, 0.235, axh]), keys_pks, RUNS_PKS, S_PKS, "f1_strict_macro_missing_as_zero", incomplete_pks,
           "pk-summary: F1, strict (0-1), 9 papers")]
for ax, keys, runs, S, col, faded, ttl in panels:
    ax.set_ylim(n - 0.5, -0.5); style_bar_axes(ax)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([])
    ax.set_title(ttl, fontsize=10.5, color=INK2, loc="left", pad=8, fontweight="bold")
    fig.canvas.draw()
    bh = 0.22 * fig.dpi / (ax.get_window_extent().height / n)
    for i, k in enumerate(keys):
        v = S.loc[k, col]
        rbar(ax, i, v, bh, MODE[runs[k][1]], alpha=0.45 if k in faded else 1.0)
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", ha="left", fontsize=9.5, color=INK,
                fontweight="bold" if k == "cascade_qwen38" else "normal")
        ax.text(-0.03, i, runs[k][0], ha="right", va="center", fontsize=10.5, color=INK, transform=ax.transData,
                fontweight="bold" if k == "cascade_qwen38" else "normal")

def hdr(title, sub, x=0.03, top=0.965):
    fig.text(x, top, title, fontsize=15, fontweight="bold", ha="left", va="top", color=INK)
    wrapped = "\n".join(textwrap.wrap(sub, width=int((0.97 - x) * W * 12.6), break_on_hyphens=False))
    fig.text(x, top - 0.36 / H, wrapped, fontsize=10.5, ha="left", va="top", color=INK2, linespacing=1.4)
def ftr(text, x=0.03, y=0.012):
    width = int((0.97 - x) * W * 15.5)
    wrapped = "\n".join("\n".join(textwrap.wrap(p, width=width, break_on_hyphens=False)) for p in text.split("\n"))
    fig.text(x, y, wrapped, fontsize=8.5, ha="left", va="bottom", color=MUTED, linespacing=1.35)

c_i, c_s = S_PKI.loc["cascade_qwen38", "f1_strict_macro"], S_PKS.loc["cascade_qwen38", "f1_strict_macro_missing_as_zero"]
hdr("Row-level F1 (strict) by run: pk-individual (left) and pk-summary (right)",
    f"Each bar is the macro mean of per-paper strict F1 over the 9 gold papers of that dataset; each panel is ordered by its own F1, "
    f"so the run lists differ. Cascade qwen3.8 (bold) leads both: {c_i:.3f} on pk-individual (rank {keys_pki.index('cascade_qwen38') + 1} of {len(keys_pki)}) "
    f"and {c_s:.3f} on pk-summary (rank {keys_pks.index('cascade_qwen38') + 1} of {len(keys_pks)}).")
handles = [Line2D([0], [0], marker="s", ls="", ms=9, mfc=c, mec="none", label=m) for m, c in MODE.items()]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 1 - 1.45 / H), ncol=5, frameon=False, fontsize=10,
           labelcolor=INK2, handletextpad=0.3, columnspacing=1.6, handlelength=1.1)
ftr("Cascade qwen3.8 (2026-10-31) = per-table simple prompt, a 3-tier validity gate, and the pipeline workflow only on tables the gate rejects "
    "(pk-individual: round 7, job 8242232, worst-tier gate, 2 of 9 papers escalated; pk-summary: round 6, job 8241591, 5 of 9 escalated) - the same qwen3.8:27b model for every call.\n"
    "pk-individual shows only runs with all 9 papers (excluded: the qwen3.6 pipeline variants that lost a paper, the qwen3.8 skills pilot, skills gemma4 with 7 of 9). "
    "pk-summary counts a missing paper as 0 and fades the two incomplete skills runs (8 of 9); its baseline paper 30950674 has no output in any run and is excluded. "
    "F1 strict counts a matched row as correct when >= 80% of the rating weight agrees (benchmark/test_f1_with_semantic.py, optimal identity-gated matching).\n"
    "Values: f1_strict_macro in pk-individual/.../20261031-cascade-qwen38/scores_summary.csv and f1_strict_macro_missing_as_zero in pk-summary/.../20261031-cascade-qwen38/scores_summary.csv.")
fig.savefig(f"{OUT}/{NAME}", dpi=200); plt.close(fig); print("wrote", f"{OUT}/{NAME}")
