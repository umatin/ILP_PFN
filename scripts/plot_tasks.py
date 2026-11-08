"""Plot example tasks from the prior: world graph, target labels, and rules.

    python scripts/plot_tasks.py --split iid --n 6 --seed 0 --out figures/tasks

Each figure has three panels:
  * World: constants (nodes), binary base facts (arrows, one colour + line style per predicate),
    unary base facts (predicate names under each node). This is what the model sees as facts.
  * Target labels: every tuple of the target predicate, coloured by label (positive / negative)
    and whether it is shown to the model or held out.
  * Rules: the BK rules (shown to the model), the derived facts they entail (not shown), and the
    true target rule (what the model must write).
Names are the anonymised tokens the model sees in this episode (p3, c17, ...).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ilp_pfn.logic import format_clause, true_tuples  # noqa: E402
from ilp_pfn.prior import sample_task  # noqa: E402
from ilp_pfn.splits import SPLITS, split_config  # noqa: E402
from ilp_pfn.tokenizer import encode  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
EDGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]       # categorical slots 1-3 (validated)
EDGE_STYLES = ["-", "--", ":"]                        # secondary encoding
# polarity -> diverging blue/red; shown = strong step, held-out = light step
LABEL_COLORS = {("pos", "shown"): "#2a78d6", ("pos", "held"): "#b7d3f6",
                ("neg", "shown"): "#e34948", ("neg", "held"): "#f6c9c8"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def const_order(task, enc):
    return sorted(range(task.n), key=lambda c: int(enc.const_tok[c][1:]))


def draw_world(ax, task, enc):
    n = task.n
    order = const_order(task, enc)
    pos = {c: (math.cos(2 * math.pi * i / n + math.pi / 2), math.sin(2 * math.pi * i / n + math.pi / 2))
           for i, c in enumerate(order)}
    binaries = [p for p in task.base_preds if task.arities[p] == 2]
    unaries = [p for p in task.base_preds if task.arities[p] == 1]
    handles = []
    for k, p in enumerate(binaries):
        color, style = EDGE_COLORS[k % 3], EDGE_STYLES[k % 3]
        rad = 0.12 + 0.1 * k
        for a, b in true_tuples(task.base_ext[p]):
            if a == b:  # self loop: small circle outside the node
                x, y = pos[a]
                r = 0.09 + 0.035 * k
                ax.add_patch(plt.Circle((x * (1.13 + 0.05 * k), y * (1.13 + 0.05 * k)), r,
                                        fill=False, lw=1.3, ls=style, color=color, zorder=1))
                continue
            ax.add_patch(FancyArrowPatch(pos[a], pos[b], connectionstyle=f"arc3,rad={rad}",
                                         arrowstyle="-|>", mutation_scale=9, lw=1.3, ls=style,
                                         color=color, shrinkA=11, shrinkB=11, zorder=1))
        handles.append(Line2D([], [], color=color, ls=style, lw=1.6,
                              label=f"{enc.pred_tok[p]}(X,Y): X→Y"))
    for c in order:
        x, y = pos[c]
        ax.scatter([x], [y], s=330, color=SURFACE, edgecolor=INK2, lw=1.2, zorder=3)
        ax.text(x, y, enc.const_tok[c], ha="center", va="center", fontsize=7.5, zorder=4)
        tags = [enc.pred_tok[p] for p in unaries if task.base_ext[p][c]]
        if tags:
            ax.text(x * 1.42, y * 1.42 - 0.02, ",".join(tags), ha="center", va="center",
                    fontsize=7.5, color=INK2, style="italic")
    if unaries:
        handles.append(Line2D([], [], ls="none", label="italic: unary facts "
                              + ", ".join(f"{enc.pred_tok[p]}(X)" for p in unaries)))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7.5,
              frameon=False, ncol=1, handlelength=2.4)
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("World: base facts (input)", loc="left", fontsize=10, color=INK)


def draw_labels(ax, task, enc):
    order = const_order(task, enc)
    names = [enc.const_tok[c] for c in order]
    shown_pos, shown_neg = set(task.pos), set(task.neg)
    a = task.target_arity
    rows = order if a == 2 else [None]
    for i, r in enumerate(rows):
        for j, c in enumerate(order):
            t = (r, c) if a == 2 else (c,)
            pol = "pos" if task.target_ext[t] else "neg"
            vis = "shown" if t in shown_pos or t in shown_neg else "held"
            ax.add_patch(plt.Rectangle((j, i), 0.92, 0.92, color=LABEL_COLORS[pol, vis], lw=0))
            if vis == "shown":
                ax.text(j + 0.46, i + 0.46, "+" if pol == "pos" else "−", ha="center",
                        va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.1, len(order))
    ax.set_ylim(len(rows), -0.1)
    ax.set_xticks([j + 0.46 for j in range(len(order))], names, rotation=90, fontsize=7.5)
    if a == 2:
        ax.set_yticks([i + 0.46 for i in range(len(order))], names, fontsize=7.5)
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        ax.set_aspect("equal")
    else:
        ax.set_yticks([])
        ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    title = "Target labels: T(X,Y)" if a == 2 else "Target labels: T(X)"
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    legend = [Patch(color=LABEL_COLORS["pos", "shown"], label=f"positive, shown (+) [{len(task.pos)}]"),
              Patch(color=LABEL_COLORS["pos", "held"], label=f"positive, held out [{len(task.query_pos)}]"),
              Patch(color=LABEL_COLORS["neg", "shown"], label=f"negative, shown (−) [{len(task.neg)}]"),
              Patch(color=LABEL_COLORS["neg", "held"], label=f"negative, held out [{len(task.query_neg)}]")]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0, -0.2 if a == 2 else -1.5),
              fontsize=7.5, frameon=False)


def draw_rules(ax, task, enc):
    name = lambda p: enc.pred_tok[p]  # noqa: E731
    lines = [("Background rules (input)", INK, "bold")]
    lines += [(format_clause(c, name), INK, "normal") for c in task.bk] or [("(none)", MUTED, "normal")]
    if task.derived_preds:
        lines += [("", INK, "normal"), ("Derived facts (NOT shown; model must infer)", INK, "bold")]
        for p in task.derived_preds:
            tup = true_tuples(task.bk_ext[p])
            txt, shown = f"{name(p)}:", 0
            for t in tup:  # keep each line short enough not to wrap
                item = " (" + ",".join(enc.const_tok[c] for c in t) + ")"
                if len(txt) + len(item) > 62:
                    break
                txt, shown = txt + item, shown + 1
            more = f" … +{len(tup) - shown} more" if shown < len(tup) else ""
            lines.append((txt + more, INK2, "normal"))
    lines += [("", INK, "normal"), ("Target rule (output to learn)", INK, "bold")]
    lines += [(format_clause(c, name), "#1c5cab", "bold") for c in task.target_clauses]
    m = task.meta
    lines += [("", INK, "normal"),
              (f"constants {m['n']} · base preds {m['n_base']} · derived {m['n_derived']}"
               f"{' (incl. transitive closure)' if m['has_tc'] else ''}", MUTED, "normal"),
              (f"uses derived: {m['uses_derived']} · reasoning depth {m['reasoning_depth']}"
               f" · positive rate {m['pos_rate']:.2f}", MUTED, "normal"),
              (f"prompt length {enc.rule_start} tokens · rule {len(enc.ids) - enc.rule_start} tokens",
               MUTED, "normal")]
    y = 1.0
    for text, color, weight in lines:
        ax.text(0, y, text, family="monospace", fontsize=8, color=color, weight=weight,
                va="top", transform=ax.transAxes)
        y -= 0.052
    ax.axis("off")


def plot_task(task, enc, title, path):
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 1.35], wspace=0.25)
    draw_world(fig.add_subplot(gs[0]), task, enc)
    draw_labels(fig.add_subplot(gs[1]), task, enc)
    draw_rules(fig.add_subplot(gs[2]), task, enc)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=INK, weight="bold")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def pick_diverse(rng, cfg, n):
    """Cycle through task 'kinds' so the examples show different features of the prior."""
    wants = [lambda t: t.target_arity == 1 and not t.meta["uses_derived"],
             lambda t: t.target_arity == 2 and t.meta["n_clauses"] == 1 and not t.meta["uses_derived"],
             lambda t: t.meta["uses_derived"] and t.meta["reasoning_depth"] >= 2,
             lambda t: t.meta["has_tc"] and any(l.pred in t.derived_preds and len(t.bk_ext[l.pred].shape) == 2
                                               for c in t.target_clauses for l in c.body),
             lambda t: t.meta["n_clauses"] == 2,
             lambda t: True]
    out = []
    for i in range(n):
        want = wants[i % len(wants)]
        for _ in range(5000):
            t = sample_task(rng, cfg)
            if want(t) and t.n <= 8:
                out.append((t, encode(t, rng)))
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="iid", choices=sorted(SPLITS))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/tasks")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i, (task, enc) in enumerate(pick_diverse(rng, split_config(args.split), args.n)):
        path = out / f"{args.split}_{i}.png"
        plot_task(task, enc, f"Example task {i + 1} ({args.split})", path)
        print(path)


if __name__ == "__main__":
    main()
