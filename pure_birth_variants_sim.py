#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

VARIANTS: Dict[str, str] = {
    "a": "(a) N -> N + 1 w.p. p; else N",
    "b": "(b) N -> 2N w.p. p; else N",
    "c": "(c) N -> N + 1 w.p. p; else N - 1",
    "d": "(d) N -> 2N w.p. p; else N - 1",
}


def step_update(n: np.ndarray, births: np.ndarray, variant: str) -> np.ndarray:
    """Apply one-step transition for a chosen variant."""
    if variant == "a":
        n = n + births.astype(np.float64)
    elif variant == "b":
        n = np.where(births, 2.0 * n, n)
    elif variant == "c":
        n = n + np.where(births, 1.0, -1.0)
        n = np.maximum(n, 0.0)
    elif variant == "d":
        n = np.where(births, 2.0 * n, n - 1.0)
        n = np.maximum(n, 0.0)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return n


def simulate_variant(
    variant: str,
    p: float,
    trials: int,
    steps: int,
    rng: np.random.Generator,
    max_value: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run trials and return (mean_t, var_t, divergence_times)."""
    n = np.ones(trials, dtype=np.float64)
    history = np.full((steps + 1, trials), np.nan, dtype=np.float64)
    history[0, :] = n

    divergence_times = np.full(trials, np.nan, dtype=np.float64)
    active = np.ones(trials, dtype=bool)

    for t in range(1, steps + 1):
        active_idx = np.where(active)[0]
        if active_idx.size == 0:
            break

        births = rng.random(active_idx.size) < p
        updated = step_update(n[active_idx], births, variant)

        exploded = (~np.isfinite(updated)) | (updated > max_value)
        if np.any(exploded):
            exploded_global_idx = active_idx[exploded]
            divergence_times[exploded_global_idx] = t
            active[exploded_global_idx] = False
            updated[exploded] = np.nan

        n[active_idx] = updated
        history[t, :] = n

    valid = np.isfinite(history)
    counts = np.sum(valid, axis=1)

    sums = np.nansum(history, axis=1)
    mean_t = np.full(steps + 1, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean_t, where=counts > 0)

    centered = np.where(valid, history - mean_t[:, None], 0.0)
    sumsq = np.sum(centered * centered, axis=1)
    var_t = np.full(steps + 1, np.nan, dtype=np.float64)
    np.divide(sumsq, counts, out=var_t, where=counts > 0)
    return mean_t, var_t, divergence_times


def summarize_results(
    results: Dict[str, Dict[float, Dict[str, np.ndarray]]],
    steps: int,
) -> None:
    """Print concise numeric summaries to stdout."""
    print("\nSummary at final simulated time:")
    print("variant | p   | mean(N_T) | var(N_T) | diverged trials")
    print("-" * 60)
    for variant in VARIANTS:
        for p in sorted(results[variant].keys()):
            mean_t = results[variant][p]["mean"]
            var_t = results[variant][p]["var"]
            div = results[variant][p]["divergence_times"]
            diverged = np.sum(~np.isnan(div))
            print(
                f"{variant:>7} | {p:0.1f} | {mean_t[steps]:10.4g} |"
                f" {var_t[steps]:10.4g} | {int(diverged):14d}"
            )


def plot_mean_variance(
    results: Dict[str, Dict[float, Dict[str, np.ndarray]]],
    p_values: List[float],
    steps: int,
    output_dir: Path,
) -> None:
    """Save mean and variance trajectories for all variants."""
    t = np.arange(steps + 1)

    fig_mean, axes_mean = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    fig_var, axes_var = plt.subplots(4, 1, figsize=(11, 14), sharex=True)

    for i, variant in enumerate(VARIANTS):
        axm = axes_mean[i]
        axv = axes_var[i]
        for p in p_values:
            axm.plot(t, results[variant][p]["mean"], label=f"p={p}", linewidth=1.8)
            axv.plot(t, results[variant][p]["var"], label=f"p={p}", linewidth=1.8)

        axm.set_title(VARIANTS[variant])
        axm.set_ylabel("mean N_t")
        axm.grid(alpha=0.3)
        axm.legend(loc="upper left")

        axv.set_title(VARIANTS[variant])
        axv.set_ylabel("var N_t")
        axv.grid(alpha=0.3)
        axv.legend(loc="upper left")

        if variant in {"b", "d"}:
            axm.set_yscale("log")
            axv.set_yscale("log")

    axes_mean[-1].set_xlabel("time step t")
    axes_var[-1].set_xlabel("time step t")

    fig_mean.tight_layout()
    fig_var.tight_layout()

    fig_mean.savefig(output_dir / "mean_vs_time.png", dpi=160)
    fig_var.savefig(output_dir / "variance_vs_time.png", dpi=160)
    plt.close(fig_mean)
    plt.close(fig_var)


def plot_divergence_histograms(
    results: Dict[str, Dict[float, Dict[str, np.ndarray]]],
    p_values: List[float],
    steps: int,
    output_dir: Path,
) -> None:
    """Save divergence-time histograms for each (variant, p)."""
    fig, axes = plt.subplots(4, len(p_values), figsize=(4.5 * len(p_values), 13), sharex=True)

    if len(p_values) == 1:
        axes = np.array(axes).reshape(4, 1)

    for i, variant in enumerate(VARIANTS):
        for j, p in enumerate(p_values):
            ax = axes[i, j]
            div = results[variant][p]["divergence_times"]
            div = div[~np.isnan(div)]

            if div.size == 0:
                ax.text(0.5, 0.5, "No divergence", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0, steps)
            else:
                bins = min(60, steps)
                ax.hist(div, bins=bins, color="#3E79B5", alpha=0.85)

            if i == 0:
                ax.set_title(f"p={p}")
            if j == 0:
                ax.set_ylabel(f"variant {variant}")
            if i == 3:
                ax.set_xlabel("divergence time")

            ax.grid(alpha=0.25)

    fig.suptitle("Divergence-Time Distributions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "divergence_times.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate birth/death process variants")
    parser.add_argument("--trials", type=int, default=1000, help="Number of trials per (variant, p)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of time steps")
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.8],
        help="List of p values to run",
    )
    parser.add_argument("--seed", type=int, default=460, help="Random seed")
    parser.add_argument(
        "--max-value",
        type=float,
        default=1e100,
        help=(
            "Practical cap for declaring divergence (default: 1e100). "
            "Use ~1e308 to match float64 hard limit."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sim_outputs"),
        help="Directory for saved plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (in addition to saving)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.trials < 1000 or args.steps < 1000:
        print("Warning: assignment asks for at least 1000 trials and 1000 steps.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    results: Dict[str, Dict[float, Dict[str, np.ndarray]]] = {v: {} for v in VARIANTS}

    for variant in VARIANTS:
        for p in args.p_values:
            mean_t, var_t, div_t = simulate_variant(
                variant=variant,
                p=p,
                trials=args.trials,
                steps=args.steps,
                rng=rng,
                max_value=args.max_value,
            )
            results[variant][p] = {
                "mean": mean_t,
                "var": var_t,
                "divergence_times": div_t,
            }

    summarize_results(results, args.steps)
    plot_mean_variance(results, args.p_values, args.steps, args.output_dir)
    plot_divergence_histograms(results, args.p_values, args.steps, args.output_dir)

    print(f"\nSaved plots to: {args.output_dir.resolve()}")
    print("  - mean_vs_time.png")
    print("  - variance_vs_time.png")
    print("  - divergence_times.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
