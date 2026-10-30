from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


DIMENSION = 500
NOISE_COEFFICIENT = 0.1
DEFAULT_LEARNING_RATE_MAX = 1.0
INITIAL_BETA1 = 0.9
BETA1_DECAY = 0.99
BETA2 = 0.999
EPSILON = 1e-8
TRIALS = 20
SEED = 42
OUTPUT_DIR = Path("decbeta")

HEATMAP_STEPS = 4000
HEATMAP_DECAY_INTERVAL = 400
HEATMAP_TOP_COMBINATIONS = 35
HEATMAP_H_VALUES = np.linspace(0.1, 1.0, 10)
HEATMAP_ALPHA_VALUES = np.linspace(1.1, 2.0, 10)

CONV_TRAJ_STEPS = 1000
CONV_TRAJ_DECAY_INTERVAL = 100
CONV_TRAJ_H_VALUES = (0.1, 0.3, 0.6, 1.0)
CONV_TRAJ_ALPHA_VALUES = (1.1, 1.3, 1.6, 2.0)


def sample_noise(alpha, size, rng):
    if alpha == 2.0:
        return rng.normal(loc=0.0, scale=1.0, size=size)
    if not 1.0 < alpha < 2.0:
        raise ValueError("alpha must satisfy 1 < alpha <= 2")

    signs = rng.choice((-1.0, 1.0), size=size)
    uniform = rng.uniform(1e-12, 1.0, size=size)
    return signs * uniform ** (-1.0 / alpha)


def beta1_schedule(
    step,
    decay_interval,
    initial_beta1=INITIAL_BETA1,
    decay=BETA1_DECAY,
):
    if step < 1:
        raise ValueError("step must be positive")
    if decay_interval < 1:
        raise ValueError("decay_interval must be positive")
    return initial_beta1 * decay ** (step // decay_interval)


class HolderObjective:
    def __init__(self, h):
        if not 0.0 < h <= 1.0:
            raise ValueError("h must satisfy 0 < h <= 1")
        self.h = h

    def holder_integral(self, x):
        abs_x = np.abs(x)
        return np.where(
            abs_x <= 1.0,
            abs_x ** (1.0 + self.h) / (1.0 + self.h),
            abs_x - self.h / (1.0 + self.h),
        )

    def value(self, x):
        return np.sum(
            self.holder_integral(x)
            - 0.5 * np.cos(2.0 * np.pi * x)
            + 0.5
        )

    def gradient(self, x):
        holder_gradient = np.sign(x) * np.minimum(np.abs(x) ** self.h, 1.0)
        return holder_gradient + np.pi * np.sin(2.0 * np.pi * x)


def run_adam(
    h,
    alpha,
    learning_rate_max,
    steps,
    decay_interval,
    dim=DIMENSION,
    noise_coefficient=NOISE_COEFFICIENT,
    initial_beta1=INITIAL_BETA1,
    beta1_decay=BETA1_DECAY,
    beta2=BETA2,
    epsilon=EPSILON,
    record_every=10,
    seed=None,
):
    if dim < 1:
        raise ValueError("dim must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    if learning_rate_max <= 0:
        raise ValueError("learning_rate_max must be positive")
    if record_every < 1:
        raise ValueError("record_every must be positive")

    rng = np.random.default_rng(seed)
    objective = HolderObjective(h)
    x = rng.normal(loc=0.0, scale=1.0, size=dim)
    first_moment = np.zeros_like(x)
    second_moment = np.zeros_like(x)
    beta1_product = 1.0
    learning_rate_exponent = 1.0 / (1.0 + h)
    iterations = []
    gradient_norms = []

    for step in range(1, steps + 1):
        full_gradient = objective.gradient(x)
        stochastic_gradient = full_gradient + noise_coefficient * sample_noise(
            alpha, size=x.shape, rng=rng
        )
        learning_rate = learning_rate_max * step ** (-learning_rate_exponent)
        beta1 = beta1_schedule(
            step,
            decay_interval,
            initial_beta1=initial_beta1,
            decay=beta1_decay,
        )

        first_moment = (
            beta1 * first_moment + (1.0 - beta1) * stochastic_gradient
        )
        second_moment = (
            beta2 * second_moment + (1.0 - beta2) * stochastic_gradient**2
        )

        beta1_product *= beta1
        corrected_first_moment = first_moment / (1.0 - beta1_product)
        corrected_second_moment = second_moment / (1.0 - beta2**step)
        x = x - (
            learning_rate
            * corrected_first_moment
            / (np.sqrt(corrected_second_moment) + epsilon)
        )

        if step == 1 or step % record_every == 0 or step == steps:
            iterations.append(step)
            gradient_norms.append(np.linalg.norm(objective.gradient(x)))

    return np.asarray(iterations), np.asarray(gradient_norms)


def summarize_trials(trial_values, iterations):
    values = np.vstack(trial_values)
    return {
        "iterations": iterations,
        "mean": values.mean(axis=0),
        "minimum": values.min(axis=0),
        "maximum": values.max(axis=0),
        "raw": values,
    }


def run_trials(
    h,
    alpha,
    learning_rate_max=DEFAULT_LEARNING_RATE_MAX,
    dim=DIMENSION,
    steps=CONV_TRAJ_STEPS,
    decay_interval=CONV_TRAJ_DECAY_INTERVAL,
    trials=TRIALS,
    noise_coefficient=NOISE_COEFFICIENT,
    record_every=10,
    seed=SEED,
):
    seed_rng = np.random.default_rng(seed)
    trial_seeds = seed_rng.integers(1 << 30, size=trials)
    trial_values = []

    for trial_seed in trial_seeds:
        iterations, values = run_adam(
            h=h,
            alpha=alpha,
            learning_rate_max=learning_rate_max,
            steps=steps,
            decay_interval=decay_interval,
            dim=dim,
            noise_coefficient=noise_coefficient,
            record_every=record_every,
            seed=trial_seed,
        )
        trial_values.append(values)

    return summarize_trials(trial_values, iterations)


def plot_trajectory(results, h, output_file):
    fig, ax = plt.subplots(figsize=(6, 4))
    all_means = []

    for alpha, result in results.items():
        iterations = result["iterations"]
        mean = result["mean"]
        all_means.append(mean)
        ax.semilogy(iterations, mean, label=fr"$\alpha$={alpha}")
        ax.fill_between(
            iterations,
            np.maximum(result["minimum"], 1e-16),
            result["maximum"],
            alpha=0.2,
        )

    minimum_mean = min(np.min(values) for values in all_means)
    maximum_mean = max(np.max(values) for values in all_means)
    ax.set_ylim(bottom=max(0.5 * minimum_mean, 1e-12), top=2.0 * maximum_mean)
    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel(r"Full gradient norm $\|\nabla f(x)\|$", fontsize=14)
    ax.set_title(fr"ADAM: $h$={h}, beta1 decay", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=800)
    plt.close(fig)


def run_conv_traj(
    h_values=CONV_TRAJ_H_VALUES,
    alpha_values=CONV_TRAJ_ALPHA_VALUES,
    learning_rate_max=DEFAULT_LEARNING_RATE_MAX,
    dim=DIMENSION,
    steps=CONV_TRAJ_STEPS,
    decay_interval=CONV_TRAJ_DECAY_INTERVAL,
    trials=TRIALS,
    noise_coefficient=NOISE_COEFFICIENT,
    record_every=10,
    seed=SEED,
    output_dir=OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_results = {}
    output_files = []
    started = time.time()

    for h in h_values:
        print(f"\nConvergence trajectory: h={h}, learning_rate_max={learning_rate_max}")
        results = {}
        for alpha in alpha_values:
            print(f"  alpha={alpha} ...", end="", flush=True)
            result = run_trials(
                h=h,
                alpha=alpha,
                learning_rate_max=learning_rate_max,
                dim=dim,
                steps=steps,
                decay_interval=decay_interval,
                trials=trials,
                noise_coefficient=noise_coefficient,
                record_every=record_every,
                seed=seed,
            )
            results[alpha] = result
            print(" done")

        output_file = output_dir / (
            f"conv_traj_ADAM,h={h},lr_max={learning_rate_max},beta1_decay.pdf"
        )
        plot_trajectory(results, h=h, output_file=output_file)
        figure_results[h] = results
        output_files.append(output_file)
        print(f"Saved plot to {output_file}")

    print(f"Convergence trajectories finished in {time.time() - started:.1f}s")
    return figure_results, output_files


def compute_heatmap_data(
    h_values=HEATMAP_H_VALUES,
    alpha_values=HEATMAP_ALPHA_VALUES,
    learning_rate_max=DEFAULT_LEARNING_RATE_MAX,
    dim=DIMENSION,
    steps=HEATMAP_STEPS,
    decay_interval=HEATMAP_DECAY_INTERVAL,
    trials=TRIALS,
    noise_coefficient=NOISE_COEFFICIENT,
    seed=SEED,
    output_dir=OUTPUT_DIR,
):
    h_values = np.asarray(h_values, dtype=float)
    alpha_values = np.asarray(alpha_values, dtype=float)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_gradient_norms = np.zeros((len(h_values), len(alpha_values)))
    started = time.time()

    print(f"Computing heatmap data, learning_rate_max={learning_rate_max}")
    for row, h in enumerate(h_values):
        for column, alpha in enumerate(alpha_values):
            result = run_trials(
                h=float(h),
                alpha=float(alpha),
                learning_rate_max=learning_rate_max,
                dim=dim,
                steps=steps,
                decay_interval=decay_interval,
                trials=trials,
                noise_coefficient=noise_coefficient,
                record_every=steps,
                seed=seed,
            )
            final_gradient_norms[row, column] = result["mean"][-1]
        print(f"  h={h:.1f} done")

    output_file = output_dir / (
        f"heatmap_data,lr_max={learning_rate_max}.npz"
    )
    np.savez(
        output_file,
        data=final_gradient_norms,
        hs=h_values,
        alphas=alpha_values,
        learning_rate_max=learning_rate_max,
        dimension=dim,
        steps=steps,
        trials=trials,
        noise_coefficient=noise_coefficient,
        initial_beta1=INITIAL_BETA1,
        beta1_decay=BETA1_DECAY,
        beta1_decay_interval=decay_interval,
        beta2=BETA2,
    )
    print(f"Saved data to {output_file} ({time.time() - started:.1f}s)")
    return output_file


def plot_heatmap(data_file, output_file=None, top_n=HEATMAP_TOP_COMBINATIONS):
    if top_n < 1:
        raise ValueError("top_n must be positive")

    data_file = Path(data_file)
    with np.load(data_file) as loaded:
        data = loaded["data"]
        h_values = loaded["hs"]
        alpha_values = loaded["alphas"]
        learning_rate_max = float(loaded["learning_rate_max"])

    positive_mask = np.isfinite(data) & (data > 0)
    if not np.any(positive_mask):
        raise ValueError("heatmap data has no positive finite values")

    positive = data[positive_mask]
    vmin = positive.min()
    vmax = positive.max()
    if vmax <= vmin:
        vmax = vmin * (1.0 + 1e-12)
    plot_data = np.where(positive_mask, data, vmin)

    valid_flat_indices = np.flatnonzero(positive_mask.ravel())
    count = min(top_n, valid_flat_indices.size)
    order = np.argsort(data.ravel()[valid_flat_indices])[:count]
    selected_flat_indices = valid_flat_indices[order]
    threshold = data.ravel()[selected_flat_indices].max()
    selected_rows, selected_columns = np.unravel_index(
        selected_flat_indices, data.shape
    )

    fig, ax = plt.subplots(figsize=(8, 6.4))
    mesh = ax.pcolormesh(
        alpha_values,
        h_values,
        plot_data,
        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        shading="auto",
    )
    ax.scatter(
        alpha_values[selected_columns],
        h_values[selected_rows],
        color="red",
        marker="o",
        s=80,
        edgecolors="white",
        zorder=5,
    )

    line_alpha = np.linspace(alpha_values.min(), alpha_values.max(), 200)
    ax.plot(
        line_alpha,
        line_alpha - 1.0,
        color="red",
        linestyle="--",
        linewidth=4,
        zorder=6,
    )
    ax.set_xlim(alpha_values.min() - 0.05, alpha_values.max() + 0.05)
    ax.set_ylim(h_values.min() - 0.05, h_values.max() + 0.05)
    ax.set_xlabel(r"$\alpha$", fontsize=23)
    ax.set_ylabel(r"$\mathfrak{h}$", fontsize=23)
    ax.set_title(
        fr"ADAM: beta1 decay(Marker:$\leq${threshold:.1e})",
        fontsize=23,
    )
    ax.set_xticks(alpha_values)
    ax.set_yticks(h_values)
    ax.tick_params(axis="both", labelsize=18)

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(r"Average final $\|\nabla f(x)\|$", fontsize=20)
    colorbar.ax.tick_params(labelsize=18)
    fig.tight_layout()

    if output_file is None:
        output_file = data_file.with_name(
            f"heatmap,lr_max={learning_rate_max}.pdf"
        )
    output_file = Path(output_file)
    fig.savefig(output_file, dpi=800)
    plt.close(fig)
    print(f"Saved heatmap to {output_file}")
    return output_file


def run_heatmap(output_dir=OUTPUT_DIR, **experiment_kwargs):
    data_file = compute_heatmap_data(
        output_dir=output_dir,
        **experiment_kwargs,
    )
    plot_file = plot_heatmap(data_file)
    return data_file, plot_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run heatmap and convergence-trajectory experiments."
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default="conv_traj",
        choices=("heatmap", "conv_traj", "all"),
        help="experiment to run (default: conv_traj)",
    )
    parser.add_argument(
        "--learning-rate-max",
        type=float,
        default=DEFAULT_LEARNING_RATE_MAX,
        help=(
            "coefficient in lr_k = learning_rate_max*k^(-1/(1+h)) "
            "(default: 1.0)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for generated PDF and NPZ files (default: decbeta)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.experiment in ("heatmap", "all"):
        run_heatmap(
            learning_rate_max=args.learning_rate_max,
            output_dir=args.output_dir,
        )
    if args.experiment in ("conv_traj", "all"):
        run_conv_traj(
            learning_rate_max=args.learning_rate_max,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
