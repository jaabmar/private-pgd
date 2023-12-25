import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation

from mechanisms.ektelo_matrix import Identity
from mechanisms.privacy_calibrator import gaussian_mech


def measure_noisy_marginals(data, epsilon, delta, workload):
    delta_f = 2 * len(workload)
    b = delta_f / epsilon
    marginal_sensitivity = np.sqrt(2)
    sigma = gaussian_mech(epsilon, delta)["sigma"] * np.sqrt(len(workload))
    measurements = []
    for cl in workload:
        Q = Identity(data.domain.size(cl))
        x = data.project(cl).datavector()
        y = x + np.random.normal(
            loc=0, scale=marginal_sensitivity * sigma, size=x.size
        )
        measurements.append((Q, y, b, cl))

    return measurements


def visualize_measurements(data, measurements):
    for _, y, _, cl in measurements:
        # Extracting original counts
        original_counts = data.project(cl).datavector()
        # Extracting noisy counts
        noisy_counts = y

        # Create a figure for the subplots with enhanced aesthetics
        _, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

        # Customizing bar plots for original counts
        axes[0].bar(
            range(len(original_counts)),
            original_counts,
            color=sns.color_palette("Blues_d", n_colors=len(original_counts)),
        )
        axes[0].set_title(f"Original Counts for {cl}", fontweight="bold")
        axes[0].set_xlabel("Attribute Combinations", fontweight="bold")
        axes[0].set_ylabel("Counts", fontweight="bold")

        # Customizing bar plots for noisy counts
        axes[1].bar(
            range(len(noisy_counts)),
            noisy_counts,
            color=sns.color_palette("Reds_d", n_colors=len(noisy_counts)),
        )
        axes[1].set_title(f"Noisy Counts for {cl}", fontweight="bold")
        axes[1].set_xlabel("Attribute Combinations", fontweight="bold")
        axes[1].set_ylabel("Counts", fontweight="bold")

        plt.tight_layout()
        plt.show()


def visualize_quantization(tensor_data):
    # Extracting X1 and X2 coordinates from the tensor
    X1 = tensor_data[:, 0]
    X2 = tensor_data[:, 1]
    # Enhancing the scatter plot aesthetics with Seaborn
    plt.figure(figsize=(6, 4), dpi=100)
    sns.scatterplot(x=X1, y=X2, s=100, edgecolor="k", alpha=0.8)
    plt.title(
        "2D Scatter Plot of Quantized Data", fontsize=15, fontweight="bold"
    )
    plt.xlabel("X1", fontsize=12, fontweight="bold")
    plt.ylabel("X2", fontsize=12, fontweight="bold")
    # Customizing grid and layout for better appearance
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()


def visualize_projection_comparison(measurements, processed_measurements):
    for _, (_, y, _, cl) in enumerate(measurements):
        # Extracting noisy counts before projection
        noisy_counts_before = y

        # Extracting noisy counts after projection
        probability_distribution = np.array(processed_measurements)
        total_probability = np.sum(probability_distribution)

        # Create a figure for the subplots
        _, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

        # Plot the noisy counts before projection
        axes[0].bar(
            range(len(noisy_counts_before)),
            noisy_counts_before,
            color=sns.color_palette(
                "Reds_d", n_colors=len(noisy_counts_before)
            ),
        )
        axes[0].set_title(
            f"Noisy Counts Before Projection for {cl}", fontweight="bold"
        )
        axes[0].set_xlabel("Attribute Combinations", fontweight="bold")
        axes[0].set_ylabel("Counts", fontweight="bold")

        # Plot the noisy counts after projection with y-axis limits

        axes[1].bar(
            range(len(probability_distribution)),
            probability_distribution,
            color=sns.color_palette(
                "Greens_d", n_colors=len(probability_distribution)
            ),
        )
        axes[1].plot(
            range(len(probability_distribution)),
            probability_distribution,
            color="darkgreen",
            marker="o",
            linestyle="-",
            linewidth=0.2,
            markersize=3,
        )
        axes[1].set_title(
            f"Probability Distribution After Projection for {cl}",
            fontweight="bold",
        )
        axes[1].set_xlabel("Attribute Combinations", fontweight="bold")
        axes[1].set_ylabel("Probability", fontweight="bold")
        axes[1].set_ylim(
            -0.01, 0.05
        )  # Setting y-axis limits for the second plot

        # Annotating the total probability
        axes[1].annotate(
            f"Total Probability: {total_probability:.2f}",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=12,
            color="darkgreen",
        )

        plt.suptitle(
            "Visualization of Noisy Counts Before and After Projection",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return noisy_counts_before, probability_distribution


def animate_optimization_to_gif(
    particle_data,
    reference,
    filename="optimization_animation.gif",
    fps=50,
    pause_duration=3,
):
    fig, ax = plt.subplots()
    scatter_plot = ax.scatter(
        [],
        [],
        alpha=0.8,
        marker="o",
        s=100,
        edgecolors="k",
        label="Empirical Distribution",
    )
    _ = ax.scatter(
        reference[:, 0],
        reference[:, 1],
        alpha=0.3,
        marker="o",
        s=100,
        color="red",
        label="Quantized Private Data",
    )
    total_frames = (
        len(particle_data) + pause_duration * fps
    )  # Total frames including pause
    plt.subplots_adjust(top=0.90)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.15), shadow=True, ncol=2
    )

    ax.set_title("Optimization Process", fontweight="bold")
    ax.set_xlabel("X1", fontweight="bold")
    ax.set_ylabel("X2", fontweight="bold")
    ax.grid(True)

    iteration_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, backgroundcolor="white"
    )
    total_frames = max(len(particle_data), pause_duration * fps)

    def update(frame):
        if frame < len(particle_data):
            scatter_plot.set_offsets(particle_data[frame])
            iteration_text.set_text(f"Iteration: {frame+1}")
        return scatter_plot, iteration_text

    anim = FuncAnimation(fig, update, frames=total_frames, blit=True)

    # Save as GIF using Pillow
    anim.save(filename, writer="pillow", fps=fps)
    plt.close(fig)
    return filename
