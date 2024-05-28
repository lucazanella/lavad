import matplotlib.pyplot as plt
import numpy as np


def plot_scores(scores, labels, video_name, save_dir, normal_id=7):
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 4))
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.06, right=0.99)

    x = np.arange(scores.shape[0])
    ax.plot(x, scores, color="#4e79a7", linewidth=1)
    ymin, ymax = 0, 1
    xmin, xmax = 0, scores.shape[0]
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    title = video_name
    title = title.replace("#", r"\#")
    title = title.replace("%", r"\%")
    title = title.replace("_", r"\_")
    title = title.replace("&", r"\&")
    title = title.replace("{", r"\{")
    title = title.replace("}", r"\}")
    title = title.replace("^", r"\^{}")

    start_idx = None
    for i in range(labels.shape[0]):
        if labels[i] != normal_id and start_idx is None:
            start_idx = i
        elif labels[i] == normal_id and start_idx is not None:
            rect = plt.Rectangle(
                (start_idx, ymin),
                i - start_idx,
                ymax - ymin,
                color="#e15759",
                alpha=0.5,
            )
            ax.add_patch(rect)
            start_idx = None
    if start_idx is not None:
        rect = plt.Rectangle(
            (start_idx, ymin),
            labels.shape[0] - start_idx,
            ymax - ymin,
            color="#e15759",
            alpha=0.5,
        )
        ax.add_patch(rect)

    ax.text(0.02, 0.90, title, fontsize=28, transform=ax.transAxes)
    for yline in [0.25, 0.5, 0.75]:
        ax.axhline(y=yline, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.tick_params(axis="y", labelsize=28)
    ax.tick_params(axis="x", labelsize=28)

    ax.set_ylabel("Anomaly score", fontsize=28)
    ax.set_xlabel("Frame number", fontsize=28)

    fig_file = save_dir / f"{video_name}_scores.png"
    plt.savefig(fig_file)
    plt.close()
