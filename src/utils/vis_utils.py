import os
import textwrap

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# plt.style.use('science')
import numpy as np

# import scienceplots


def find_closest_key_value(d, frame_idx):
    sorted_items = sorted(
        (int(key), dict(value)) for key, value in d.items() if int(key) <= frame_idx
    )
    return sorted_items[-1] if sorted_items else (None, None)


def visualize_video(
    video_name,
    video_labels,
    video_scores,
    video_captions,
    video_path,
    video_fps,
    save_path,
    normal_label,
    imagefile_template,
    optimal_threshold,
    font_size=18,
):
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])

    video_writer = None

    x = np.arange(len(video_scores))
    ax3.plot(x, video_scores, color="#4e79a7", linewidth=1)
    ymin, ymax = 0, 1
    xmin, xmax = 0, len(video_scores)
    ax3.set_xlim([xmin, xmax])
    ax3.set_ylim([ymin, ymax])
    title = video_name

    start_idx = None
    for frame_idx, label in enumerate(video_labels):
        if label != normal_label and start_idx is None:
            start_idx = frame_idx
        elif label == normal_label and start_idx is not None:
            rect = plt.Rectangle(
                (start_idx, ymin), frame_idx - start_idx, ymax - ymin, color="#e15759", alpha=0.5
            )
            ax3.add_patch(rect)
            start_idx = None

    if start_idx is not None:
        rect = plt.Rectangle(
            (start_idx, ymin),
            len(video_labels) - start_idx,
            ymax - ymin,
            color="#e15759",
            alpha=0.5,
        )
        ax3.add_patch(rect)

    ax3.text(0.02, 0.90, title, fontsize=16, transform=ax3.transAxes)
    for y_value in [0.25, 0.5, 0.75]:
        ax3.axhline(y=y_value, color="grey", linestyle="--", linewidth=0.8)

    ax3.set_yticks([0.25, 0.5, 0.75])
    ax3.tick_params(axis="y", labelsize=16)
    ax3.set_ylabel("Anomaly score", fontsize=font_size)
    ax3.set_xlabel("Frame number", fontsize=font_size)
    previous_line = None

    for i, score in enumerate(video_scores):
        ax1.set_title("Video frame", fontsize=font_size)
        ax2.set_title("Temporal summary", fontsize=font_size)

        img_name = imagefile_template.format(i)
        img_path = os.path.join(video_path, img_name)
        img = cv2.imread(img_path)

        if video_labels:
            box_color = (255, 0, 0) if score < optimal_threshold else (0, 0, 255)
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), box_color, 5)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(img)
        ax1.axis("off")

        # Display captions in a box on ax2
        clip_frame_idx, clip_caption = find_closest_key_value(video_captions, i)
        frame_caption = clip_caption.get(str(0), "")
        wrapped_caption = textwrap.fill(frame_caption, width=35)  # Adjust the width as needed

        ax2.text(
            0.5,
            0.5,
            wrapped_caption,
            fontsize=18,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                boxstyle="round",
                pad=0.5,
                edgecolor="black",
                linewidth=2,
            ),
            transform=ax2.transAxes,
            wrap=True,
        )
        ax2.axis("off")

        # Update or create the axvline
        if previous_line is not None:
            # Clear previous axvline
            previous_line.remove()

        axvline = ax3.axvline(x=i, color="red")

        fig.tight_layout()

        if video_writer is None:
            fig_size = fig.get_size_inches() * fig.dpi
            video_width, video_height = int(fig_size[0]), int(fig_size[1])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(save_path), fourcc, video_fps, (video_width, video_height)
            )

        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video_writer.write(img)

        ax1.cla()
        ax2.cla()

        # Update previous_line
        previous_line = axvline

    plt.close()
    video_writer.release()
    cv2.destroyAllWindows()
