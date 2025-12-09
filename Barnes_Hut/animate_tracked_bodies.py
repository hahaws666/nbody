import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# This script reads tracked_bodies.dat and makes an animation
# Columns: iter time body_index x y vx vy mass

DATA_FILE = "tracked_bodies.dat"
OUTPUT_FILE = "tracked_bodies.mp4"  # change to .gif if you prefer GIF


def main():
    # Read data (whitespace separated)
    data = pd.read_csv(DATA_FILE, delim_whitespace=True)

    # Sort by iter then body_index to be safe
    data = data.sort_values(["iter", "body_index"]).reset_index(drop=True)

    iters = np.sort(data["iter"].unique())
    n_iters = len(iters)

    # Pre-compute axis limits
    x_min, x_max = data["x"].min(), data["x"].max()
    y_min, y_max = data["y"].min(), data["y"].max()
    span = max(x_max - x_min, y_max - y_min)
    if span == 0:
        span = 1.0
    pad = 0.05 * span

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=5)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Tracked bodies")
    ax.set_xlim(x_min - pad, x_min + span + pad)
    ax.set_ylim(y_min - pad, y_min + span + pad)
    ax.set_aspect("equal", adjustable="box")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        return scat, time_text

    def update(frame_idx):
        it = iters[frame_idx]
        df = data[data["iter"] == it]
        pts = np.column_stack([df["x"].values, df["y"].values])
        scat.set_offsets(pts)
        # All rows in this iter have the same time
        t = float(df["time"].iloc[0]) if not df.empty else 0.0
        time_text.set_text(f"iter = {it}, time = {t:.3f}")
        return scat, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=n_iters,
        init_func=init,
        blit=True,
        interval=50,  # milliseconds between frames
        repeat=False,
    )

    # Save animation. This requires a writer (e.g. ffmpeg) installed in the system.
    try:
        ani.save(OUTPUT_FILE, fps=20)
        print(f"Animation saved to {OUTPUT_FILE}")
    except Exception as e:
        print("Failed to save animation. You may need to install ffmpeg or Pillow.")
        print("Error:", e)


if __name__ == "__main__":
    main()
