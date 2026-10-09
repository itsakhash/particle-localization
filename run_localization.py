"""
Watch a particle filter find a lost robot, or measure how reliably it does.

    python run_localization.py                      # animated demo (close the window to quit)
    python run_localization.py --seed 7             # a different start position
    python run_localization.py --kidnap-at 100      # teleport the robot at step 100, watch recovery
    python run_localization.py --save-gif demo.gif  # save the animation instead of showing it
    python run_localization.py --save-png final.png # save one picture of the end state
    python run_localization.py --trials 30          # no window: run 30 seeded trials, print the stats

In the picture: blue dots are particles (guesses), the green arrow is where the robot
really is, the red arrow is the filter's estimate. At the start the dots cover the whole
map; a few steps later they collapse onto the green arrow.
"""
import argparse
import math

import numpy as np

import localization as loc


def _arrow(ax, pose, color, label):
    return ax.arrow(pose[0], pose[1], 0.5 * math.cos(pose[2]), 0.5 * math.sin(pose[2]),
                    head_width=0.25, color=color, label=label, zorder=5, length_includes_head=True)


def draw_state(ax, grid, particles, truth, estimate, title=""):
    """Draw the map, the particle cloud, the true pose and the estimate."""
    ax.clear()
    ax.imshow(grid.occ, origin="lower", cmap="Greys", extent=[0, grid.width, 0, grid.height])
    ax.scatter(particles[:, 0], particles[:, 1], s=3, c="tab:blue", alpha=0.5, zorder=3)
    _arrow(ax, truth, "tab:green", "true pose")
    _arrow(ax, estimate, "tab:red", "estimate")
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_aspect("equal")
    ax.set_title(title)


def draw_errors(ax, result, upto):
    ax.clear()
    t = np.arange(len(result["errors"])) * result["dt"]
    ax.plot(t[:upto + 1], result["errors"][:upto + 1], color="tab:red")
    ax.axhline(0.3, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, max(1.0, max(result["errors"]) * 1.05))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (m)")
    if result["kidnap_at"] is not None:
        ax.axvline(result["kidnap_at"] * result["dt"], color="orange", linestyle=":", label="kidnapped")
        ax.legend()
    conv = result["converged_step"]
    ax.set_title("converged at %.1f s" % (conv * result["dt"]) if conv is not None else "not converged yet")


def make_figure(grid, result):
    import matplotlib.pyplot as plt
    fig, (ax_map, ax_err) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.5, 1]})

    def show(i):
        p, truth, est = result["frames"][i]
        draw_state(ax_map, grid, p, truth, est, "step %d  (error %.2f m)" % (i, result["errors"][i]))
        draw_errors(ax_err, result, i)
    return fig, show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--particles", type=int, default=loc.DEFAULT_PARTICLES)
    ap.add_argument("--kidnap-at", type=int, default=None, help="step at which the robot is teleported")
    ap.add_argument("--trials", type=int, default=0, help="run this many seeded trials without a window")
    ap.add_argument("--save-gif", default=None)
    ap.add_argument("--save-png", default=None)
    args = ap.parse_args()

    grid = loc.make_demo_map()
    if args.trials:
        ev = loc.evaluate(args.trials, args.seed, steps=args.steps, n_particles=args.particles,
                          kidnap_at=args.kidnap_at)
        print(f"{ev['trials']} trials, {args.particles} particles, {args.steps} steps each")
        print(f"success rate:               {ev['success_rate'] * 100:.0f}%  (within 0.3 m, held for 10 steps)")
        if ev["median_steps_to_converge"] is not None:
            print(f"median time to converge:    {ev['median_steps_to_converge']:.0f} steps "
                  f"({ev['median_seconds_to_converge']:.1f} s)")
        print(f"median final error:         {ev['median_final_error_m'] * 100:.1f} cm")
        return

    result = loc.run_trial(args.seed, steps=args.steps, n_particles=args.particles,
                           kidnap_at=args.kidnap_at, record=True)
    headless = bool(args.save_gif or args.save_png)
    import matplotlib
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, show = make_figure(grid, result)
    last = len(result["frames"]) - 1
    if args.save_png:
        show(last)
        fig.savefig(args.save_png, dpi=120, bbox_inches="tight")
        print("saved", args.save_png)
    if args.save_gif:
        anim = FuncAnimation(fig, show, frames=range(0, last + 1, 2), interval=80)
        anim.save(args.save_gif, writer="pillow", fps=12)
        print("saved", args.save_gif)
    if not headless:
        anim = FuncAnimation(fig, show, frames=last + 1, interval=80, repeat=False)
        plt.show()


if __name__ == "__main__":
    main()
