#!/usr/bin/env python3
"""
Interpretive still — TRM dual state (y, z) + halting metaphor.
Not a training visualization. Direction A · project-to-art v2.1.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render_dual_orbit(
    output: Path,
    t_cycles: int = 8,
    n_inner: int = 4,
    width: int = 1920,
    height: int = 1080,
    dpi: int = 100,
    seed: int = 2510,
) -> Path:
    rng = np.random.default_rng(seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig_w, fig_h = width / dpi, height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#07090f")
    ax.set_facecolor("#07090f")
    ax.set_aspect("equal")
    ax.axis("off")

    frames = 360
    t = np.linspace(0, 2 * math.pi, frames)
    r_y, r_z = 0.38, 0.22
    omega_y = 1.0 / max(t_cycles, 1)
    omega_z = n_inner * omega_y

    yx = r_y * np.cos(omega_y * t)
    yy = r_y * np.sin(omega_y * t)
    zx = r_z * np.cos(omega_z * t + 0.4)
    zy = r_z * np.sin(omega_z * t + 0.4)

    halt_phase = int(0.82 * frames)
    alpha = np.ones(frames)
    alpha[halt_phase:] = 0.35

    ax.plot(yx[:halt_phase], yy[:halt_phase], color="#4ecdc4", lw=1.2, alpha=0.85, label="y")
    ax.plot(zx[:halt_phase], zy[:halt_phase], color="#e056fd", lw=0.9, alpha=0.85, label="z")
    ax.plot(yx[halt_phase:], yy[halt_phase:], color="#4ecdc4", lw=0.6, alpha=0.25)
    ax.plot(zx[halt_phase:], zy[halt_phase:], color="#e056fd", lw=0.5, alpha=0.25)

    ax.scatter([0], [0], s=120, c="white", alpha=0.9, zorder=5)
    ax.scatter([yx[halt_phase - 1]], [yy[halt_phase - 1]], s=40, c="#4ecdc4", zorder=4)
    ax.scatter([zx[halt_phase - 1]], [zy[halt_phase - 1]], s=30, c="#e056fd", zorder=4)

    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "trm-lab-xs-orbit-candidate-a.png",
    )
    p.add_argument("--t-cycles", type=int, default=8)
    p.add_argument("--n-inner", type=int, default=4)
    p.add_argument("--seed", type=int, default=2510)
    args = p.parse_args()
    path = render_dual_orbit(args.output, t_cycles=args.t_cycles, n_inner=args.n_inner, seed=args.seed)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
