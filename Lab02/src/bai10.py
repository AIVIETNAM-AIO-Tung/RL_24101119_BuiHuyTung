"""Bai 10. Anh huong cua gamma len G_0, ve va luu bieu do."""

import os

import matplotlib.pyplot as plt
import numpy as np

from mdp_utils import compute_return

rewards = [0, 0, 0, 0, 10]
gammas = np.linspace(0, 1, 101)

FIGURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "figures", "gamma_comparison.png"
)


def main():
    G0_values = [compute_return(rewards, g) for g in gammas]

    plt.figure(figsize=(7, 5))
    plt.plot(gammas, G0_values, color="tab:blue")
    plt.title("Anh huong cua Discount Factor (gamma) len G_0")
    plt.xlabel("gamma")
    plt.ylabel("G_0")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    print(f"Da luu bieu do tai: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
