"""Bai 6. So sanh tan suat mo phong voi phan phoi ly thuyet."""

import os

import matplotlib.pyplot as plt
import numpy as np

from bai01 import P, STATES
from mdp_utils import sample_next_state, state_distribution

N_STEPS = 50
N_TRANSITIONS = 100_000

FIGURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "figures", "markov_distribution.png"
)


def main():
    rng = np.random.default_rng(seed=0)

    # --- Mo phong ---
    current_state = 0
    counts = np.zeros(len(STATES))

    for _ in range(N_TRANSITIONS):
        current_state = sample_next_state(current_state, P, rng)
        counts[current_state] += 1

    empirical = counts / N_TRANSITIONS

    # --- Ly thuyet: phan phoi sau N_STEPS buoc tu p0 = [1,0,0] ---
    p0 = np.array([1.0, 0.0, 0.0])
    theoretical = state_distribution(p0, P, N_STEPS)

    print(f"Sau {N_TRANSITIONS} transitions (mo phong):")
    for name, v in zip(STATES, empirical):
        print(f"  {name}: {v:.4f}")

    print(f"\nPhan phoi ly thuyet sau {N_STEPS} buoc (p0 @ P^{N_STEPS}):")
    for name, v in zip(STATES, theoretical):
        print(f"  {name}: {v:.4f}")

    print("\nNhan xet:")
    print("- Tan suat mo phong va phan phoi ly thuyet rat gan nhau (sai so nho).")
    print("- Dieu nay phu hop voi ly thuyet: khi so buoc/so lan mo phong du lon,")
    print("  Markov chain hoi tu ve stationary distribution (neu chain la")
    print("  irreducible va aperiodic), va tan suat thuc nghiem se xap xi")
    print("  phan phoi ly thuyet do Luat so lon (Law of Large Numbers).")

    # --- Ve bieu do so sanh ---
    x = np.arange(len(STATES))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, empirical, width, label="Mo phong (thuc nghiem)")
    plt.bar(x + width / 2, theoretical, width, label="Ly thuyet (p0 @ P^n)")
    plt.xticks(x, STATES)
    plt.title("So sanh phan phoi Markov chain: mo phong vs ly thuyet")
    plt.xlabel("Trang thai")
    plt.ylabel("Xac suat")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    print(f"\nDa luu bieu do tai: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
