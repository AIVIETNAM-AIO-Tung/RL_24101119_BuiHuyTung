"""Bai 11. So sanh reward som va reward tre, tim khoang gamma ma B > A."""

import numpy as np

from mdp_utils import compute_return

sequence_A = [5, 0, 0, 0, 0]
sequence_B = [0, 0, 0, 0, 10]


def main():
    gammas = np.linspace(0, 1, 1001)
    crossover = None

    for g in gammas:
        G_A = compute_return(sequence_A, g)
        G_B = compute_return(sequence_B, g)
        if G_B > G_A and crossover is None:
            crossover = g

    print(f"B > A khi gamma >= xap xi {crossover:.3f}")

    # kiem tra bang cong thuc dai so: 5 = 10 * gamma^4 => gamma = (0.5)^(1/4)
    gamma_exact = 0.5 ** (1 / 4)
    print(f"Gia tri chinh xac tu dai so: gamma = 0.5^(1/4) = {gamma_exact:.6f}")


if __name__ == "__main__":
    main()
