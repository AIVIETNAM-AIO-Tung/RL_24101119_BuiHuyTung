"""Bai 15. Stochastic policy - uniform random."""

import numpy as np

from bai12 import N_ACTIONS, N_STATES

policy = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS


def main():
    print("Stochastic policy (uniform):")
    print(policy)

    row_sums = policy.sum(axis=1)
    print("\nTong xac suat action tai moi state:", row_sums)
    print("Hop le (tong = 1 tai moi state)?", np.allclose(row_sums, 1.0))


if __name__ == "__main__":
    main()
