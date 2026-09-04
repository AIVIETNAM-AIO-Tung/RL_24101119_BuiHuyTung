"""Bai 4. Phan phoi trang thai sau nhieu buoc."""

import numpy as np

from bai01 import P, STATES
from mdp_utils import state_distribution

p0 = np.array([1.0, 0.0, 0.0])


def main():
    for t in [1, 2, 5, 10, 50]:
        p_t = state_distribution(p0, P, t)
        formatted = ", ".join(f"{n}={v:.4f}" for n, v in zip(STATES, p_t))
        print(f"t={t:>2d}: {formatted}")


if __name__ == "__main__":
    main()
