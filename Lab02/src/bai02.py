"""Bai 2. Kiem tra tinh hop le cua transition matrix."""

import numpy as np

from bai01 import P
from mdp_utils import validate_transition_matrix


def main():
    print("P hop le?", validate_transition_matrix(P))

    P_bad = np.array([
        [0.7, 0.2, 0.2],  # tong hang = 1.1 -> khong hop le
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
    ])
    print("P_bad hop le?", validate_transition_matrix(P_bad))


if __name__ == "__main__":
    main()
