"""Bai 5. Mo phong Markov chain bang sampling."""

import numpy as np

from bai01 import P, STATES
from mdp_utils import sample_next_state


def main():
    rng = np.random.default_rng(seed=42)

    current_state = 0  # bat dau tu Sunny
    trajectory = [current_state]

    for _ in range(30):
        current_state = sample_next_state(current_state, P, rng)
        trajectory.append(current_state)

    names = [STATES[s] for s in trajectory]
    print(" -> ".join(names))


if __name__ == "__main__":
    main()
