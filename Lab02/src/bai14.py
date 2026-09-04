"""Bai 14. Deterministic policy cho MDP 2 state."""

import numpy as np

from mdp_utils import print_policy

# State 0 -> chon action 0 (tiep tuc hoc)
# State 1 -> chon action 1 (chuyen sang hoc)
policy = np.array([0, 1])

CUSTOM_ACTION_NAMES = {0: "Tiep tuc", 1: "Chuyen hoat dong"}


def main():
    print_policy(policy, action_names=CUSTOM_ACTION_NAMES)


if __name__ == "__main__":
    main()
