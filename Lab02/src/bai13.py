"""Bai 13. Kiem tra tinh hop le cua model MDP tu Bai 12."""

from bai12 import N_ACTIONS, N_STATES, P
from mdp_utils import validate_mdp


def main():
    is_valid = validate_mdp(P, N_STATES, N_ACTIONS)
    print("MDP hop le?", is_valid)


if __name__ == "__main__":
    main()
