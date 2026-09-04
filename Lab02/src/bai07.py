"""Bai 7. Undiscounted return."""

from mdp_utils import compute_return

rewards = [1, 1, 1, 1, 1]


def main():
    G = compute_return(rewards, gamma=1.0)
    print(f"rewards = {rewards}")
    print(f"G (gamma=1.0) = {G}")


if __name__ == "__main__":
    main()
