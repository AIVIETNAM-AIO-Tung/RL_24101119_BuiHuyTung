"""Bai 18. Kiem thu ham describe_state() voi cac state 0, 1, 14."""

import gymnasium as gym

from mdp_utils import describe_state


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    for s in [0, 1, 14]:
        describe_state(env, s)
        print()

    env.close()


if __name__ == "__main__":
    main()
