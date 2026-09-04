"""Bai 17. In toan bo transition model cua mot state."""

import gymnasium as gym

from mdp_utils import ACTION_NAMES

STATE = 0


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    P = env.unwrapped.P

    for action in range(env.action_space.n):
        name = ACTION_NAMES[action]
        print(f"Action {action} ({name}):")
        for probability, next_state, reward, terminated in P[STATE][action]:
            print(
                f"  Probability={probability:.3f}  Next state={next_state:>2d}"
                f"  Reward={reward}  Terminated={terminated}"
            )

    env.close()


if __name__ == "__main__":
    main()
