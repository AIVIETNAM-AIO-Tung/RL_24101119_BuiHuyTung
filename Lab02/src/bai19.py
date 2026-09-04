"""Bai 19. Kiem tra tong xac suat transition = 1 cho moi (state, action)."""

import gymnasium as gym
import numpy as np


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    P = env.unwrapped.P
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    all_ok = True
    for s in range(n_states):
        for a in range(n_actions):
            probabilities = [t[0] for t in P[s][a]]
            ok = np.isclose(sum(probabilities), 1.0)
            if not ok:
                all_ok = False
                print(f"LOI tai state={s}, action={a}: tong={sum(probabilities)}")

    print("Tat ca (state, action) co tong xac suat = 1?", all_ok)

    env.close()


if __name__ == "__main__":
    main()
