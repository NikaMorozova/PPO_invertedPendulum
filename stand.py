from collections import defaultdict
import numpy as np
import sys
import torch

def act(env, agent, nsteps = 2048, device = "cuda"):
    state = {"observation": env.reset_model()}
    trajectory = defaultdict(list, {"actions": []})
    observations = []
    rewards = []
    resets = []

    for i in range(nsteps):
        action, log_prob = agent.actor.get_actions(state["observation"])
        observations.append(state["observation"])
        trajectory["actions"].append(action)
        trajectory["log_probs"].append(log_prob)

        obs, reward, terminated, truncated = env.step(action)
        state["observation"] = obs
        rewards.append(reward)
        reset = terminated or truncated
        resets.append(float(reset))
        if np.all(reset):
            state["observation"] = env.reset_model()

    trajectory.update(observations=observations, rewards=rewards, resets=resets)
    trajectory["state"] = state

    return trajectory
