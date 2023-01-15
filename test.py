import time
import torch
import numpy as np
import imageio
from ppo import PPO
from envs import InvertedPendulumEnv

def test(
      actor_weights,
      critic_weights,
      upswing: bool = False,
      target: bool = False,
      mass: float = None):
    
    env = InvertedPendulumEnv(  #initialize Environment
        target=target,
        upswing=upswing,
        mass=None,
        test=True
    )

    if target:
        obs_shape = 5    # Here we define the size of observation space
                         # (If we solve target task, we need to extend initial observation space, so the size is 5)
    else:
        obs_shape = 4
    action_space = env.action_space.shape[0]

    agent = PPO(input_dim=obs_shape, output_dim=action_space) # initialize Actor-Critic Neural Network

    agent.actor.load_state_dict(torch.load(actor_weights, map_location=torch.device('cpu')))
    agent.critic.load_state_dict(torch.load(critic_weights, map_location=torch.device('cpu')))
    agent.actor.train(False)
    agent.critic.train(False)

    observation = env.reset_model()
    while env.current_time < 10:
        print(env.current_time)
        observation = torch.tensor(observation, dtype=torch.float32)
        actions, _ = agent.actor.get_actions(observation)
        observation, _, _, _ = env.step(actions)
    gif_path = "./gifs/env_simulation.gif"
    imageio.mimsave(gif_path, env.frames, fps=30)

