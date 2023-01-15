import torch
import numpy as np
import tqdm
from ppo import PPO
from stand import act
from envs import InvertedPendulumEnv

def train(
      upswing: bool = False,
      target: bool = False,
      num_epochs: int = 500,
      horizon: int = 4096):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = InvertedPendulumEnv(  #initialize Environment
        target=target,
        upswing=upswing,
        test=False
    )

    if target:
        obs_shape = 5    # Here we define the size of observation space
                         # (If we solve target task, we need to extend initial observation space, so the size is 5)
    else:
        obs_shape = 4
    action_space = env.action_space.shape[0]

    agent = PPO(input_dim=obs_shape, output_dim=action_space) # initialize Actor-Critic Neural Network
                                                              # and PPO algorithm implementation
    
    current_steps = 0
    best_reward = -np.inf
    for epoch in tqdm.trange(num_epochs, desc="PPO is training"):
        # Firstly collect trjectories
        trajectory = act(env, agent, horizon, device)

        # Here we update networks of agent by learning for several epochs
        model_update = agent.step(trajectory)

        if np.mean(trajectory["rewards"]) > best_reward:
            best_reward = np.mean(trajectory["rewards"])

            print(f"\n Iter: {epoch} Critic loss: {model_update['actor_loss']} | Actor loss: {model_update['critic_loss']} | \
                  new best reward: {best_reward}"
            )
            torch.save(agent.actor.state_dict(), 'actor.pth')
            torch.save(agent.critic.state_dict(), 'critic.pth')
    env.close()
    del agent
