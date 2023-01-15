# Proximal Policy Optimization for Inverted pendulum

This repository contains implementation of PPO algorithm for Inverted pendulum env to solve upswing, balancing and target-reaching tasks.

## Summary


```bash
- main.py      # Main file to run train or test phase
- train.py     # File containing train loop for PPO
- test.py      # File to get model tested and saved simulation in gif
- envs.py      # File containing custom MuJoCo env
- ppo.py       # PPO algorithm, Actor and Critic networks
- stand.py     # File containing function for trajectory collecting
- Report.ipynb # Technical report of completing this task
- gifs         # Gifs after running tests.py
- weight       # Weights of the model
```

## How to use it

Install requirements
```bash
pip install -r requirements.txt
```

```bash
python3 main.py --run=train --upswing=True --target=True 
```

```bash
Options:
    --run                     str       Choose Train or Test mode
    --upswing                 bool      Choose whether you want to solve upswing task
    --target                  bool      Choose whether you want to solve target task
    --mass                    float     Pendulum mass value
    --num_epochs              int       default=500
    --horizon                 int       default=2048
    --actor_weights           str       default=None
    --critic_weights          str       default=None

```
