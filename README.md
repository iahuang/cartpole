# cartpole

A PyTorch implementation of the classic CartPole reinforcement learning task I wrote to learn RL. Implements the REINFORCE and PPO algorithms as well as the environment and simulation code.

```
./
  notebook/         Miscellaneous Markdown notes
  
  cartpole.py       Headless simulation, physics and environment
  gfx.py            Drawing utilities
  models.py         Base policy, value MLP definitions
  play.py           Interactive human-play simulation
  ppo.py            PPO training loop
  ppo.py            PPO training loop
  reinforce.py      Vanilla REINFORCE training loop
  rl.py             Base RL training utilities
  watch.py          Watch a trained model play
```

For experiment tracking, please create a `.env` file in the base directory with the following contents:

```
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_PROJECT=...
```
