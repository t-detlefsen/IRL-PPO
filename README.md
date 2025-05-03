# Proximal Policy Optimization for Door Opening

CMU 16-831 Intro to Robot Learning final project Proximal Policy Optimization for a door opening task using the SAPIEN simulator and ManiSkill library.

## Environment Setup

To set up a Conda environment for running this code with [SAPIEN](https://sapien.ucsd.edu/) and [ManiSkill](https://github.com/haosulab/ManiSkill), follow the steps below:

```bash
# 1. Create and activate Conda environment
conda create -n sapien_maniskill3 python=3.10 -y
conda activate sapien_maniskill3

# 2. Install SAPIEN
pip install sapien==3.0.0b1

# 3. Install ManiSkill
git clone https://github.com/haosulab/ManiSkill.git
cd ManiSkill
git checkout tags/v3.0.0b20
pip install -e .
```

## Running the Custom Code

To run the code, first clone the repository and navigate to the `ppo_custom_code` directory, which contains `train.py`. 

## Example Custom Code Command

```bash
python train.py --num_envs 24 --K_epochs 2

# will turn off video and model saving. Uses 24 training environments and updates the policy for 2 epochs per iteration.

```

## Running the Baseline Code

To run the code, first clone the repository and navigate to the `ppo_baseline_code` directory, which contains `ppo_trainer.py`.

> ⚠️ **Note:** This code assumes you are using a CUDA-compatible GPU.

This project uses [Tyro](https://github.com/brentyi/tyro) for command-line argument parsing based on the `Args` dataclass. You can override any default value directly via the command line without modifying the source code.

For this baseline code, we significantly restructured the original [ManiSkill PPO benchmark code](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo) by splitting it into modular scripts and functions to improve clarity, maintainability, and ease of use. We added comments explaining every part of the code. Additionally, we added the ability to customize the architectures of the actor and critic neural networks.


## Example Baseline Code Command

```bash
python ppo_trainer.py --no-capture_video --no-save_model --num_envs 512 --update_epochs 2

# will turn off video and model saving. Uses 512 training environments and updates the policy for 2 epochs per iteration.
```

## Tuning Parameters 

Here are some useful parameters you might want to adjust:

| Argument                  | Default                  | Description                                                                 |
|---------------------------|--------------------------|-----------------------------------------------------------------------------|
| `--env_id`               | `"OpenCabinetDrawer-v1"` | Name of the ManiSkill task                                                 |
| `--learning_rate`        | `3e-4`                    | Learning rate for the optimizer                                            |
| `--num_envs`             | `512`                     | Number of parallel training environments                                   |
| `--num_eval_envs`        | `8`                       | Number of parallel evaluation environments                                 |
| `--clip_coef`            | `0.2`                     | PPO clipping coefficient                                                   |
| `--ent_coef`             | `0.005`                   | Entropy bonus coefficient                                                  |
| `--vf_coef`              | `0.5`                     | Value function loss coefficient                                            |
| `--gamma`                | `0.8`                     | Discount factor                                                            |
| `--finite_horizon_gae`  | `True`                    | If set, uses finite horizon variant of GAE                                 |
| `--gae_lambda`           | `0.9`                     | Lambda for Generalized Advantage Estimation                                |
| `--update_epochs`        | `4`                       | Number of PPO update epochs per iteration                                  |
| `--num_minibatches`      | `32`                      | Number of minibatches per PPO update                                       |
| `--n_hl_actor`           | `3`                       | Number of hidden layers in actor network                                   |
| `--hl_size_actor`        | `256`                     | Size of each hidden layer in actor network                                 |
| `--activation_actor`     | `"tanh"`                  | Activation function for actor hidden layers                                |
| `--n_hl_critic`          | `3`                       | Number of hidden layers in critic network                                  |
| `--hl_size_critic`       | `256`                     | Size of each hidden layer in critic network                                |
| `--activation_critic`    | `"tanh"`                  | Activation function for critic hidden layers                               |
| `--evaluate`             | `False`                   | Run in evaluation-only mode using a checkpoint                             |
| `--checkpoint`           | `None`                    | Path to a pretrained model checkpoint                                      |
| `--save_model`           | `True`                    | Save model to disk                                                         |
| `--capture_video`        | `True`                    | Save eval videos                                           |

## References

- [ManiSkill: Generalizable Manipulation Skills Benchmark](https://github.com/haosulab/ManiSkill/tree/main). Hao Su Lab. GitHub. Accessed May 3, 2025.




