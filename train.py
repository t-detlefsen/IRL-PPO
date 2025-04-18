import argparse
import datetime import datetime

from PPO import PPO
from envs.env import make_envs
from utils import *

class PPO_trainer(object):
    '''
    Train an RL agent using PPO
    '''
    def __init__(self, params):
        self.params = params

        # Setup environments
        run_name = f"{self.params['env_id']}_{datetime.now()}"
        self.envs, self.eval_envs = make_envs(argparse.Namespace(**self.params), run_name)

        # Observation/action shapes from env
        obs_shape = self.envs.single_observation_space.shape[0]
        act_shape = self.envs.single_action_space.shape[0]

        # Setup PPO agent
        self.agent = PPO(obs_shape,
                         act_shape,
                         params["lr_actor"],
                         params["lr_critic"],
                         params["gamma"],
                         params["K_epochs"],
                         params["eps_clip"],
                         params["action_std_init"],
                         torch.device(params["device"]))
        
        if params["checkpoint"] is not None:
            self.agent.load(params["checkpoint"])

    def run_training_loop(self):
        '''
        Train the RL agent usng PPO
        '''
        # Loop through iterations
        for i in range(self.params["n_iter"] + 1):
            # ----------- TRAINING -----------
            # Reset train environment
            obs, _ = self.envs.reset()
            train_rewards = []

            # Loop through steps
            for j in range(self.params["max_ep_len"]):
                # Generate action
                self.agent.select_action(obs)

                # Interact w/ environment
                action = self.agent.select_action(obs)
                obs, reward, terminal, _, _ = self.envs.step(action)

                # Update replay buffer
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(terminal)

            # Evaluate performance
            print("Train mean reward:", torch.tensor(train_rewards).mean().item())

            # Update agent if specified
            if i % self.params["update_freq"] == 0:
                self.agent.update()

            # ---------- EVALUATION ----------
            # Evaluate if specified
            if i % self.params["eval_freq"] == 0:
                # Reset eval environment
                eval_obs, _ = self.eval_envs.reset()
                eval_rewards = []

                # Loop through steps
                for j in range(self.params["max_ep_len"]):
                    # Generate action
                    self.agent.select_action(eval_obs)

                    # Interact w/ environment
                    action = self.agent.select_action(obs)
                    obs, reward, _, _, _ = self.envs.step(action)
                    eval_rewards.append(reward)
                
                # Evaluate performance
                print("Eval mean reward:", torch.tensor(eval_rewards).mean().item())

                # Save model
                self.agent.save("ckpts/epoch_{}.pth", i)

def main():
    # Setup argument parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--update_freq', type=int, default=4)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--max_', type=float, default=3e-4)
    parser.add_argument('--lr_actor', type=float, default=3e-4)
    parser.add_argument('--lr_critic', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--K_epochs', type=int, default=4)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--action_std_init', type=float, default=0.6)
    parser.add_argument('--device', type=str, default="cpu", choices={"cpu", "cuda"})
    parser.add_argument('--checkpoint', type=str, default=None) # Might need modified if used

    # Environment specific arguments
    parser.add_argument('--env_id', type=str, default='PickCube-v1')
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_eval_envs', type=int, default=2)
    parser.add_argument('--capture_video', action='store_true')
    parser.add_argument('--save_train_video_freq', type=int, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--control_mode', type=str, default='pd_joint_delta_pos')
    args = parser.parse_args()

    params = vars(args)

    # Set random seeds
    set_random_seeds()

    # Start training loop
    trainer = PPO_trainer(params)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()