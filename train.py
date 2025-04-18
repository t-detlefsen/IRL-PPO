import argparse
from datetime import datetime

from PPO import PPO
from envs.env import make_envs
from utils import *
from tqdm import tqdm
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
        time_step = 0
        i_episode = 0
        print_running_reward = 0
        print_running_episodes = 0

        while time_step <= 3e6:
            obs, _ = self.envs.reset()
            current_ep_reward = 0

            print(f"Episode {i_episode}")
            for t in range(1, 1000+1):

                # select action with policy
                action = self.agent.select_action(obs)
                obs, reward, terminal, _, _ = self.envs.step(action)

                # saving reward and is_terminals
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(terminal)

                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % 4000 == 0:
                    self.agent.update()

                # printing average reward
            if time_step % 10000 == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward.item() / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1

        # # Loop through iterations
        # for i in range(self.params["n_iter"] + 1):
        #     # ----------- TRAINING -----------
        #     # Reset train environment
        #     obs, _ = self.envs.reset()
        #     train_rewards = torch.zeros((self.params["max_ep_len"], self.params["num_envs"]))

        #     # Loop through steps
            
        #     for j in tqdm(range(self.params["max_ep_len"])):
        #         # Generate action by interacting w/ environment
        #         action = self.agent.select_action(obs)
        #         obs, reward, terminal, _, _ = self.envs.step(action)
        #         train_rewards[j] = reward
                
        #         # Update replay buffer
        #         self.agent.buffer.rewards.append(reward)
        #         self.agent.buffer.is_terminals.append(terminal)

        #     # Evaluate performance
        #     print("Train mean reward:", train_rewards.mean().item())

        #     # Update agent if specified
        #     if i % self.params["update_freq"] == 0:
        #         self.agent.update()

        #     # ---------- EVALUATION ----------
        #     # Evaluate if specified
        #     if i % self.params["eval_freq"] == 0:
        #         # Reset eval environment
        #         eval_obs, _ = self.eval_envs.reset()
        #         eval_rewards = torch.zeros((self.params["max_ep_len"], self.params["num_envs"]))

        #         # Loop through steps
        #         for j in tqdm(range(self.params["max_ep_len"])):
        #             # Generate action by interacting w/ environment
        #             action = self.agent.select_action(eval_obs, eval=True)
        #             eval_obs, reward, _, _, _ = self.eval_envs.step(action)
        #             eval_rewards[j] = reward
                
        #         # Evaluate performance
        #         # print(eval_rewards)
        #         print("Eval mean reward:", eval_rewards.mean().item())

        #         # Save model
        #         self.agent.save(f"ckpts/epoch_{i}.pth")

def main():
    # Setup argument parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--update_freq', type=int, default=4)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--lr_actor', type=float, default=3e-3)
    parser.add_argument('--lr_critic', type=float, default=3e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--K_epochs', type=int, default=80)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--action_std_init', type=float, default=0.6)
    parser.add_argument('--device', type=str, default="cpu", choices={"cpu", "cuda"})
    parser.add_argument('--checkpoint', type=str, default=None) # Might need modified if used

    # Environment specific arguments
    # Environment specific arguments
    parser.add_argument('--env_id', type=str, default='PushCube-v1')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--num_eval_envs', type=int, default=1)
    parser.add_argument('--num_eval_steps', type=int, default=50)

    parser.add_argument('--capture_video', action='store_true')
    parser.add_argument('--save_train_video_freq', type=int, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--control_mode', type=str, default='pd_joint_delta_pos')
    parser.add_argument('--reconfiguration_freq', type=int, default=None)
    parser.add_argument('--eval_reconfiguration_freq', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=50)

    parser.add_argument('--partial_reset', action='store_true')
    parser.add_argument('--eval_partial_reset', action='store_true')

    args = parser.parse_args()

    params = vars(args)

    # Set random seeds
    set_random_seeds()

    # Start training loop
    trainer = PPO_trainer(params)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()