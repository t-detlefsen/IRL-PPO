from args import Args
import tyro
import os
import time
from logger import Logger
from agent import Agent
import torch.optim as optim
from utils import *
from mani_skill.utils import gym_utils
from collections import defaultdict
from buffer import Rollout_Data

class PPO_Trainer():
    def __init__(self,args):
        
        if args.exp_name is None:
            args.exp_name = os.path.basename(__file__)[: -len(".py")]
            args.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}_nEnvs_{args.num_envs}_nEvalEnvs_{args.num_eval_envs}_iter_{args.num_iterations}_lr_{args.learning_rate}_clipCoef_{args.clip_coef}_vfCoef_{args.vf_coef}_entCoef_{args.ent_coef}"
        else:
            args.run_name = args.exp_name

        #seeding
        set_random_seeds(args.seed)

        #init gpu
        self.device = torch.device("cuda" )

        #get environments
        self.envs,self.eval_envs=getEnvs(args)
        args.max_episode_steps = gym_utils.find_max_episode_steps_value(self.envs._env)

        # will run 1 full episode in training and eval
        args.num_steps=args.max_episode_steps
        args.num_eval_steps=args.max_episode_steps

        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.total_timesteps= int(args.num_iterations * args.batch_size)
       
        
        #setup logger
        self.logger=Logger(args)

        #setup agent
        self.obs_shape=self.envs.single_observation_space.shape
        self.act_shape=self.envs.single_action_space.shape
        print("obs_shape= ", self.obs_shape )
        print("act_shape= ", self.act_shape)
      
        self.obs_dim=np.array(self.obs_shape).prod()
        self.act_dim=np.array(self.act_shape).prod()
        print("obs_dim= ", self.obs_dim )
        print("act_dim= ", self.act_dim)
        self.agent = Agent(self.obs_dim, self.act_dim).to(self.device)
        if args.checkpoint: #if you want to load the model as a specific checkpoint
            self.agent.load_state_dict(torch.load(args.checkpoint))
        
        # save necessary values and variables
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)
        self.action_space_low = torch.from_numpy(self.envs.single_action_space.low).to(self.device)
        self.action_space_high = torch.from_numpy(self.envs.single_action_space.high).to(self.device)
        self.num_steps = args.num_steps
        self.num_envs = args.num_envs
        self.num_eval_envs = args.num_eval_envs
        self.seed=args.seed
        self.global_step = 0
        self.run_name = args.run_name
        self.eval_freq=args.eval_freq

        # for calculating advantages
        self.gamma=args.gamma
        self.gae=args.finite_horizon_gae
        self.gae_lambda= args.gae_lambda
        self.norm_adv= args.norm_adv

        # for training
        self.K_epochs = args.update_epochs
        self.num_iterations = args.num_iterations
        self.minibatch_size = args.minibatch_size
        self.max_grad_norm = args.max_grad_norm 

        # for the loss:
        self.clip_eps= args.clip_coef
        self.vf_coef=args.vf_coef
        self.ent_coef= args.ent_coef


    def clip_action(self,action):
        return torch.clamp(action.detach(), self.action_space_low, self.action_space_high)
    
    def rollout_and_collect_data(self):
        cur_obs, _ = self.envs.reset(seed=self.seed) # will store current state that policy will use for each step 
        cur_done = torch.zeros(self.num_envs, device=self.device) # all envs start as not done

        actions = torch.zeros((self.num_steps, self.num_envs) + self.act_shape).to(device)# row: timestep, column: environment, entry: action
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)# row: timestep, column: environment, entry: logprob
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)# row: timestep, column: environment, entry: reward

        values = torch.zeros((self.num_steps+1, self.num_envs)).to(device) # row: timestep, column: environment, entry: value
        observations = torch.zeros((self.num_steps+1, self.num_envs) + self.obs_shape).to(self.device) # row: timestep, column: environment, entry: observation
        dones = torch.zeros((self.num_steps+1, self.num_envs)).to(device)# row: timestep, column: environment, entry: done 
        # add 1 to save the final done and obs 
        

        vals_at_eps_end = torch.zeros((self.num_steps+1, self.num_envs), device=device) # stores the value of the very last state of each episode in each env
        # handles cases where an environment terminates early while the other environments keep running
        vals_at_eps_end[0]= torch.zeros(self.num_envs, device=self.device) #initial value is zero because this is the start of the rollout
        
        for step in range(0, self.num_steps):
            self.global_step += self.num_envs # keeps track of total # of steps taken in all envs
            observations[step] = cur_obs
            dones[step] = cur_done

            # get an action from the current policy using the current observation
            with torch.no_grad():
                action_from_policy, logprob, _, value = self.agent.get_action_and_value(cur_obs)
                values[step] = value.flatten()
            actions[step] = action_from_policy
            logprobs[step] = logprob

            # Clip action to fit the constraints of the environment
            action_clipped=self.clip_action(action_from_policy)
            cur_obs, reward, terminations, truncations, infos = self.envs.step(action_clipped)
            # terminations: 
            #   shape: (num_envs, )
            #   if True: env terminated because episode ended

            # truncations:
            #   shape: (num_envs)
            #   if True: env episode ended because it hit the time limit 

            cur_done = torch.logical_or(terminations, truncations).to(torch.float32) # true if the env episode ended or timed out
            rewards[step] = reward.view(-1) 

            if "final_info" in infos: # checks if any env finished an episode
                final_info = infos["final_info"] # dictionary containing episode info for terminated envs ONLY (e.g., return, success_once, etc.)
                done_mask = infos["_final_info"] # says if the episode terminated, shape: (num_envs,)
                final_observations = infos["final_observation"][done_mask] # saves the final observations for the terminated envs ONLY
                term_env_idxs=torch.arange(self.num_envs, device=device)[done_mask] # saves the indices the terminated envs ONLY

                for key, value in final_info["episode"].items(): # logs training data
                    self.logger.add_scalar(f"train/{key}", value[done_mask].float().mean(), self.global_step)
                with torch.no_grad():
                    vals_at_eps_end[step+1,term_env_idxs ] = self.agent.get_value(final_observations).view(-1)

        # save the info for the last state in the rollout
        observations[self.num_steps]= cur_obs
        dones[self.num_steps]= cur_done
        with torch.no_grad():
            values[self.num_steps]= self.agent.get_value(cur_obs).reshape(1, -1)
        return Rollout_Data(observations, actions, logprobs, rewards,dones, values, vals_at_eps_end)

    def compute_advantages(self,rollout_data):
            
            with torch.no_grad():
                advantages = torch.zeros_like(rollout_data.rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    next_not_done = 1.0 - rollout_data.dones[t + 1]
                    nextvalues = rollout_data.values[t + 1]
                    real_next_values = next_not_done * nextvalues + rollout_data.vals_at_eps_end[t+1] 
                    if self.gae:
                 
                        if t == self.num_steps - 1: 
                            lam_coef_sum = 0.
                            reward_term_sum = 0. 
                            value_term_sum = 0. 
                        lam_coef_sum = lam_coef_sum * next_not_done
                        reward_term_sum = reward_term_sum * next_not_done
                        value_term_sum = value_term_sum * next_not_done

                        lam_coef_sum = 1 + self.gae_lambda * lam_coef_sum
                        reward_term_sum = self.gae_lambda * self.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                        value_term_sum = self.gae_lambda * self.gamma * value_term_sum + self.gamma * real_next_values

                        advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                    else:
                        delta = rewards[t] + self.gamma * real_next_values - values[t]
                        advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * next_not_done * lastgaelam
                returns = advantages + values
            return advantages, returns
    
    def evaluate(self):
        self.agent.eval()
        with torch.no_grad():

            print("Evaluating")
            eval_obs, _ = self.eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(self.num_eval_steps):
                with torch.no_grad():
                    act=self.get_action(eval_obs, deterministic=True)#why use deterministic?
                    eval_obs, _, _, _, eval_infos = self.eval_envs.step(act)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {self.num_eval_steps * self.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if self.logger is not None:
                    self.logger.add_scalar(f"eval/{k}", mean, self.global_step)
                print(f"eval_{k}_mean={mean}")

    def run_training_loop(self):


        # TRY NOT TO MODIFY: start the game
       
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_done = torch.zeros(self.num_envs, device=self.device)
        print(f"####")
        print(f"args.num_iterations={self.num_iterations} args.num_envs={self.num_envs} args.num_eval_envs={self.num_eval_envs}")
        print(f"args.minibatch_size={self.minibatch_size} args.batch_size={int(self.num_envs * self.num_steps)} args.update_epochs={self.K_epochs}")
        print(f"####")
        for iteration in range(1, self.args.num_iterations + 1):
            vals_at_eps_end = torch.zeros((args.num_steps, args.num_envs), device=device)
            if iteration % args.eval_freq == 1:
                self.logger=self.agent.evaluate(self.eval_envs,self.logger,self.global_step)
                if args.evaluate:
                    break
                if args.save_model:
                    save_model(self.agent,self.args.run_name,iteration)
            rollout_data= self.rollout_and_collect_data()
            
            rollout_data=self.compute_advantages(rollout_data)


            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            ####################### Training #####################################
            self.agent.train()
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            update_time = time.time()
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()


if __name__ == "__main__":
    ############ Process command line inputs ####################
    args = tyro.cli(Args)
    ppo_trainer=PPO_Trainer(args)
    ppo_trainer.run_training_loop()