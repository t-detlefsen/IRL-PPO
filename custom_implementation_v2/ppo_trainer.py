from args import Args
import tyro
import os
import time
from logger import Logger
from agent import Agent
import torch.optim as optim
from utils import *
from mani_skill.utils import gym_utils
class PPO_Trainer():
    def __init__(self,args ):
        args = tyro.cli(Args)
        if args.exp_name is None:
            args.exp_name = os.path.basename(__file__)[: -len(".py")]
            args.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}_nEnvs_{args.num_envs}_nEvalEnvs_{args.num_eval_envs}_iter_{args.num_iterations}_lr_{args.learning_rate}_clipCoef_{args.clip_coef}_vfCoef_{args.vf_coef}_entCoef_{args.ent_coef}"
        else:
            args.run_name = args.exp_name

        #seeding
        set_random_seeds(args.seed)

        #init gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        #get environments
        self.envs,self.eval_envs=getEnvs(args)
        args.max_episode_steps = gym_utils.find_max_episode_steps_value(self.envs._env)

        # will run 1 full episode in training and eval
        args.num_steps=args.max_episode_steps
        args.num_eval_steps=args.max_episode_steps

        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.total_timesteps= int(args.num_iterations * args.batch_size)
        self.args=args
        
        #setup logger
        self.logger=Logger(args)

        #setup agent
        self.agent = Agent(args,self.envs).to(self.device)
        
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)
        self.action_space_low, self.action_space_high = torch.from_numpy(envs.single_action_space.low).to(self.device), torch.from_numpy(envs.single_action_space.high).to(self.device)
        if args.checkpoint:
            self.agent.load_state_dict(torch.load(args.checkpoint))
        self.global_step = 0
    

    
    


    def run_training_loop(self):


        # TRY NOT TO MODIFY: start the game
       
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        eval_obs, _ = self.eval_envs.reset(seed=self.args.seed)
        next_done = torch.zeros(num_envs, device=self.device)
        print(f"####")
        print(f"args.num_iterations={self.args.num_iterations} args.num_envs={num_envs} args.num_eval_envs={self.args.num_eval_envs}")
        print(f"args.minibatch_size={self.args.minibatch_size} args.batch_size={self.args.batch_size} args.update_epochs={self.args.update_epochs}")
        print(f"####")
        obs, actions, logprobs, rewards,dones, values, final_values, next_done = collect_training_data(self.agent, self.envs, self.args.num_steps, self.args.num_envs, self.device)


        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        ####################### Training #####################################
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
             
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()


