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
            args.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}_nEnvs_{args.num_envs}_nEvalEnvs_{args.num_eval_envs}_FHgae_{args.finite_horizon_gae}_lr_{args.learning_rate}_clipCoef_{args.clip_coef}_vfCoef_{args.vf_coef}_entCoef_{args.ent_coef}"
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
        if args.num_steps is None:
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
        # print("obs_dim= ", self.obs_dim )
        # print("act_dim= ", self.act_dim)
        self.agent = Agent(args,self.obs_dim, self.act_dim).to(self.device)
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
        self.num_eval_steps=args.num_eval_steps

        # for calculating advantages
        self.gamma=args.gamma
        if (args.finite_horizon_gae):
            self.gae=False
            self.fh_gae=True
        else:
            self.gae=True
            self.fh_gae=False
        self.gae_lambda= args.gae_lambda
        self.norm_adv= args.norm_adv

        # for training
        self.K_epochs = args.update_epochs
        self.num_iterations = args.num_iterations
        self.minibatch_size = args.minibatch_size
        self.batch_size= args.batch_size
        self.max_grad_norm = args.max_grad_norm 
        self.total_epochs=0
        self.total_minibatches=0

        # for the loss:
        self.clip_eps= args.clip_coef
        self.vf_coef=args.vf_coef
        self.ent_coef= args.ent_coef
        print(f"Environment: {args.env_id}")
        


    def clip_action(self,action):
        return torch.clamp(action.detach(), self.action_space_low, self.action_space_high)
    
    def rollout_and_collect_data(self,iter):
        cur_obs, _ = self.envs.reset(seed=self.seed) # will store current state that policy will use for each step 
        cur_done = torch.zeros(self.num_envs, device=self.device) # all envs start as not done

        actions = torch.zeros((self.num_steps, self.num_envs) + self.act_shape).to(self.device)# row: timestep, column: environment, entry: action
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)# row: timestep, column: environment, entry: logprob
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)# row: timestep, column: environment, entry: reward

        values = torch.zeros((self.num_steps+1, self.num_envs)).to(self.device) # row: timestep, column: environment, entry: value
        observations = torch.zeros((self.num_steps+1, self.num_envs) + self.obs_shape).to(self.device) # row: timestep, column: environment, entry: observation
        dones = torch.zeros((self.num_steps+1, self.num_envs)).to(self.device)# row: timestep, column: environment, entry: done 
        # add 1 to save the final done and obs 
        

        vals_at_eps_end = torch.zeros((self.num_steps+1, self.num_envs), device=self.device) # stores the value of the very last state of each episode in each env
        # handles cases where an environment terminates early while the other environments keep running
        vals_at_eps_end[0]= torch.zeros(self.num_envs, device=self.device) #initial value is zero because this is the start of the rollout

        experiences=0
        print("Rolling out policy in training environments and collecting data")
        for step in range(0, self.num_steps):
            # print('At step:    ', step, '/', self.num_steps, end='\r')
            
            self.global_step += self.num_envs # keeps track of total # of steps taken in all envs
            experiences+= self.num_envs
            print(f"Collected {experiences} / {self.batch_size} experiences", end='\r')
            observations[step] = cur_obs
            dones[step] = cur_done

            ################ QUERY CURRENT POLICY #########################################
            # get an action from the current policy using the current observation
            with torch.no_grad():
                action_from_policy, logprob = self.agent.get_action_and_logprob(cur_obs)
                value= self.agent.get_value(cur_obs)
                values[step] = value.flatten()
            actions[step] = action_from_policy
            logprobs[step] = logprob

            ################ TAKE ACTION IN THE ENVIRONMENT #########################################
            # Clip action to fit the constraints of the environment
            action_clipped=self.clip_action(action_from_policy)
            next_obs, reward, next_termination, next_truncation, next_info = self.envs.step(action_clipped)
            rewards[step] = reward.view(-1) 
            # termination: 
            #   shape: (num_envs, )
            #   if True: env terminated because episode ended

            # truncation:
            #   shape: (num_envs)
            #   if True: env episode ended because it hit the time limit 

            next_done = torch.logical_or(next_termination, next_truncation).to(torch.float32) # true if the env episode ended or timed out
        
            
            ############ SAVE VALUES FOR ENV EPISODES THAT ENDED #####################
            if "final_info" in next_info: # checks if any env finished an episode
                final_info = next_info["final_info"] # dictionary containing episode info for terminated envs ONLY (e.g., return, success_once, etc.)
                done_mask = next_info["_final_info"] # says if the episode terminated, shape: (num_envs,)
                final_observation = next_info["final_observation"][done_mask] # saves the final observations for the terminated envs ONLY
                term_env_idxs=torch.arange(self.num_envs, device=self.device)[done_mask] # saves the indices the terminated envs ONLY
                for key, value in final_info["episode"].items(): # logs training data
                    self.logger.add_scalar(f"train/{key}", value[done_mask].float().mean(), iter)
                with torch.no_grad():
                    vals_at_eps_end[step+1,term_env_idxs ] = self.agent.get_value(final_observation).view(-1)

            ######### UPDATE DONE AND OBS FOR NEXT STEP ########################
            cur_done=next_done
            cur_obs=next_obs

        # save the info for the last state/observation in the rollout
        observations[self.num_steps]= cur_obs
        dones[self.num_steps]= cur_done
        with torch.no_grad():
            values[self.num_steps]= self.agent.get_value(cur_obs).reshape(1, -1)
        print()
        return Rollout_Data(observations, actions, logprobs, rewards,dones, values, vals_at_eps_end)


    
    def evaluate(self, iter):
        self.agent.eval()
        with torch.no_grad():

            print("Evaluating")
            eval_currobs, _ = self.eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(self.num_eval_steps):
                with torch.no_grad():
                    act=self.agent.get_action_for_inference(eval_currobs)
                    eval_nxtobs, _, _, _, eval_infos = self.eval_envs.step(act)
                    if "final_info" in eval_infos:
                        done_mask = eval_infos["_final_info"] # says if the episode terminated, shape: (num_envs,)
                        num_episodes += done_mask.sum() # adds the number of terminated episodes
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v) #extract the data from final infos
                    eval_currobs=eval_nxtobs
            print(f"Evaluated {self.num_eval_steps * self.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if self.logger is not None:
                    self.logger.add_scalar(f"eval/{k}", mean, iter)
                print(f"eval_{k}_mean={mean}")

    def run_training_loop(self):

       
        print(f"####")
        print(f"args.num_iterations={self.num_iterations} args.num_envs={self.num_envs} args.num_eval_envs={self.num_eval_envs}")
        print(f"args.minibatch_size={self.minibatch_size} args.batch_size={int(self.num_envs * self.num_steps)} args.update_epochs={self.K_epochs}")
        print(f"####")
        for iteration in range(1, self.num_iterations + 1):
            if iteration % args.eval_freq == 1:
                self.evaluate(iteration)
                if args.evaluate:
                    break
                if args.save_model:
                    save_model(self.agent,self.run_name,iteration)

            print(f"\n\nTraining iteration {iteration} / {self.num_iterations}")
            rollout_data= self.rollout_and_collect_data(iteration)
            
            rollout_data.compute_advantages_and_returns(num_steps=self.num_steps,
                                                        gamma= self.gamma,
                                                        use_gae=self.gae,
                                                        use_fh_gae=self.fh_gae,
                                                        gae_lambda=self.gae_lambda,
                                                        device=self.device)

           
            rollout_data.flatten(obs_shape=self.obs_shape,
                                 act_shape=self.act_shape)
            # FOR LOGGING
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_entropy_bonuses = []
            epoch_total_losses = []

            # Optimizing the policy and value network
            ####################### Training #####################################
            self.agent.train()
            batch_idxs = np.arange(self.batch_size)
            print("Updating Policy")
            for epoch in range(self.K_epochs):
                print(f"Update epoch = {epoch+1} / {self.K_epochs}", end='\r')
                np.random.shuffle(batch_idxs)

                # FOR LOGGING
                mb_policy_losses = []
                mb_value_losses = []
                mb_entropy_bonuses = []
                mb_total_losses = []

                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    #get minibatch data
                    mb_idxs = batch_idxs[start:end]
                    mb_observations=rollout_data.observations[mb_idxs]
                    mb_actions= rollout_data.actions[mb_idxs]
                    mb_oldlogprobs=rollout_data.logprobs[mb_idxs]
                    mb_advantages= rollout_data.advantages[mb_idxs]
                    mb_returns=rollout_data.returns[mb_idxs]

                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # recompute the logprobs and values using the updated policy
                    # compares what the old policy did with what the current policy would do
                    newlogprobs , entropys = self.agent.get_newlogprob_and_entropy(observation=mb_observations, old_action=mb_actions)
                    newvalues = self.agent.get_value(observation=mb_observations)

             
                    # Policy loss
                    ratios = torch.exp(newlogprobs - mb_oldlogprobs)
                    surr_loss1 = mb_advantages * ratios
                    surr_loss2 =  torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                    policy_loss= (-torch.min(surr_loss1, surr_loss2)).mean()

                    # Value loss
                    newvalues = newvalues.view(-1)
                    v_loss = (0.5 * ((newvalues - mb_returns) ** 2)).mean() # MSE

                    #entropy bonus
                    entropy_bonus = entropys.mean()

                    #Calculate loss and update 
                    loss = (policy_loss - self.ent_coef * entropy_bonus + self.vf_coef * v_loss )
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    # LOGGING
                    with torch.no_grad():
                        mb_policy_losses.append(policy_loss.item())
                        mb_entropy_bonuses.append(entropy_bonus.item())
                        mb_value_losses.append(v_loss.item())
                        mb_total_losses.append(loss.item())

                        self.logger.add_scalar(tag="train/minibatch_policy_loss", scalar_value=policy_loss.item(), step=self.total_minibatches)
                        self.logger.add_scalar(tag="train/minibatch_entropy_bonus", scalar_value=entropy_bonus.item(), step=self.total_minibatches)
                        self.logger.add_scalar(tag="train/minibatch_value_loss", scalar_value=v_loss.item(), step=self.total_minibatches)
                        self.logger.add_scalar(tag="train/minibatch_total_loss", scalar_value=loss.item(), step=self.total_minibatches)

                        self.total_minibatches += 1

                #LOGGING
                entropy_bonus_this_epoch=sum(mb_entropy_bonuses) / len(mb_entropy_bonuses)
                epoch_entropy_bonuses.append(entropy_bonus_this_epoch)

                policy_loss_this_epoch= sum(mb_policy_losses) / len(mb_policy_losses)
                epoch_policy_losses.append(policy_loss_this_epoch)

                value_loss_this_epoch=sum(mb_value_losses) / len(mb_value_losses)
                epoch_value_losses.append(value_loss_this_epoch)

                total_loss_this_epoch=sum(mb_total_losses) / len(mb_total_losses)
                epoch_total_losses.append(total_loss_this_epoch)
            
                self.logger.add_scalar(tag="train/epoch_total_loss", scalar_value=total_loss_this_epoch,step= self.total_epochs)
                self.logger.add_scalar(tag="train/epoch_policy_loss", scalar_value=policy_loss_this_epoch, step=self.total_epochs)
                self.logger.add_scalar(tag="train/epoch_value_loss", scalar_value=value_loss_this_epoch, step=self.total_epochs)
                self.logger.add_scalar(tag="train/epoch_entropy_bonus", scalar_value=entropy_bonus_this_epoch, step=self.total_epochs)

                self.total_epochs +=1
            # Iteration Complete
            print()
            del rollout_data #delete rollout data to stay on-policy

            #LOGGING
            iter_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses)
            self.logger.add_scalar("train/policy_loss", iter_policy_loss, iteration)

            iter_value_loss = sum(epoch_value_losses) / len(epoch_value_losses)
            self.logger.add_scalar("train/value_loss", iter_value_loss, iteration)

            iter_entropy_bonus = sum(epoch_entropy_bonuses) / len(epoch_entropy_bonuses)
            self.logger.add_scalar("train/entropy_bonus", iter_entropy_bonus, iteration)

            iter_total_loss = sum(epoch_total_losses) / len(epoch_total_losses)
            self.logger.add_scalar("train/total_loss", iter_total_loss, iteration)

            if iteration == self.num_iterations:
                print("Running Final Evaluations")
                self.evaluate(iteration)
                if args.save_model:
                    save_model(self.agent,self.run_name,iteration)




if __name__ == "__main__":
    ############ Process command line inputs ####################
    args = tyro.cli(Args)
    ppo_trainer=PPO_Trainer(args)
    ppo_trainer.run_training_loop()