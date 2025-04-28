from args import Args
import tyro
from utils import *
import os
from agent import Agent
import time

if __name__ == "__main__":
    ############ Process command line inputs ####################
    args = tyro.cli(Args)
    # args.batch_size = int(args.num_envs * args.num_steps)
    # args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        args.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}_nEnvs_{args.num_envs}_nEvalEnvs_{args.num_eval_envs}_iter_{args.num_iterations}_lr_{args.learning_rate}_clipCoef_{args.clip_coef}_vfCoef_{args.vf_coef}_entCoef_{args.ent_coef}"
    else:
        args.run_name = args.exp_name

    #seeding
    set_random_seeds(args.seed)

    #init gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    #get environments
    args.env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    envs,eval_envs=getEnvs(args)
    args.max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    # will run 1 full episode in training and eval
    args.num_steps=args.max_episode_steps
    args.num_eval_steps=args.max_episode_steps

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.total_timesteps= int(args.num_iterations * args.batch_size)

    
    #setup logger
    logger=Logger(args)

    #setup agent
    agent = Agent(args,envs).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    action_space_low, action_space_high = torch.from_numpy(envs.single_action_space.low).to(device), torch.from_numpy(envs.single_action_space.high).to(device)

    
    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)

    ######################## EVALUATE ##############################
        if iteration % args.eval_freq == 1:
            logger=agent.evaluate(eval_envs,logger,global_step)
            if args.evaluate:
                break
            if args.save_model:
                save_model(agent,args.run_name,iteration)
    ###################################################################
        # Annealing the rate if instructed to do so.

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_time = time.time()

     ################# Collecting training data ##########################
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"][done_mask]).view(-1)
        ################# ENd Collecting training data ##########################

        #
        rollout_time = time.time() - rollout_time
        # bootstrap value according to termination and truncation


        ############COMPUTE ADVANTAGES AND RETURNS ##################################

        #### Compute Values #########################
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done

        ##### Compute Advantages ################################## 
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated

        ################# Compute Returns ###############################
            returns = advantages + values
##############################################################################

        # flatten the batch
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
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

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

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    envs.close()
    eval_envs.close()
