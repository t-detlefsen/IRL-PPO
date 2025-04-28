from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self,args) -> None:
        env_kwargs=args.env_kwargs
        self.log_wandb=args.track
        if not args.evaluate:
            print("Running training")
            if self.log_wandb:
                import wandb
                config = vars(args)
                config["env_cfg"] = dict(**env_kwargs,
                                        num_envs=args.num_envs,
                                        env_id=args.env_id, 
                                        reward_mode="normalized_dense", 
                                        env_horizon=args.max_episode_steps, 
                                        partial_reset=args.partial_reset)
                config["eval_env_cfg"] = dict(**env_kwargs, 
                                            num_envs=args.num_eval_envs, 
                                            env_id=args.env_id, 
                                            reward_mode="normalized_dense", 
                                            env_horizon=args.max_episode_steps,
                                            partial_reset=False)
                wandb.init(
                    project=args.wandb_project_name,
                    entity=args.wandb_entity,
                    sync_tensorboard=False,
                    config=config,
                    name=args.run_name,
                    save_code=True,
                    group="PPO",
                    tags=["ppo", "walltime_efficient"]
                )
            self.writer = SummaryWriter(f"runs/{args.run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),)
        else:
            self.writer=None
            print("Running evaluation")

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

