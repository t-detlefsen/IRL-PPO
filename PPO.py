import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import List

class ReplayBuffer:
    '''
    Buffer to store agent experiences
    '''
    def __init__(self):
        # Initialize arrays
        self.reset()

    def add_rollouts(self,
                     actions: torch.tensor,
                     states: torch.tensor,
                     logprobs: torch.tensor,
                     rewards: torch.tensor,
                     state_values: torch.tensor,
                     is_terminals: torch.tensor):
        '''
        Add rollouts to the replay buffer

        Args:
            actions (torch.Tensor): Selected actor actions (B, action_dim)
            states (torch.Tensor): Observations fed to the policy (B, state_dim)
            logprobs (torch.Tensor): Log π(a\|s) from the actor (B, 1)
            rewards (torch.Tensor): Rewards returned **after** the step (B,)
            state_values (torch.Tensor): Critic estimates V(s) (B, 1)
            is_terminals (torch.Tensor): Episode-done flags (bool/float) (B,)
        '''
        self.actions.append(actions.detach())
        self.states.append(states.detach())
        self.logprobs.append(logprobs.detach())
        self.rewards.append(rewards.detach())
        self.state_values.append(state_values.detach())
        self.is_terminals.append(is_terminals.detach())

    def reset(self):
        '''
        Reset the replay buffer 
        '''
        self.actions:      List[torch.Tensor] = []
        self.states:       List[torch.Tensor] = []
        self.logprobs:     List[torch.Tensor] = []
        self.rewards:      List[torch.Tensor] = []
        self.state_values: List[torch.Tensor] = []
        self.is_terminals: List[torch.Tensor] = []

class ActorCritic(nn.Module):
    '''
    Actor-Critic Agent:

    3-layer 256-unit MLP for both actor (mean) and critic,
    matching ManiSkill's default PPO baseline.
    Continuous actions only.
    '''
    def __init__(self, state_dim: int, action_dim: int,
                 action_std_init: float = 0.6) -> None:
        super().__init__()
        self.action_dim = action_dim

        # internal helper for orthogonal init
        def init(layer: nn.Module, std: float = 2 ** 0.5, bias: float = 0.0):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias)
            return layer

        # critic 
        self.critic = nn.Sequential(
            init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            init(nn.Linear(256, 256)),
            nn.Tanh(),
            init(nn.Linear(256, 256)),
            nn.Tanh(),
            init(nn.Linear(256, 1)),
        )

        # actor (mean branch)
        self.actor_mean = nn.Sequential(
            init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            init(nn.Linear(256, 256)),
            nn.Tanh(),
            init(nn.Linear(256, 256)),
            nn.Tanh(),
            init(nn.Linear(256, action_dim), std=0.01 * (2 ** 0.5)),
        )

        # State‑independent log‑std parameter
        self.actor_logstd = nn.Parameter(
            torch.ones(1, action_dim) * torch.log(torch.tensor(action_std_init))
        )

    def act(self, state: torch.tensor):
        '''
        Select an action given the current state.

        Args:
            state (torch.Tensor): Tensor of current shape (state_dim, ) or (B, state_dim)

        Returns:
            action (torch.Tensor): Sampled action -(action_dim, ) or (B, action_dim)
            prob   (torch.Tensor): Log π(a\|s)    - (1,) or (B, 1)
            value  (torch.Tensor): Critic value V(s) - (1,) or (B, 1)
        '''

        if state.dim() == 1:
            state = state.unsqueeze(0)

        mean = self.actor_mean(state)
        std  = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)

        action  = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        value   = self.critic(state)

        return action.squeeze(0), logprob.squeeze(0), value.squeeze(0)

    def evaluate(self, state, action):
        '''
        Select action given current state

        Args:
            state (torch.Tensor): Batch of states (B, state_dim)
            action (torch.Tensor): Batch of actions (B, action_dim)

        Returns:
            prob (torch.Tensor): Log π(a\|s)   - (B, 1)
            value (torch.Tensor): V(s)         - (B, 1)
            entropy (torch.Tensor): Entropy of π - (B, 1)
        '''
        mean = self.actor_mean(state)
        std  = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)

        logprob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value   = self.critic(state)

        return logprob, value, entropy

class PPO:
    """
    Proximal Policy Optimisation wrapper around our Actor Critic network.
    Only the initialisation logic and action selection path are filled
    here,  the update routine can stay exactly like the ManiSkill reference.
    """
    def __init__(
        self,
        state_dim:   int,
        action_dim:  int,
        lr_actor:    float = 3e-4,
        lr_critic:   float = 3e-4,
        gamma:       float = 0.99,
        K_epochs:    int   = 4,
        eps_clip:    float = 0.2,
        action_std_init: float = 0.6,
        device: str | torch.device = "cpu",
    ):
        # All core hyper‑parameters
        self.gamma      = gamma          # discount factor
        self.eps_clip   = eps_clip       # PPO clip range
        self.K_epochs   = K_epochs       # gradient epochs per update
        self.device     = torch.device(device)

        # Experience storage
        self.buffer = ReplayBuffer()

        # Networks
        # current policy (θ) and a frozen snapshot (θ_old) used to
        # compute the importance‑sampling ratio π_θ / π_θ_old
        self.policy      = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
        self.policy_old  = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())   # θ_old ← θ

        # Optimiser
        #   - actor_mean parameters + actor_logstd share lr_actor
        #   - critic parameters get lr_critic
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor_mean.parameters(), "lr": lr_actor},
                {"params": [self.policy.actor_logstd],         "lr": lr_actor},
                {"params": self.policy.critic.parameters(),    "lr": lr_critic},
            ]
        )

        # value‑function loss
        self.mse_loss = nn.MSELoss()

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Query π_θ_old to obtain an action and log probability, then stash
        everything we;ll need later into the replay buffer.

        Args:
            state (torch.Tensor): single observation or batch shape (state_dim)  or  (B, state_dim)

        Returns:
            action (torch.Tensor): action to execute in the env shape (action_dim) or (B, action_dim)
        """
        # ensure tensor, move to device
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        # no gradients while sampling from θ_old
        with torch.no_grad():
            action, logprob, value = self.policy_old.act(state)

        # save pieces needed for the PPO loss
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.state_values.append(value)
        # rewards and terminal flags are added *after* env.step()

        # return numpy array for the environment API
        return action.cpu().numpy()

    def update(self):
        '''
        Update the agent using PPO
        '''
        # TODO: Loop through buffer rewards backwards
            # TODO: Determine discounted reward + and add to list

        # TODO: Normalize rewards

        # TODO: Calculate Advantages

        # TODO: Loop through epochs
            # TODO: Evaluate policy on old policy states + actions
            # TODO: Calculate ratio
            # TODO: Calculate Surrogate Loss + Clip
            # TODO: Take gradient step

        # TODO: Update old policy
        # TODO: Clear replay buffer

        pass

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))