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

    def select_action(self, state: torch.Tensor, eval=False) -> torch.Tensor:
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
            
        if not eval:
            # save pieces needed for the PPO loss
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(logprob)
            self.buffer.state_values.append(value)
            
        # rewards and terminal flags are added *after* env.step()
        # print("state shape:", state.shape)
        # print("action shape:", action.shape)
        # print("logprob shape:", logprob.shape)
        # print("value shape:", value.shape)
        # return numpy array for the environment API
        return action.cpu().numpy()

    def update(self):
        '''
        Update the agent using PPO
        '''
        # Unpack replay buffer
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        # print("old_states shape:", old_states.shape)
        # print("old_state_values shape:", old_state_values.shape)
        # Loop through buffer rewards backwards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            # import ipdb; ipdb.reset_trace()
            if is_terminal:
                discounted_reward = 0
            # Determine discounted reward + and add to list
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        print(rewards.shape)
        print(old_state_values.shape)
        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Loop through epochs
        for _ in range(self.K_epochs):

            # use the policy's critic to evaluate the state action pairs in the buffer
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            ########## LOSS CALCULATION ##########
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_loss = -torch.min(surr1, surr2)
            value_loss = self.mse_loss(state_values, rewards)
            entropy_loss = -0.01 * dist_entropy
            loss = (policy_loss + 0.5 * value_loss + entropy_loss).mean()
            ######################################
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # the old_policy is updated with the updated policy's parameters
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer to make PPO on policy
        self.buffer.reset()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))