import torch
from torch import nn

class ReplayBuffer:
    '''
    Buffer to store agent experiences
    '''
    def __init__(self):
        # TODO: Initialize arrays
        pass

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
            actions (torch.tensor): Selected actor actions (TODO: dims)
            states (torch.tensor): Resultant actor states (TODO: dims)
            logprobs (torch.tensor): Actor action distributions (TODO: dims)
            rewards (torch.tensor): Obtained rewards (TODO: dims)
            state_values (torch.tensor): Critic values (TODO: dims)
            is_terminals (torch.tensor): End of rollout (TODO: dims)
        '''
        # TODO: Update arrays

        pass

    def reset(self):
        '''
        Reset the replay buffer 
        '''
        # TODO: Reset arrays
        
        pass

class ActorCritic(nn.Module):
    '''
    Actor-Critic Agent
    '''
    def __init__(self):
        pass

        # TODO: Initialize Params
        # TODO: Initialize Actor
        # TODO: Initialize Critic

    def act(self, state: torch.tensor):
        '''
        Select action given current state

        Args:
            state (torch.tensor): tensor of current state (TODO: dims)
        Returns:
            action (torch.tensor): Action selected by actor (TODO: dims)
            prob (torch.tensor): Log probability of action (TODO: dims)
            value (torch.tensor): Value determined by critic (TODO: dims)
        '''

        # TODO: Query Actor for distribution
        # TODO: Sample distribution
        # TODO: Get log probabilities
        # TODO: Query Critic for state values
        # TODO: Return action, probs, states

        action = torch.zeros(1)
        prob = torch.zeros(1)
        value = torch.zeros(1)

        return action, prob, value

    def evaluate(self, state, action):
        '''
        Select action given current state

        Args:
            state (torch.tensor): tensor of current state (TODO: dims)
            action (torch.tensor): Action selected (TODO: dims)
        Returns:
            prob (torch.tensor): Log probability of action (TODO: dims)
            value (torch.tensor): Value determined by critic (TODO: dims)
            entropy (torch.tensor): Entropy of action distribution (TODO: dims)
        '''

        # TODO: Query Actor for distribution
        # TODO: Get log probabilities
        # TODO: Get distribution entropy
        # TODO: Query critic for state values
        # TODO: Return probs, states, entropy

        prob = torch.zeros(1)
        value = torch.zeros(1)
        entropy = torch.zeros(1)

        return prob, value, entropy

class PPO:
    '''
    Actor-Critic Method trained via PPO
    '''
    def __init__(self):
        # TODO: Initialize params
        # TODO: Initialize Replay Buffer
        # TODO: Initialize Old Policy (Actor / Critic)
        # TODO: Initialize Policy (Actor / Critic)
        # TODO: Initialize Optimizer + Loss

        pass

    def select_action(self, state: torch.tensor) -> torch.tensor:
        '''
        Select an action using the agent

        Args:
            state (torch.tensor): tensor of current state (TODO: dims)
        Returns:
            action (torch.tensor): tensor of action (TODO: dims)
        '''
        # TODO: Query Old Policy
        # TODO: Add information to replay buffer
        # TODO: Return action

        action = torch.zeros(1)

        return action

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