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