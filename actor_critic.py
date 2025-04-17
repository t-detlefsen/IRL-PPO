from utils import *
from torch import distributions
class ActorCritic(nn.Module):
    '''
    Actor-Critic Agent
    '''
    def __init__(self,
                obs_dim,
                act_dim,
                logstd:float=-0.5, 
                #Actor architecture
                n_hl_actor: int =3,# num of hidden layers
                hl_size_actor =256, # hidden layer sizes
                activation_actor: str = 'tanh',
                output_activation_actor: str = 'identity',
                # Critic architecture
                n_hl_critic: int =3,# num of hidden layers
                hl_size_critic =256, # hidden layer sizes
                activation_critic: str = 'tanh',
                output_activation_critic: str = 'identity',
                ):

        # TODO: Initialize Params
        # TODO: Initialize Actor
        # TODO: Initialize Critic
        self.logstd=logstd
        self.actor = build_mlp(
                                input_size=obs_dim,
                                output_size=act_dim,
                                n_layers=n_hl_actor,
                                size=hl_size_actor,
                                activation=activation_actor,
                                output_activation=output_activation_actor
                                )
        self.actor.to(get_device())

        self.critic = build_mlp(
                                input_size=obs_dim,
                                output_size=1,
                                n_layers=n_hl_critic,
                                size=hl_size_critic,
                                activation=activation_critic,
                                output_activation=output_activation_critic
                                )
        self.critic.to(get_device())

    def forward(self, observation: torch.FloatTensor):
        action_means= self.actor(observation) # the mean action from each observation in the batch
        covariance_matrix = torch.diag(torch.exp(self.logstd)) #shape = [action_dim, action_dim]
        batch_size=action_means.shape[0]
        batch_covariance_matrix= covariance_matrix.repeat(batch_size,1,1)#repeats the matrix so each batch has its own covariance matrix
        
        action_dist_out = distributions.MultivariateNormal(action_means,
                                                           scale_tril = batch_covariance_matrix)
        return action_dist_out


    def act(self, obs: torch.tensor): 
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

        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        
        dist_action= self.forward(observation=observation)
        
        action = dist_action.sample()
        action_logprob= dist_action.log_prob(action)
        value = self.critic(observation)

        return action.detach(), action_logprob.detach(), value.detach()

    def evaluate(self, observation, action):
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
        dist_action= self.forward(observation)
        act_logprob= dist_action.log_prob(action)
        values= self.critic(observation)
        entropy = dist_action.entropy()

        return act_logprob, values, entropy
    

    