from utils import *
class ActorCritic(nn.Module):
    '''
    Actor-Critic Agent
    '''
    def __init__(self,
                obs_dim,
                act_dim, 
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
        actor_means= self.actor(observation)
        covariance_matrix = torch.diag(torch.exp(self.logstd)) #shape = [action_dim, action_dim]
        batch_size=batch_mean_from_nn.shape[0]
        batch_covariance_matrix= covariance_matrix.repeat(batch_size,1,1)#repeats the matrix so each batch has its own covariance matrix
        
        action_distro_out = distributions.MultivariateNormal(batch_mean_from_nn,scale_tril=batch_covariance_matrix)


    def get_action(self, state: torch.tensor): 
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
    

    