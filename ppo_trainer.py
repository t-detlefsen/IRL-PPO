import argparse
from utils import *
from PPO import PPO

class PPO_trainer(object):
    '''
    Train an RL agent using PPO
    '''
    def __init__(self, params):
        self.params = params

        # TODO: Setup PPO agent
        agent = PPO()

        # TODO: Setup environment

    def run_training_loop(self):
        '''
        Train the RL agent usng PPO
        '''
        # TODO: Loop through iterations
            # ----------- TRAINING -----------
            # TODO: Reset train environment
            # TODO: Loop through steps
                # TODO: Generate action
                # TODO: Interact w/ environment
                # TODO: Update agent
            # TODO: Evaluate performance

            # ---------- EVALUATION ----------
            # TODO: Evaluate if specified
                # TODO: Reset eval environment
                # TODO: Loop through steps
                    # TODO: Generate action
                    # TODO: Interact w/ environment
                # TODO: Evaluate performance
                # TODO: Save model?

        pass

