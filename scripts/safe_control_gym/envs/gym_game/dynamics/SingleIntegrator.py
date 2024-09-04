import numpy as np
import heterocl as hcl

from safe_control_gym.envs.gym_game.dynamics.BaseDynamics import BaseDynamics
 
 
class SingleIntegrator(BaseDynamics):
    '''2D * num single integrator agents dynamics.
    x_dot = v * u1
    y_dot = v * u2
    '''
    def __init__(self, number, initials, frequency, uMin=-1, uMax=1, speed=1.0):
        ''' Initialize the dynamics of the agents.
        
        Args:
            number (int): the number of agents
            initials (np.ndarray): the initial states of all agents
        '''
        super().__init__(number=number, initials=initials, frequency=frequency)
        self.uMax = uMax
        self.uMin = uMin
        self.speed = speed
        assert self.dim == number*2, "The dimension of the initial states are not correct for the SingleIntegrator."
    

    def _dynamics(self, state, action):
        """Return the partial derivative equations of one agent.

        Args:
            state (np.ndarray, ): the state of one agent
            action (np.ndarray, shape (2, )): the action of one agent
        """
        dx = self.speed * action[0]
        dy = self.speed * action[1]
        return (dx, dy)
    

    def forward(self, state, action):
        """Update and return the next state of one agent with the action based on the Runge Kutta method.
        
        Args:
            state (np.ndarray, ): the state of one agent
            action (np.ndarray, shape (state_dim, )): the action of one agent

        """
        dx, dy = self._dynamics(state, action)
        # Compute the k1 terms
        k1_x = 1/self.frequency * dx
        k1_y = 1/self.frequency * dy
        
        # Compute the k2 terms
        k2_x = 1/self.frequency * dx
        k2_y = 1/self.frequency * dy
        
        # Compute the k3 terms
        k3_x = 1/self.frequency * dx
        k3_y = 1/self.frequency * dy
        
        # Compute the k4 terms
        k4_x = 1/self.frequency * dx
        k4_y = 1/self.frequency * dy
        
        # Combine to get the final state
        x_new = state[0] + (1/6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y_new = state[1] + (1/6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        
        return x_new, y_new


    def step(self, action):
        """Update and return the next state of all agents after executing the action.
        
        Args:
            action (np.ndarray, shape num_attacker x action_dim): the actions of all agents
        """
        for i in range(self.numbers):
            self.state[i] = self.forward(self.state[i], action[i])
    

    def _get_state(self):
        """Return the current states of all agents in the form of xxx.
        
        Returns:
            np.ndarray (shape (num_agents, state_dim)): the current states of all agents

        """
        return self.state
    
    