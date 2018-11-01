import sys
import numpy as np


class GRID:

    def __init__(self, grid_dim, num_traps, num_steps, alpha, discount, epsilon):

        # Copy the problem parameters into class variables
        self.grid_dim = grid_dim
        self.num_traps = num_traps
        self.num_steps = num_steps
        self.alpha = alpha
        self.discount = discount
        self.epsilon = epsilon

        # Use explicit names for the different actions for clarity
        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3

        # Define the starting and ending states
        self.START = [0, 0]
        self.GOAL = [7, 7]

        # Set up action values and policy as 3-D arrays
        self.action_values = np.zeros((self.grid_dim, self.grid_dim, 4))
        self.policy = np.zeros((self.grid_dim, self.grid_dim, 4))    # USE np.argmax or arr == np.amax(arr)

        # Set up a 2-D array representing trap locations
        self.trap_locations = np.zeros((self.grid_dim, self.grid_dim), dtype=int)

        # Generate the locations of the traps randomly, or use default
        # TODO --> Make default less lame
        if self.num_traps == -1:
            trap_indices = set([19,25,29,35,42,46,49,52,54,59])
    
        # TODO --> Should (0,0) be allowed for a trap position? Probably not if the mouse has to start there
        #      --> What about the cheese? Is it always at the bottom right?
        else:
            trap_indices = set()
            for i in range(self.num_traps):
                while (True):
                    x = np.random.randint(self.grid_dim ** 2)
                    if (x not in trap_indices):
                        trap_indices.add(x)
                        break
        for i in trap_indices:
            self.trap_locations[i//self.grid_dim][i%self.grid_dim] = 1
        print(self.trap_locations)

        # TODO --> Add mouse and cheese locations to the class, differentiate between traps and cheese
    

    # Take an action according to the policy in the grid
#    def step(self):
        # Choose an action according to the policy (but epsilon greedy)
#        if (np.random.binomial(1, self.epsilon) == 1):
#            action = np.random.choice(self.action_values[

    # Performs n-step Temporal Difference learning
#    def sarsa(self):
        

if __name__ == "__main__":

    # Set up the default parameters
    if (len(sys.argv) == 1):
        grid_dim = 8
        num_traps = -1
        num_steps = 1                 # TODO --> Make num_steps an array?
        alpha = 0.1
        discount = 1
        epsilon = 0.01

    # Read parameters from the command line
    elif (len(sys.argv) == 7):
        grid_dim = int(sys.argv[1])
        num_traps = int(sys.argv[2])
        num_steps = int(sys.argv[3])
        alpha = float(sys.argv[4])
        discount = float(sys.argv[5])
        epsilon = float(sys.argv[6])

    # Create the grid for the problem
    # TODO --> Create a grid
    grid = GRID(grid_dim, num_traps, num_steps, alpha, discount, epsilon)
