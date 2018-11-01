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

        # Set up action values and policy as 3-D arrays
        self.action_values = np.zeros((self.grid_dim, self.grid_dim, 4))
        self.policy = np.zeros((self.grid_dim, self.grid_dim, 4))    # USE np.argmax or arr == np.amax(arr)

        # Set up a 2-D array representing trap locations
        self.trap_locations = np.zeros((self.grid_dim, self.grid_dim), dtype=int)

        # Generate the locations of the traps randomly, or use default
        # TODO --> Make default less lame
        if self.num_traps == -1:
            self.num_traps = 10
    
        # set starting point and ending point (cheese)
        self.start = (0, 0)
        self.cheese = (self.grid_dim-1, self.grid_dim-1)     

        # TODO --> Should (0,0) be allowed for a trap position? Probably not if the mouse has to start there
        #      --> What about the cheese? Is it always at the bottom right?
        trap_indices = set()
        while len(trap_indices) < self.num_traps:
                trap_x = np.random.randint(self.grid_dim)
                trap_y = np.random.randint(self.grid_dim)
                loc = (trap_x, trap_y)
                if (loc != self.start and loc != self.cheese):
                    if (loc not in trap_indices):
                        trap_indices.add(loc)

        for loc in trap_indices:
            x = loc[0]
            y = loc[1]
            self.trap_locations[x, y] = 1
        print(self.trap_locations)

   

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
