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
        self.START_POS = (0, 0)
        self.CHEESE_POS = (self.grid_dim-1, self.grid_dim-1)

        # Set up action values and policy as 3-D arrays
        self.action_values = np.zeros((self.grid_dim, self.grid_dim, 4))
        self.policy = np.zeros((self.grid_dim, self.grid_dim, 4))    # USE np.argmax or arr == np.amax(arr)

        # Set up a 2-D array representing trap locations
        self.trap_locations = np.zeros((self.grid_dim, self.grid_dim), dtype=int)

        # Generate the locations of the traps randomly, or use default
        # default case, use the example on the instructions
        if self.num_traps == -1:
            trap_indices = set([(3, 1), (6, 1), (5, 2), (2, 3), (4, 3), (7, 3), (6, 4), (3, 5), (5, 6), (6, 6)])
        # else place traps randomly
        else:
            trap_indices = set()
            while len(trap_indices) < self.num_traps:
                    trap_x = np.random.randint(self.grid_dim)
                    trap_y = np.random.randint(self.grid_dim)
                    loc = (trap_x, trap_y)
                    if (loc != self.START_POS and loc != self.CHEESE_POS):
                        if (loc not in trap_indices):
                            trap_indices.add(loc)

        for loc in trap_indices:
            x = loc[0]
            y = loc[1]
            self.trap_locations[x, y] = 1
        print(self.trap_locations)

    def generate_episode(self):
        mouse_pos = self.START_POS

        while (True):
            a = self.choose_action(mouse_pos)
            # todo: working here, committing after adding the choose_action function


    def choose_action(self, pos):
        action_values = [self.policy[pos[0], pos[1], a] for a in range(0, 4)]
        max_value = max(action_values)
        num_max_actions = action_values.count(max_value)

        action = None

        # special case, all actions are considered equal, then epsilon case or not, 
        # we will select equally among them
        if num_max_actions == 4:
            action = np.random.randint(4)

        # else use probability to check for episilon case. if random value is greater than or equal
        # to epsilon, then we are doing the greedy case
        elif (np.random.random() >= self.epsilon):
            # only one optimal action, take it
            if (num_max_actions == 1):
                action = action_values.index(max_value)
            # else multiple optimal actions
            else:
                # randomly select which of the optimal values to take
                max_actions = []
                for i in range(0, 4):
                    if action_values[i] == max_value:
                        max_actions.append(i)
        
                action = max_actions[np.random.randint(num_max_actions)]

        # if random test fails, we do the epilon case and choose a random non-optimal action
        else:
            non_optimal_actions = []
            for i in range(0, 4):
                if action_values[i] != max_value:
                    non_optimal_actions.append(i)
            
            action = non_optimal_actions[np.random.randint(len(non_optimal_actions))]

        return action
        

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
