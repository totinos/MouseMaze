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
                    trap_row = np.random.randint(self.grid_dim)
                    trap_col = np.random.randint(self.grid_dim)
                    loc = (trap_row, trap_col)
                    if (loc != self.START_POS and loc != self.CHEESE_POS):
                        if (loc not in trap_indices):
                            trap_indices.add(loc)

        for loc in trap_indices:
            row = loc[0]
            col = loc[1]
            self.trap_locations[row, col] = 1
        print(self.trap_locations)

        # testing
        while True:
            self.generate_episode()

    def generate_episode(self):
        mouse_pos = self.START_POS

        state_backup = []
        action_backup = []
        reward_backup = []

        state_backup.append(mouse_pos)

        while (True):
            #print('current state: ', mouse_pos)

            a = self.choose_action(mouse_pos)
            action_backup.append(a)

            #print('chosen action: ', a)

            next_pos, reward = self.take_action(mouse_pos, a)

            if len(state_backup) > self.num_steps:
                updated_state = state_backup[0]
                updated_action = action_backup[0]

                g = 0
                for i in range(0, self.num_steps):
                    g += reward_backup[i]

                g += self.action_values[state_backup[-1][0], state_backup[-1][1], action_backup[-1]]

                cur_action_value = self.action_values[updated_state[0], updated_state[1], updated_action]

                self.action_values[updated_state[0], updated_state[1], updated_action] = cur_action_value + self.alpha * (g - cur_action_value)

                self.update_policy(updated_state)

                if g != 0 or next_pos == self.CHEESE_POS:
                    print('state backup:\n', state_backup)
                    print('action backup:\n', action_backup)
                    print('reward backup:\n', reward_backup)
                    print('updating state', updated_state, 'and action', updated_action, 'with action value', cur_action_value, 'with return', g)
                    print(self.action_values)
                    print(self.policy)
                del state_backup[0]
                del action_backup[0]
                del reward_backup[0]


            state_backup.append(next_pos)
            reward_backup.append(reward)

            #print('reward: ', reward)


            # fell in trap
            if self.trap_locations[next_pos[0], next_pos[1]]:
                #print('fell into a trap, end of episode')
                return

            # end of episode
            if next_pos == self.CHEESE_POS:
                # TODO: need to fix this function
                # Update all remaining states that haven't reached enough steps to update yet
                for i in range(0, len(state_backup)-1):
                    g = 0
                    updated_state = state_backup[i]
                    updated_action = action_backup[i]

                    for j in range(i, len(state_backup)-1):
                        g += reward_backup[j]
                    if i != len(state_backup)-1:
                        g += self.action_values[updated_state[0], updated_state[1], updated_action]

                    cur_action_value = self.action_values[updated_state[0], updated_state[1], updated_action]
                    
                    self.action_values[updated_state[0], updated_state[1], updated_action] = cur_action_value + self.alpha * (g - cur_action_value)

                    self.update_policy(updated_state)


                print('found cheese, end of episode')
                print(self.action_values)
                return

            mouse_pos = next_pos


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
        
    # returns tuple of (next_pos, reward)
    # where next_pos is itself a tuple (row, col)
    def take_action(self, pos, action):
        row = pos[0]
        col = pos[1]

        if action == self.UP:
            if row > 0:
                row -= 1
        elif action == self.DOWN:
            if row < self.grid_dim-1:
                row += 1
        elif action == self.LEFT:
            if col > 0:
                col -= 1
        elif action == self.RIGHT:
            if col < self.grid_dim-1:
                col += 1

        next_pos = (row, col)
        reward = 0

        # if next pos is a trap
        if self.trap_locations[row, col]:
            reward = 0
        
        # if next pos is the goal
        if self.CHEESE_POS == next_pos:
            reward = 1

        return next_pos, reward

    def update_policy(self, pos):
        action_values = [self.action_values[pos[0], pos[1], a] for a in range(0, 4)]

        max_value = max(action_values)
        num_max_actions = action_values.count(max_value)

        # only one maximum action
        if (num_max_actions == 1):
            max_a = action_values.index(max_value)
            for a in range(0, 4):
                if a == max_a:
                    self.policy[pos[0], pos[1], a] = 1
                else:
                    self.policy[pos[0], pos[1], a] = 0
        # multiple max actions
        else:
            p = 1 / num_max_actions

            for a in range(0, 4):
                if action_values[a] == max_value:
                    self.policy[pos[0], pos[1], a] = p
                else:
                    self.policy[pos[0], pos[1], a] = 0
        

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
