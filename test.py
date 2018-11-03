import sys
import numpy as np
import matplotlib.pyplot as plt


class GRID:

    def __init__(self, grid_dim, num_traps, num_steps, alpha, discount, epsilon, trap_reward, cheese_reward):

        # Copy the problem parameters into class variables
        self.grid_dim = grid_dim
        self.num_traps = num_traps
        self.num_steps = num_steps
        self.alpha = alpha
        self.discount = discount
        self.epsilon = epsilon
        self.trap_reward = trap_reward
        self.cheese_reward = cheese_reward

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
        self.policy = np.zeros((self.grid_dim, self.grid_dim, 4), dtype=int)

        # Set up a 2-D array representing trap locations
        self.trap_locations = np.zeros((self.grid_dim, self.grid_dim), dtype=int)

        # Generate trap locations randomly, or use default from instructions
        if self.num_traps == -1:
            trap_indices = set([(3, 1), (6, 1), (5, 2), (2, 3), (4, 3), (7, 3), (6, 4), (3, 5), (5, 6), (6, 6)])
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

        ####################### TESTING ######################
        self.sarsa(20000)

    def sarsa(self, num_episodes):

        episode_lengths = np.zeros(num_episodes)
        episode_rewards = np.zeros(num_episodes)
        avg_reward = np.zeros(num_episodes)
        reward_total = 0

        for episode in range(num_episodes):

            if ((episode + 1) % 1000 == 0):
                print('Episode {0}/{1}.'.format(episode+1, num_episodes))

            # Reset to starting position and choose the first action
            state = self.START_POS
            #action_probs = self.policy(state[0], state[1])
            action = self.choose_action(state)

            # One step in the environment
            t = 0
            while (True):

                #print('state:', state)
                #print('action', action)

                # Take a step in the episode
                next_state, reward, done = self.take_action(state, action)

                # Determine the next action
                next_action = self.choose_action(next_state)

                # Update the episode information
                episode_lengths[episode] = t
                episode_rewards[episode] += reward

                # TD update
                td_target = reward + self.discount * self.action_values[next_state[0], next_state[1], next_action]
                td_delta = td_target - self.action_values[state[0], state[1], action]
                self.action_values[state[0], state[1], action] += self.alpha * td_delta

                # Check to see if the end of the episode has been reached
                if (done):
                    break

                action = next_action
                state = next_state
                t += 1

            reward_total += episode_rewards[episode]
            avg_reward[episode] = reward_total / episode

        print(self.action_values)
        plt.figure()
        plt.plot(episode_lengths)
        plt.figure()
        plt.plot(episode_rewards)
        plt.figure()
        plt.plot(avg_reward)
        plt.show()
        return

    def choose_action(self, pos):

        actions = [self.policy[pos[0], pos[1], a] for a in range(4)]
        action_values = np.array([self.action_values[pos[0], pos[1], a] for a in range(4)])
        best_actions = (action_values == np.amax(action_values))
        num_best = np.sum(best_actions)

        # If all actions are equal, choose randomly between them
        if (num_best == 4):
            return np.random.randint(len(actions))

        # If explore option taken, then choose a random non-optimal action
        elif (np.random.binomial(1, self.epsilon) == 1):
            non_optimal = [a for a in range(4) if (best_actions[a] != True)]
            return non_optimal[np.random.randint(len(non_optimal))]

        # Otherwise randomly choose from the optimal actions
        else:
            optimal = [a for a in range(4) if (best_actions[a] == True)]
            return optimal[np.random.randint(len(optimal))]

            
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
        done = False

        # if next pos is a trap
        if self.trap_locations[row, col]:
            reward = self.trap_reward
            done = True
        
        # if next pos is the goal
        if self.CHEESE_POS == next_pos:
            reward = self.cheese_reward
            done = True

        return next_pos, reward, done
        

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
    # THE LAST TWO COMMAND LINE ARGUMENTS ALLOW THE TRAP AND CHEESE REWARDS TO BE MODIFIED
    grid = GRID(grid_dim, num_traps, num_steps, alpha, discount, epsilon, 0, 1)


        





            


