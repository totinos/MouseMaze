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
        self.policy = np.ones((self.grid_dim, self.grid_dim, 4))*0.25

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

        # Add arrays to keep track of plotting information
        self.avg_reward = []
        self.avg_length = []

        ####################### TESTING ######################
        # Set the number of steps, reset the action values and policy, and then run algorithm
        # self.sarsa(5000)
        self.num_steps = 1
        self.action_values = np.zeros((self.grid_dim, self.grid_dim, 4))
        self.policy = np.ones((self.grid_dim, self.grid_dim, 4))*0.25
        self.n_step_sarsa(5000)
        # self.num_steps = 2
        self.action_values = np.zeros((self.grid_dim, self.grid_dim, 4))
        self.policy = np.ones((self.grid_dim, self.grid_dim, 4))*0.25
        self.n_step_sarsa(5000)
        # self.num_steps = 4
        self.action_values = np.zeros((self.grid_dim, self.grid_dim, 4))
        self.policy = np.ones((self.grid_dim, self.grid_dim, 4))*0.25
        self.n_step_sarsa(5000)
        # self.num_steps = 8
        self.action_values = np.zeros((self.grid_dim, self.grid_dim, 4))
        self.policy = np.ones((self.grid_dim, self.grid_dim, 4))*0.25
        self.n_step_sarsa(5000)
        # self.num_steps = 16
        self.action_values = np.zeros((self.grid_dim, self.grid_dim, 4))
        self.policy = np.ones((self.grid_dim, self.grid_dim, 4))*0.25
        self.n_step_sarsa(5000)
        self.plot_learning()

    def sarsa(self, num_episodes):
        """Performs one-step on-policy SARSA for a given number of episodes.

        Uses a single-step temporal difference reinforcement learning
        algorithm, SARSA, to update the action values for each of the
        different states of the problem. Also tracks episode length and
        reward so that they can be displayed graphically to the user.

        Args:
            num_episodes (int): The number of episodes to generate

        """


        # Keep track of learning statistics
        episode_lengths = np.zeros(num_episodes)
        episode_rewards = np.zeros(num_episodes)
        avg_reward = np.zeros(num_episodes)
        avg_length = np.zeros(num_episodes)
        reward_total = 0
        length_total = 0

        for episode in range(num_episodes):

            if ((episode + 1) % 1000 == 0):
                print('Episode {0}/{1}.'.format(episode+1, num_episodes))

            # Reset to starting position and choose the first action
            state = self.START_POS
            action = self.choose_action(state)

            # One step in the environment
            t = 0
            while (True):

                #print('state:', state)
                #print('action', action)

                # Take a step in the episode
                next_state, reward, terminal = self.take_action(state, action)

                # Determine the next action
                next_action = self.choose_action(next_state)

                # Update the episode information
                episode_lengths[episode] = t
                episode_rewards[episode] += reward

                # TD update
                td_target = reward + self.discount * self.action_values[next_state[0], next_state[1], next_action]
                td_delta = td_target - self.action_values[state[0], state[1], action]
                self.action_values[state[0], state[1], action] += self.alpha * td_delta
                self.update_policy(state)

                # Check to see if the end of the episode has been reached
                if (terminal):
                    break

                action = next_action
                state = next_state
                t += 1

            # Update the learning statistics
            reward_total += episode_rewards[episode]
            length_total += episode_lengths[episode]
            avg_reward[episode] = reward_total / (episode + 1)
            avg_length[episode] = length_total / (episode + 1)

        # Add to class episode information
        self.avg_reward.append(avg_reward)
        self.avg_length.append(avg_length)
        return

    def n_step_sarsa(self, num_episodes):
        """Performs n-step on-policy SARSA for a given number of episodes.

        Performs n-step temporal difference reinforcement learning
        algorithm, n-step SARSA, to update the action values for each of the
        different states of the problem. Also tracks episode length and reward
        so that they can be displayed graphically to the user.

        Args:
            num_episodes (int): The number of episodes to generate

        """

        # Keep track of learning statistics
        episode_lengths = np.zeros(num_episodes)
        episode_rewards = np.zeros(num_episodes)
        avg_reward = np.zeros(num_episodes)
        avg_length = np.zeros(num_episodes)
        reward_total = 0
        length_total = 0

        for episode in range(num_episodes):

            # Update the user on training progress
            if ((episode + 1) % 1000 == 0):
                print('Episode {0}/{1}.'.format(episode+1, num_episodes))

            stored_states = {}
            stored_actions = {}
            stored_rewards = {}

            # Reset to starting position and choose the first action
            state = self.START_POS
            action = self.choose_action(state)

            # Initialize time step and set episode length
            time = 0
            end_time = float('inf')
            while (True):

                # Advance to the next time step
                time += 1

                # Generate new states/actions until terminal state reached
                if (time < end_time):

                    # Take a step in the episode
                    next_state, reward, terminal = self.take_action(state, action)

                    # Store episode information for n-step calculation
                    stored_states[time%self.num_steps] = state
                    stored_actions[time%self.num_steps] = action
                    stored_rewards[time%self.num_steps] = reward

                    # Determine the next action
                    next_action = self.choose_action(next_state)

                    # Update the episode information
                    episode_lengths[episode] = time
                    episode_rewards[episode] += reward

                    # Check to see if the end of the episode has been reached
                    if (terminal):
                        end_time = time

                
                update_time = time - self.num_steps
                if (update_time > 0):
                    returns = 0

                    # Sum the returns for the next n steps or until episode end
                    for t in range(update_time, min(end_time, update_time + self.num_steps)):
                        returns += (self.discount ** (t - update_time)) * stored_rewards[t%self.num_steps]
                    if (update_time + self.num_steps <= end_time):
                        returns += (self.discount ** self.num_steps) * self.action_values[next_state[0], next_state[1], next_action]

                    # Update the action value function
                    td_target = reward + returns
                    td_delta = td_target - self.action_values[state[0], state[1], action]
                    self.action_values[state[0], state[1], action] += self.alpha * td_delta
                    self.update_policy(state)
                
                # If there are no more updates to make, exit episode
                if (update_time == end_time - 1):
                    break

                # Update the state and action for the next time step
                action = next_action
                state = next_state

            # Update the learning statistics
            reward_total += episode_rewards[episode]
            length_total += episode_lengths[episode]
            avg_reward[episode] = reward_total / (episode + 1)
            avg_length[episode] = length_total / (episode + 1)

        # Add to class episode information
        self.avg_reward.append(avg_reward)
        self.avg_length.append(avg_length)
        return

    def update_policy(self, pos):
        """Updates the policy to be epsilon-greedy based on action values.

        Uses the action values stored by the class to determine the epsilon-
        greedy policy update. If there is more than one best action, they are
        all weighted equally.

        Args:
            pos (tuple): pos[0] is current row, pos[1] is current column.

        """

        action_values = np.array([self.action_values[pos[0], pos[1], a] for a in range(4)])
        best_actions = (action_values == np.amax(action_values))
        num_best = np.sum(best_actions)

        for a in range(4):
            if (num_best == 4):
                self.policy[pos[0], pos[1], a] = 1 / num_best
            elif (best_actions[a] == True):
                self.policy[pos[0], pos[1], a] = (1 - self.epsilon)/num_best
            else:
                self.policy[pos[0], pos[1], a] = self.epsilon/(4 - num_best)
        return

    def choose_action(self, pos):
        """Choose an action based on the current policy.

        Chooses an action based on the probabilites of the different actions
        from the current state.

        Args:
            pos (tuple): pos[0] is current row, pos[1] is current column.

        Returns:
            action (int): The index corresponding to the chosen action from
                          the current state.
        """

        action_probs = [self.policy[pos[0], pos[1], a] for a in range(4)]
        return np.random.choice(4, p=action_probs)

            
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
        terminal = False

        # if next pos is a trap
        if self.trap_locations[row, col]:
            reward = self.trap_reward
            terminal = True
        
        # if next pos is the goal
        if self.CHEESE_POS == next_pos:
            reward = self.cheese_reward
            terminal = True

        return next_pos, reward, terminal

    def plot_learning(self):
        # print('----------  POLICY  ----------')
        # print(self.policy)
        # print('---------- Q VALUES ----------')
        # print(self.action_values)

        plt.figure()
        for i in range(len(self.avg_length)):
            plt.plot(self.avg_length[i], label='{0}-step'.format(i))
        plt.title('Average Episode Length')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Average Episode Length')
        plt.legend(loc='upper right')

        plt.figure()
        for i in range(len(self.avg_reward)):
            plt.plot(self.avg_reward[i], label='{0}-step'.format(i))
        plt.title('Average Episode Reward')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Average Episode Reward')
        plt.legend(loc='upper left')

        plt.show()
        return
        

if __name__ == "__main__":

    # Set up the default parameters
    if (len(sys.argv) == 1):
        grid_dim = 8
        num_traps = -1
        num_steps = 12                 # TODO --> Make num_steps an array?
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


        





            


