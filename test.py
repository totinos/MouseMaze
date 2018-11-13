import sys
import numpy as np
import matplotlib.pyplot as plt


class GRID:

    def __init__(self, grid_dim=8, num_traps=-1, num_steps=1, alpha=0.1, discount=1, epsilon=0.01, trap_reward=0, cheese_reward=1, step_reward=0):

        # Copy the problem parameters into class variables
        self.grid_dim = grid_dim
        self.num_traps = num_traps
        self.num_steps = num_steps
        self.alpha = alpha
        self.discount = discount
        self.epsilon = epsilon
        self.trap_reward = trap_reward
        self.cheese_reward = cheese_reward
        self.step_reward = step_reward

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
        self.step_reward_averages = []
        self.step_length_averages = []

        ##############################################################
        ##########                                          ##########
        ########## FOR NOW USING THIS TO EXECUTE ALGORITHMS ##########
        ##########                                          ##########
        ##############################################################

        # If Monte Carlo is used, exploring starts will be used by default
        self.every_visit = True

        # Exploring starts will not be used by default (only works for n-step SARSA)
        self.exploring_starts = False

        # Iterate over all step sizes
        num_runs = 1
        num_episodes = 8000
        # num_steps = [1, 2, 4, 8, 16, float('inf')]
        # num_steps = [1, 2, 4, 8, 16]
        num_steps = [self.num_steps]

        for n in num_steps:

            # Set up arrays to store averaged results
            self.num_steps = n
            self.avg_reward = np.zeros(num_episodes)
            self.avg_length = np.zeros(num_episodes)

            # Average results over some number of runs
            print('\n>>> Beginning first of {0} runs. <<<\n'.format(num_runs))
            for i in range(num_runs):

                print('---------- RUN {0} ----------'.format(i+1))

                # Reset the action values and policy, then run algorithm
                self.action_values = np.zeros((self.grid_dim, self.grid_dim, 4))
                self.policy = np.ones((self.grid_dim, self.grid_dim, 4))*0.25
                
                # Run Monte Carlo algorithm for n = infinity
                if (self.num_steps == float('inf')):
                    self.monte_carlo(num_episodes)

                # Run n-step SARSA algorithm for other values of n
                else:
                    self.n_step_sarsa(num_episodes)

            self.step_reward_averages.append(self.avg_reward.copy())
            self.step_length_averages.append(self.avg_length.copy())

        # self.plot_learning(num_runs, num_steps)
        # self.export_policy()
        # self.print_action_values()


    def sarsa(self, num_episodes):
        """Performs one-step on-policy SARSA for a given number of episodes.

        Uses a single-step temporal difference reinforcement learning
        algorithm, SARSA, to update the action values for each of the
        different states of the problem. Also tracks episode length and
        reward so that they can be displayed graphically to the user.

        Args:
            num_episodes (int): The number of episodes to generate.
        """


        # Keep track of learning statistics
        episode_lengths = np.zeros(num_episodes)
        episode_rewards = np.zeros(num_episodes)
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
            self.avg_reward[episode] += reward_total / (episode + 1)
            self.avg_length[episode] += length_total / (episode + 1)

        return

    def n_step_sarsa(self, num_episodes):
        """Performs n-step on-policy SARSA for a given number of episodes.

        Performs n-step temporal difference reinforcement learning
        algorithm, n-step SARSA, to update the action values for each of the
        different states of the problem. Also tracks episode length and reward
        so that they can be displayed graphically to the user.

        Args:
            num_episodes (int): The number of episodes to generate.
        """

        # Keep track of learning statistics
        episode_lengths = np.zeros(num_episodes)
        episode_rewards = np.zeros(num_episodes)
        reward_total = 0
        length_total = 0

        for episode in range(num_episodes):

            # Update the user on training progress
            if ((episode + 1) % 1000 == 0):
                print('Episode {0}/{1}.'.format(episode+1, num_episodes))

            stored_states = {}
            stored_actions = {}
            stored_rewards = {}

            # Uses exploring starts if told to do so (resets to some start position)
            if (self.exploring_starts):
                while True:
                    start_row = np.random.randint(self.grid_dim)
                    start_col = np.random.randint(self.grid_dim)
                    loc = (start_row, start_col)
                    if (loc != self.CHEESE_POS and self.trap_locations[start_row, start_col] != 1):
                        state = loc
                        break
            else:
                state = self.START_POS
                            

            # Reset to starting position and choose the first action
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

                    updated_state = stored_states[update_time%self.num_steps]
                    updated_action = stored_actions[update_time%self.num_steps]

                    # Sum the returns for the next n steps or until episode end
                    for t in range(update_time, min(end_time, update_time + self.num_steps)):
                        returns += (self.discount ** (t - update_time)) * stored_rewards[t%self.num_steps]
                    if (update_time + self.num_steps <= end_time):
                        returns += (self.discount ** self.num_steps) * self.action_values[next_state[0], next_state[1], next_action]

                    # Update the action value function
                    td_target = reward + returns
                    td_delta = td_target - self.action_values[updated_state[0], updated_state[1], updated_action]
                    self.action_values[updated_state[0], updated_state[1], updated_action] += self.alpha * td_delta
                    self.update_policy(updated_state)
                
                # If there are no more updates to make, exit episode
                if (update_time == end_time - 1):
                    break

                # Update the state and action for the next time step
                action = next_action
                state = next_state

            # Update the learning statistics
            reward_total += episode_rewards[episode]
            length_total += episode_lengths[episode]
            self.avg_reward[episode] += reward_total / (episode + 1)
            self.avg_length[episode] += length_total / (episode + 1)

        return

    def monte_carlo(self, num_episodes):
        """Performs Monte Carlo algorithm for a given number of episodes.

        Uses a Monte Carlo reinforcement learning algorithm to update the
        action values for each of the different states of the problem. Also
        tracks the average episode length and reward received over the number
        of episodes so that they can be displayed graphically to the user.

        Args:
            num_episodes (int): The number of episodes to generate.
        """

        # Keep track of learning statistics
        episode_lengths = np.zeros(num_episodes)
        episode_rewards = np.zeros(num_episodes)
        reward_total = 0
        length_total = 0

        for episode in range(num_episodes):

            if ((episode + 1) % 1000 == 0):
                print('Episode {0}/{1}.'.format(episode+1, num_episodes))

            # backup tree
            stored_states = {}
            stored_actions = {}
            stored_rewards = {}

            # Reset to starting position and choose the first action
            state = self.START_POS
            action = self.choose_action(state)

            # One step in the environment
            t = 0
            while (True):

                stored_states[t] = state
                stored_actions[t] = action

                # Take a step in the episode
                next_state, reward, terminal = self.take_action(state, action)

                # Determine the next action
                next_action = self.choose_action(next_state)

                stored_rewards[t] = reward
                stored_states[t+1] = next_state
                stored_actions[t+1] = next_action

                # Update the episode information
                episode_lengths[episode] = t
                episode_rewards[episode] += reward

                # Check to see if the end of the episode has been reached
                if (terminal):

                    # MC update (every visit)
                    if (self.every_visit):
                        for i in range(0, t+1):
                            updated_state = stored_states[i]
                            updated_action = stored_actions[i]

                            # calculate return from this state
                            g = 0
                            for j in range(i, t+1):
                                g = g + stored_rewards[j] * pow(self.discount, (j-i))

                            # Q(s, a) <-- Q(s, a) + alpha * [g - Q(s, a)]
                            current_value = self.action_values[updated_state[0], updated_state[1], updated_action]
                            self.action_values[updated_state[0], updated_state[1], updated_action] += self.alpha * (g - current_value)
                            self.update_policy(updated_state)

                    # MC update (first visit)
                    else:
                        visited = {}
                        for i in range(0, t+1):
                            updated_state = stored_states[i]
                            updated_action = stored_actions[i]

                            if (updated_state, updated_action) in visited.keys():
                                continue

                            visited[(updated_state, updated_action)] = True

                            # calculate return from this state
                            g = 0
                            for j in range(i, t+1):
                                g = g + stored_rewards[j] * pow(self.discount, (j-i))

                            # Q(s, a) <-- Q(s, a) + alpha * [g - Q(s, a)]
                            current_value = self.action_values[updated_state[0], updated_state[1], updated_action]
                            self.action_values[updated_state[0], updated_state[1], updated_action] += self.alpha * (g - current_value)
                            self.update_policy(updated_state)

                    break

                action = next_action
                state = next_state
                t += 1

            # Update the learning statistics
            reward_total += episode_rewards[episode]
            length_total += episode_lengths[episode]
            self.avg_reward[episode] += reward_total / (episode + 1)
            self.avg_length[episode] += length_total / (episode + 1)

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
            pos (tuple):  pos[0] is current row, pos[1] is current column.

        Returns:
            action (int): The index corresponding to the chosen action from
                          the current state.
        """

        action_probs = [self.policy[pos[0], pos[1], a] for a in range(4)]
        return np.random.choice(4, p=action_probs)

    def take_action(self, pos, action):
        """Takes a given action from a given position.

        Takes a given action from a given position, calculates the reward
        received for that action, and checks to see if the action taken leads
        to a terminal state. Returns the position after taking the action,
        the reward received for taking the action, and whether or not the
        next position is a terminal state.

        Args:
            pos (tuple):        pos[0] is the current row, pos[1] is current
                                column.
            action (int):       Denotes which direction to move. Constants are
                                defined within the class to make dealing with
                                actions simpler.

        Returns:
            next_pos (tuple):   next_pos[0] is the row you moved to,
                                next_pos[1] is the column you moved to.
            reward (int):       The reward received for taking the given action
                                from the given state.
            terminal (boolean): True if next_pos is a terminal state.
        """

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
        reward = self.step_reward
        terminal = False

        # if next pos is a trap
        if self.trap_locations[row, col]:
            reward += self.trap_reward
            terminal = True
        
        # if next pos is the goal
        if self.CHEESE_POS == next_pos:
            reward += self.cheese_reward
            terminal = True

        return next_pos, reward, terminal

    def plot_learning(self, num_runs=1, num_steps=[1]):
        """Plots the class' stored average episode reward and length.

        Plots the average episode reward for all different step sizes used for
        a given number of episodes, averaged again over a certain number of
        runs.

        Args:
            num_runs (int):   The number of runs over which to average the
                              data.
            num_steps (list): The number of TD steps for each set of data
                              to be plotted. MC has float('inf') steps.
        """

        print(num_runs)

        # Plot the average reward for all of the different step sizes
        plt.figure()
        for n in range(len(self.step_reward_averages)):
            # Average all of the values
            y_vals = np.true_divide(self.step_reward_averages[n], num_runs)
            if (num_steps[n] == float('inf')):
                plt.plot(y_vals, label='MC')
            else:
                plt.plot(y_vals, label='{0}-step'.format(num_steps[n]))
        plt.legend(loc='lower right')
        plt.title('Average of Averaged Episode Rewards')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Average Episode Reward')

        # Plot the average length for all of the different step sizes
        plt.figure()
        for n in range(len(self.step_length_averages)):
            # Average all of the values
            y_vals = np.true_divide(self.step_length_averages[n], num_runs)
            if (num_steps[n] == float('inf')):
                plt.plot(y_vals, label='MC')
            else:
                plt.plot(y_vals, label='{0}-step'.format(num_steps[n]))
        plt.legend(loc='upper right')
        plt.title('Average of Averaged Episode Lengths')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Average Episode Length')

        plt.show()
        return

    def export_policy(self):
        """Exports the policy to be interpreted by a visualizer.

        Exports the policy maintained by the class in a format that can be
        interpreted by a visualizer script. The format is simple. The first
        line of the file contains the number of rows and number of columns
        separated by a space. The remaining lines move sequentially through
        the possible row, column combinations and contain the string 'Trap',
        the string 'Cheese', or an integer which is a bitmask for the four
        possible actions from a state. From most significant bit to least
        significant bit, the mask is of the form: <LEFT DOWN RIGHT UP>. For
        example, the mask 1010 (integer 10) means the agent can move left or
        right, but not down or up.
        """

        filename = 'policy_out'
        with open(filename, 'w') as of:

            # Write the grid dimensions to the output file
            of.write('{0} {0}\n'.format(self.grid_dim))

            # Write all of the policy values to the file
            for r in range(self.grid_dim):
                for c in range(self.grid_dim):

                    if (self.trap_locations[r, c] == 1):
                        of.write('Trap\n')
                    elif ((r, c) == self.CHEESE_POS):
                        of.write('Cheese\n')
                    else:
                        action_probs = np.array([self.policy[r, c, a] for a in range(4)])
                        best_actions = (action_probs == np.amax(action_probs))
                        
                        # Find the best actions bitmask
                        # The bits are in the order LDRU
                        mask = 0
                        for a in range(4):
                            if (best_actions[a] == 1):
                                mask |= (1 << a)

                        of.write('{0}\n'.format(mask))
        return

    def print_action_values(self):
        """Exports the policy to be interpreted by a visualizer.

        Prints the action values to the screen in an understandable way. The
        action values are arranged in four blocks of size self.grid_dim by
        self.grid_dim. The four blocks represent the four different possible
        actions from each state. They are printed in the following order: UP,
        RIGHT, DOWN, LEFT.
        """

        for a in range(4):
            print('[')
            for r in range(self.grid_dim):
                print('[', end='')
                for c in range(self.grid_dim):
                    value = self.action_values[r, c, a]
                    print("{:.5f} ".format(value), end="")
                print(']')
            print(']')
        return
        

if __name__ == "__main__":

    # Set up the default parameters
    if (len(sys.argv) == 1):

        grid = GRID()

        # This line is for easy testing of different parameters
        # grid = GRID(grid_dim=8, num_traps=-1, epsilon=0.01, trap_reward=-14, cheese_reward=1, step_reward=-1)

    # Read parameters from the command line
    elif (len(sys.argv) == 7):
        grid_dim = int(sys.argv[1])
        num_traps = int(sys.argv[2])
        if (sys.argv[3] == "float('inf')"):
            num_steps = float('inf')
        else:
            num_steps = int(sys.argv[3])
        alpha = float(sys.argv[4])
        discount = float(sys.argv[5])
        epsilon = float(sys.argv[6])
        
        # Create the grid for the problem
        grid = GRID(grid_dim, num_traps, num_steps, alpha, discount, epsilon, trap_reward=0, cheese_reward=1, step_reward=0)

    # Read parameters from the command line (including extra parameters for the different reward values)
    elif (len(sys.argv) == 10):
        grid_dim = int(sys.argv[1])
        num_traps = int(sys.argv[2])
        if (sys.argv[3] == "float('inf')"):
            num_steps = float('inf')
        else:
            num_steps = int(sys.argv[3])
        alpha = float(sys.argv[4])
        discount = float(sys.argv[5])
        epsilon = float(sys.argv[6])
        trap_reward = int(sys.argv[7])
        cheese_reward = int(sys.argv[8])
        step_reward = int(sys.argv[9])
        
        # Create the grid for the problem
        grid = GRID(grid_dim, num_traps, num_steps, alpha, discount, epsilon, trap_reward, cheese_reward, step_reward)

    grid.export_policy()
    grid.print_action_values()
    print('FINISHED.')


        





            


