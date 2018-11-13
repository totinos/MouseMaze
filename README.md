# MouseMaze

MouseMaze is a Reinforcement Learning project that teaches a mouse how to navigate a maze such as to avoid traps and acquire cheese.

## Installation

The main file _MouseMaze.py_ in this project only requires _Python3_ and the _numpy_ and _Matplotlib_ libraries as dependencies. The script _policy_visualizer.py_ requires the _tkinter_ and _Pillow_ libraries as dependencies. These can all be installed using pip if necessary, i.e.

```Bash
~> sudo pip3 install Pillow
```

## Usage

```Bash
~> python3 MouseMaze.py d m n a g e
```

Where _d_ represents the grid dimension, _m_ represents the number of traps randomly placed (-1 for the default setup), _n_ represents the number of steps used in the _n_-step SARSA algorithm (use the string "float('inf')" to run Monte Carlo), _a_ is the value of alpha, _g_ is the discount factor, and _e_ is the value of epsilon. The default setup can be run using (this example uses "float('inf')" to run Monte Carlo):

```Bash
python3 MouseMaze.py 8 -1 "float('inf')" 0.1 0.9 0.01
```

The following is another way to run the program:

```Bash
~> python3 MouseMaze.py d m n a g e t c s
```

Where the parameters _d_ through _e_ did not change, and _t_ represents the reward for stepping into a trap, _c_ is the reward for getting the cheese, and _s_ is the reward for taking a step in any direction from any legal state in the grid.

For simplicity, the program can be run with no command line arguments at all, e.g.

```Bash
~> python3 MouseMaze.py
```

This method is equivalent to the default setup above.

Sometimes it is nice to be able to change the number of episodes that is used to generate new data. There is not currently any modular feature that allows this behavior. However, the number of episodes that is run is a parameter that can be changed from within the contstructor of the GRID class. Class functions _print_action_values()_ and _export_policy()_ are provided to read the action values at the end of training and to write the optimal policy to an output file that can be interpreted by the program policy_visualizer.py. This file is called "policy_out". Once the policy has been exported, the policy vizualizer can be run using:

```Bash
~> python3 policy_visualizer.py
```
