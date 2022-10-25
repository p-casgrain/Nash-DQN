This repository contains the code for the Nash-DQN algorithm for general-sum multi-agent reinforcement learning.
The associated paper "Deep Q-Learning for Nash Equilibria: Nash-DQN" can be found at https://arxiv.org/abs/1904.10554.

INSTRUCTIONS:

"Nash DQN-Old" pertains to files associated with a prior verion of the paper.
"Nash DQN-Updated" pertains to files associated with the current latest version on arxiv.

To generate plots based on pre-trained network:
- Open file "Visualizations.ipynb" and run all cells

To Nash DQN train network and Fictitious Play network based on default parameters:
- Open file "NashDQN-Training.ipynb" and run cell with appropiate parameters
- Open file "FR-DDQN-Training.ipynb" and run cell with appropiate parameters
- Open file "Visualizations.ipynb"
- Change save file location for both networks to the designated file name generated from the two previous notebooks
- Run all cells to generate plots
