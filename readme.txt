Usage instructions
-------------------

Requirements:
- Python 3.6 (for Strict Baselines it must be 3.6)
- Tensorflow 1.8.0 (for String Baselines it must be 1.8.0)
- OpenAI Gym
- OpenAI Gym Sokoban environment
- maybe some other packages. if when running the "sokoban.py" it complains, please install the mentioned packages.

Note: Any version of Python 3 can be used for running sokoban.py, which includes MONTECARLO, SARSA and QLEARNING.
However, for running PPO and DQN, using Stable Baselines, the Python and Tensorflow versions mentioned above are required.
We used a Python environment for this.

MONTECARLO, SARSA and QLEARNING
--------------------------------
To test the program you should run "python sokoban.py", which, by default, runs the SARSA algorithm. 
To run Montecarlo or Qlearning just comment out the specific line.
Sarsa and Qlearning should be able to solve the puzzle.

DQN and PPO
-----------
These algorithms can be run using "python algorithms/ppo.py" and "python algorithms/dqn.py".
They are implemented but you should not expect good results from them.


EDITING THE puzzle
------------------
To change the puzzle to solve, you must edit ".sokoban_cache/medium/train/000.txt". This file specifies the puzzle using Ascii Art.
