# Deep Q-learning Network(DQN) 

## Installation
Type the following command to install OpenAI Gym Atari environment in your **virutal environment**.

`$ pip install opencv-python-headless gym==0.10.4 gym[atari]`

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

## How to run :
training DQN:
* `$ python main.py --train_dqn`

testing DQN:
* `$ python test.py --test_dqn`

## Goal
In this project, I implement DQN to play [Breakout](https://gym.openai.com/envs/Breakout-v0/). This project is using Python 3[Pytorch](https://pytorch.org/). The goal is to get averaging reward in 100 episodes over **40 points** in **Breakout**, with OpenAI's Atari wrapper & unclipped reward. For more details, please see the [slides](https://docs.google.com/presentation/d/1CbYqY5DfXQy4crBw489Tno_K94Lgo7QwhDDnEoLYMbI/edit?usp=sharing).

<img src="https://github.com/yingxue-zhang/DS595CS525-RL-Projects/blob/master/Project3/materials/project3.png" width="80%" >

* **Trained Model**
  * Model file (.pth)
  * If your model is too large for Canvas, upload it to a cloud space and provide the download link 
  * Special skills: Include the skills which can improve the generation quality. Here are some [tips](https://arxiv.org/pdf/1710.02298.pdf) may help. (Optional)
  * Visualization: Learning curve of DQN. 
    * X-axis: number of episodes
    * Y-axis: average reward in last 30 episodes.
    
    <img src="https://github.com/yingxue-zhang/DS595CS525-RL-Projects/blob/master/Project3/materials/plot.png" width="60%" >

* **Python Code**
  * All the code is implemented including sample codes.
  
* **Python Code**
  * You can get full credits if the scripts can run successfully, otherwise you may loss some points based on your error.

## Tips for Using GPU on Google Cloud
* [How to use Google Cloud Platform](https://docs.google.com/document/d/1JfIG_yBi-xEIdT6KP1-eUpgLDoY3t2QrAKULB9yf01Q/edit?usp=sharing)
* [How to use Pytorch on GPU](https://docs.google.com/document/d/1i8YawKjEwg7qpfo7Io4C_FvSYiZxZjWMLkqHfcZMmaI/edit?usp=sharing)
* Other choice for GPU
  * Use your own GPU
  * Apply [Ace account](https://arc.wpi.edu/computing/accounts/) or[Turing account](https://arc.wpi.edu/computing/accounts/) from WPI 
