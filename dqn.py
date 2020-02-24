import gym
import math
import random
import numpy as np

# conda install -c conda-forge matplotlib
# https://anaconda.org/conda-forge/matplotlib
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple
#from itertools import count
#from PIL import Image

# conda install -c pytorch pytorch
# https://anaconda.org/pytorch/pytorch
import torch
import torch.nn as nn
#import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T


'''
Reinforcement learning involves:
  an Agent
  a set of States (numeric rep of what agent sees)
  a set of Actions per state (the Agent does something)
  reward, the sum of good things - bads things
  
By performing an action:
  the Agent transitions from state to state and
  receives a reward (a numerical score).

The goal of the Agent is to maximize reward. A

'''


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

# if is_ipython:
#     from IPython import display

# Turn the interactive mode on.
plt.ion()

# pylint: disable=E1101
# tensor = torch.from_numpy(np_array)
# pylint: enable=E1101

# if gpu is to be used
# A pylint problem workaround
# pylint: disable=E1101
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pylint: enable=E1101
print(device)
print(torch.__version__)


'''
Transition:
  a named tuple representing a single transition in 
  our environment. It essentially maps (state, action) pairs to their 
  (next_state, reward) result, with the state being the screen difference 
  image as described later on.
'''

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

'''
ReplayMemory
  A cyclic buffer of bounded size that holds the transitions observed 
  recently. It also implements a .sample() method for selecting a random 
  batch of transitions for training.
'''

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

'''
Problem:
Unable to import 'gym'pylint(import-error)

Solution:
Changing the library path worked for me. Hitting Ctrl + Shift + P and 
typing python interpreter and choosing one the correct Python environment. 
'''