import numpy as np


import gym

import sys
import os

print("Hi")
print(sys.getdefaultencoding())

# Number of bins per state variable
bins = [10, 10]

# ------------------------------- Start of Plot Animation --------------------
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = axes3d.Axes3D(fig)

xs = np.arange(0, bins[0], 1)
ys = np.arange(0, bins[1], 1)

print(xs,ys)
xx, yy = np.meshgrid(xs, ys)

plt.plot(xx, yy, marker='.', color='k', linestyle='none')
ax.plot_wireframe(xx, yy, xx+yy, rstride=2, cstride=2)
# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
# h = plt.contourf(xs,ys,z)
plt.show()

sys.exit()

# ------------------------------- End of Plot Animation --------------------

env = gym.make("MountainCar-v0")
env.reset()

# α: The learning rate
# γ: The discount, how much we care about future rewards
# There is "Q" value per action possible per state.
# Q: The probability of this action in this state
# Q(s,a): A lookup table. [s ∈ States, a ∈ Actions]

# Q(s,a) = (1 - α) * Q(s,a) + α * (r + γ * max(Q(s',a)))

# Q-Learning settings
α = 0.1 # Learning rate
γ = 0.95 # Discount

EPISODES = 1000
SHOW_EVERY = 100

def goal(s):
    return s[0] >= env.goal_position



binSize = (env.observation_space.high - env.observation_space.low)/bins
print("binSize", binSize)

def binned(state):
    discrete_state = (state - env.observation_space.low)/binSize
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

Q = np.random.uniform(low=-2, high=0, size=(bins + [env.action_space.n]))

for episode in range(EPISODES):


    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False
    
    # ------------------- An Episode -------------------------

    # The start state
    s0 = binned(env.reset())

    done = False
    while not done:
        # Pick the best action for this state
        a = np.argmax(Q[s0])

        # Get new state (s1) and reward (r)
        s1_raw, r, done, _ = env.step(a)
        s1 = binned(s1_raw)

        # Render once every now and again
        if render:
            env.render()

        if not done:
            Q[s0+(a,)] = (1-α) * Q[s0 + (a,)] + α * (r + γ * np.max(Q[s1]))
            # Q[s0+(a,)] = Q[s0 + (a,)] + α * (r + γ * np.max(Q[s1]))
        elif goal(s1_raw):
            Q[s0 + (a,)] = 0
        
        # Update new state
        s0 = s1
    # -----------------------------------------------------

env.close()
print("done")
