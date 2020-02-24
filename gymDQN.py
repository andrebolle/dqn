import gym
import numpy as np

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
EPISODES = 25000

# Converts state to a binned state
stateDim = [20, 20]
binSize = (env.observation_space.high - env.observation_space.low)/stateDim
print("binSize", binSize)

def binned(state):
    discrete_state = (state - env.observation_space.low)/binSize
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

Q = np.random.uniform(low=-2, high=0, size=(stateDim + [env.action_space.n]))

# The start state
s0 = binned(env.reset())
print("binned start state", s0)
print(Q[s0])
print("Argmax", np.argmax(Q[s0]))

done = False
while not done:
#for _ in range(2):
    # Pick the best action for this state
    a = np.argmax(Q[s0])

    # Get new state (s1) and reward (r)
    s1_raw, r, done, _ = env.step(a)
    s1 = binned(s1_raw)
    # print(r, s1, done)

    env.render()

    if not done:
        # max_future_Q = np.max(Q[s1])
        q0 = Q[s0 + (a,)]
        q1 = (1-α) * q0 + α * (r + γ * np.max(Q[s1]))

        # Update Q-table
        Q[s0+(a,)] = q1
    elif s1_raw[0] >= env.goal_position:
        Q[s0 + (a,)] = 0
    
    # Update new state
    s0 = s1

env.close()
print("done")
