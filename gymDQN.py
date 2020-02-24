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

EPISODES = 3000
SHOW_EVERY = 500

def goal(s):
    return s[0] >= env.goal_position

# Converts state to a binned state
stateDim = [10, 10]
binSize = (env.observation_space.high - env.observation_space.low)/stateDim
print("binSize", binSize)

def binned(state):
    discrete_state = (state - env.observation_space.low)/binSize
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

Q = np.random.uniform(low=-2, high=0, size=(stateDim + [env.action_space.n]))

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
