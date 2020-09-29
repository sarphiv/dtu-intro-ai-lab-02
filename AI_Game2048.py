# Game 2048: Artificial intelligence

# Instructions:
#   Move up, down, left, or right to merge the tiles. The objective is to 
#   get a tile with the number 2048 (or higher)


from Game2048 import Game2048
import numpy as np

import random
import time
from joblib import Parallel, delayed


actions = ['left', 'right', 'up', 'down']
#Limit of how far to simulate each simulation
#WARN: Do not pseudo-disable with a high number as an optimization is being used
# that would cause it to potentially loop all the way up to that high number.
sim_depth = 16
#Limit of how many simulations to run on each possible action
sim_tries = 100 #5, 10, 50, 100

#Limit of how many game runs to sample
env_samples = 50



def check_valid_actions(original_board):
    def is_valid_action(action):
        #Create simulation of action
        sim = Game2048((original_board, 0))
        sim.step(action)

        #If new block came, action was valid
        return np.array(sim.board).sum() - np.array(original_board).sum() >= 2


    return [action for action in actions if is_valid_action(action)]


def simulate_action(state, action):
    sim = Game2048(state)

    #Take initial step
    _, _, done = sim.step(action)
    
    #Randomly take steps until dead or 16 steps have been taken. 
    # It does not make sense to simulate too far into the future,
    # as the board state is randomly generated.
    # It would be better to run more simulations in that case.
    i = 0
    while not done and i < sim_depth:
        #NOTE: This allows randomly simulating into wrong directions (without any effect).
        # Allowing this as it is faster than to check.
        # This could cause useless loops all the way up to "sim_depth" though
        next_action = random.choice(actions)
        #next_action = random.choice(check_valid_actions(sim.board))
        _, _, done = sim.step(next_action)
        i += 1


    #Return score
    return sim.score


#Track score samples
env_scores = np.zeros(env_samples)

#Get samples for game runs
for i in range(env_samples):
    #Create game
    env = Game2048()
    env.reset()

    #Play till death
    print(f"START: {i} --------------------------", flush=True)
    while True:
        #Output board state sometimes
        if random.randint(0, 100) > 99:
            print(env.score, flush=True)


        #Retrieve valid actions
        valid_actions = check_valid_actions(env.board)

        #If dead, end game run
        if len(valid_actions) == 0:
            break


        #Alive, figure out next step
        scores = np.zeros(len(valid_actions))
        
        with Parallel(n_jobs=-1) as parallel:
            for action in valid_actions:
                sim_scores = parallel(delayed(simulate_action)((env.board, env.score), action) for i in range(sim_tries))

                scores[valid_actions.index(action)] += np.array(sim_scores).sum()

        #Take next step
        action = valid_actions[np.argmax(scores)]
        env.step(action)


    #Save score and close game
    env_scores[i] = env.score
    env.close()


    print(f"END:   {i} --------------------------")
    print("Samples:")
    print(env_scores, flush=True)


#Print final samples
print(f"DONE:  {i} --------------------------")
print(f"sim_tries: {sim_tries}; sim_depth: {sim_depth}; samples: {env_samples}")
print("Samples:")
print(env_scores)
print("Done", flush=True)

