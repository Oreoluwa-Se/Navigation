# import neccesary components
from unityagents import UnityEnvironment
from helpers.agent import Agent
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import argparse
import torch

plt.ion()

# starts the environment
def environment_start():
    env = UnityEnvironment(file_name="Banana.exe")

    # get default brain
    brain_name = env.brain_names[0] # decides actions of agents
    brain = env.brains[brain_name]

    # print to screen
    env_info = env.reset(train_mode=True)[brain_name]

    # print number of actions
    action_size = brain.vector_action_space_size
    #print("Number of actions:", action_size) 

    # examine the state space
    state = env_info.vector_observations[0]
    
    #print("States look like:", state)
    state_size = len(state)
    #print("States have length:", state_size)

    return env, state_size, action_size, brain_name

def training_run(n_episodes, n_timesteps, agent, brain_name):
    """
        Run loop for all episodes
        Parameters:
            n_episodes: (int) number of training episodes
            n_timesteps: (int) number of timesteps per episodes
            agent: (object) initialized agent - see agent.py for details
            brain_name: (string) 
    """
    score_window = deque(maxlen=100)
    scores = []
    plot_flag = False
    max_average_found = 12
    max_episode = None
    end = ""

    # loop through episodes
    while agent.episode_steps < n_episodes:
        # reset the episode to start a new run, get start state, and other parameters
        env_info = env.reset(train_mode=True)[brain_name]
        start_state = env_info.vector_observations[0]
        # call the agent start method
        agent.agent_start(start_state)

        # run for maximum number of timesteps
        for _ in range(n_timesteps):
            # environment step using action
            env_info = env.step(agent.last_action)[brain_name]
            # extract reward, next state, and information if we are in the terminalstate
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            if done:
                # end procedure
                agent.agent_end(reward, done)
                # break the loop
                break
            else:
                # get the current state
                state = env_info.vector_observations[0]
                # call the step method
                agent.agent_step(reward, state, done)
                

        # track score per episode
        score_window.append(agent.sum_rewards)
        scores.append(agent.sum_rewards)
        # calculate average score
        average_score = np.mean(score_window)
        print("\rEpisode {} \tAverage score: {: .2f}".format(agent.episode_steps, average_score))

        if agent.episode_steps % 100 == 0:
            print("\rEpisode {} \tAverage score: {: .2f}".format(agent.episode_steps, average_score))

        if average_score >= 13:      # check if environment is solved
            plot_flag = True
            print('\nEnvironment solved in {: d} episodes!\tAverage Score: {: .2f}'.format(agent.episode_steps - 100, average_score))
            torch.save(agent.network.state_dict(), 'results/ddqn_13_{}.pth'.format(agent.name))
            end = 'ddqn_13_{}.pth'.format(agent.name)
            break

        if max_average_found < average_score:
            plot_flag = True
            max_average_found = average_score
            max_episode = agent.episode_steps 

    if max_average_found > 12:
        print('\nEnvironment best score in {: d} episodes!\t Max average Score: {: .2f}'.format(max_episode - 100, max_average_found))
        torch.save(agent.network.state_dict(), 'results/ddqn_max_{}.pth'.format(agent.name))
        end = 'ddqn_max_{}.pth'.format(agent.name)
   
    return scores, plot_flag, end

def run(agent, brain_name, env, end=None, path=None):
    # load agent
    if path == None:
        agent.network.load_state_dict(torch.load('results/{}'.format(end)))
    else:
        agent.network.load_state_dict(torch.load(path))

    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state)                      # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        
    print("Score: {}".format(score))

def plot(scores, end):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('results/{}.png'.format(end[:-4]), bbox_inches='tight')
    



if __name__ == "__main__":
    # enviroment call
    env, state_size, action_size, brain_name = environment_start()
    # Agent parameters
    agent_parameters = {
        'network_config': {
            'state_size': state_size,
            'array_fc_units': [32, 32],
            'action_size': action_size
        },
        'optimizer_config': {'learning_rate': 0.0005},
        'replay_buffer_size': 100000,
        'batch_size': 64,
        'num_replay_updates_per_step': 8,
        'gamma': 0.9,
        'greedy':{'explore_start':1.0,
                  'explore_stop':0.01,
                  'decay_rate':0.00005},
        'softmax_tau': 0.001,
        "seed": 0
    }
    
    # required arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--check", required=True,
        help="Training or Evaluating? [T - Training, E - Evaluating]")
    ap.add_argument("-dt", "--weights", help="path to where file should be looaded")
    args = vars(ap.parse_args())

    if args["check"].lower() == "t":
        # initialize agent. option can be set to [1 - Q_learning, 2- Expected Sarsa]
        agent = Agent(agent_info=agent_parameters, option=1)
        # pre populate the replay memory
        agent.prepopulate(brain_name, env)
        # train the
        scores, plot_flag, end = training_run(1000, 5000, agent, brain_name)
        # plot and save the scores
        if plot_flag:
            plot(scores, end)
            # run the agent
            run(agent, brain_name, env, end=end)
    elif args["check"].lower() == "e":
        # initialize agent. option can be set to [1 - Q_learning, 2- Expected Sarsa]
        agent = Agent(agent_info=agent_parameters, option=1)
        # run the agent
        run(agent, brain_name, env, path=args["weights"])



