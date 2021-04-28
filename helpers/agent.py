# import necessary pacakges
from helpers.nn.network_model import DuelingNetwork
from helpers.priority_replay import Memory
import torch.optim as optim
import numpy as np
import torch 
import sys
# GPU INITIAILIZATION ATTEMPT
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """ 
        Initializes the type of update ware performing
            1 - Q-learning
            2 - expected sarsa
    """
    def __init__(self, agent_info, option=1):
        # initialize type
        self.type_init(option)
        # initialize agent
        self.agent_init(agent_info)
    
    # initialize update type. Q_update, Expected_Sarsa
    def type_init(self, option):
        if option==1:
            self.name = "Q_update"
        elif option == 2:
            self.name = "Expected_sarsa"

    def agent_init(self, agent_info):
        """
            Setup for the agent called when the experiment first starts
            Args:
                agent_info containg {
                    network_config: (dictionary) [state_size, action_size, array_fc_units]
                    optimizer_config: (dictionary) learning_rate
                    replay_buffer_size: (integer)
                    batch_size: (integer)
                    seed: seed for generating
                    num_replay_updates_per_step: float
                    gamma: float
                    greedy:(dictionary) [explore_start, explort_stop, decay_rate]
                    softmax_tau: (float) determines how greedy our softmax policy is
                }
        """
        # INITIALIZE THE REPLAY STORAGE
        self.replay_buffer = Memory(agent_info["replay_buffer_size"], agent_info["seed"])
        self.buffer_size   = agent_info["replay_buffer_size"]

        # INITIALIZE THE NETWORK - LOCAL AND TARGET
        self.network = DuelingNetwork(agent_info["network_config"]["state_size"], agent_info["network_config"]["action_size"],
                                      agent_info["seed"], agent_info["network_config"]["array_fc_units"]).to(device)
        self.target  = DuelingNetwork(agent_info["network_config"]["state_size"], agent_info["network_config"]["action_size"],
                                      agent_info["seed"], agent_info["network_config"]["array_fc_units"]).to(device)
        # NETWORK PARAMETERS
        self.optimizer     = optim.Adam(self.network.parameters(), lr = agent_info["optimizer_config"]["learning_rate"])
        self.num_actions   = agent_info["network_config"]["action_size"]
        self.num_replay    = agent_info["num_replay_updates_per_step"]
        self.discount      = agent_info["gamma"]
        self.batch_size    = agent_info["batch_size"]

        # RANDOM SEED INITIALIZER 
        self.rand_generator = np.random.RandomState(agent_info["seed"])

        # GREEDY EXPLORATION PARAMETER
        self.softmax_tau   = agent_info["softmax_tau"]
        self.explore_start = agent_info["greedy"]["explore_start"]
        self.explore_stop  = agent_info["greedy"]["explore_stop"]
        self.decay_rate    = agent_info["greedy"]["decay_rate"]

        # EPISODE TRACKING PARAMETER
        self.last_state    = None
        self.last_action   = None
        self.sum_rewards   = 0
        self.total_steps   = 0
        self.episode_steps = 0
        self.weight_update = 0

    def prepopulate(self, brain_name, env):
        """
            First thing called after environment has been setup
            To aviod the empty memory problem we randomly pre populate the memory. This is done
            by taking random actions and storing them as experiences
            Args:
                brain_name: (string) name of agent we are using
                env: (object) Environment we are operating in
        """
        # flag for when to reset the environment [when we hit a terminal state]
        reset_check, last_state = False, None
        for idx in range(self.buffer_size):
            # if idx is the first step get state or we have to reset
            if idx == 0 or reset_check:
                # change reset check back to false
                reset_check = False
                # resent environment and extract current state
                env_info   = env.reset(train_mode=True)[brain_name]
                last_state = env_info.vector_observations[0]

            # take random actions
            action = int(self.rand_generator.choice(range(self.num_actions)))

            # take the action, recod reward, and terminal status
            env_info = env.step(action)[brain_name]
            reward   = env_info.rewards[0]
            done     = env_info.local_done[0]

            # checking status
            if done:
                # set reset flag
                reset_check = True
                state = np.zeros(last_state.shape)
                # store in replay
                self.replay_buffer.store(last_state, action, reward, state, done)
            else:
                # get next state from the environment
                state = env_info.vector_observations[0]
                # store in replay
                self.replay_buffer.store(last_state, action, reward, state, done)
                # update state
                last_state = state

    # action selection for expected sarsa
    def softmax(self, action_values):
        """
            Args:
                action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                               The action-values computed by an action-value network.              
                self.softmax_tau (float): The temperature parameter scalar.
            Returns:
                A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
                the actions representing the policy.
        """
        # divide action values by tau
        preferences = action_values / self.softmax_tau
        # compute max
        max_preference = np.max(preferences, axis=1)
        # ensure max_preference has shape [batch, 1]
        reshaped_max_preference = max_preference.reshape((-1, 1))

        # compute numerator
        exp_preferences = np.exp(preferences - reshaped_max_preference)
        # denominator
        sum_of_experiences = np.sum(exp_preferences, axis=1)
        # reshap sum_of_expeirencs
        reshaped_sum_of_experiences = sum_of_experiences.reshape((-1, 1))
        # calculate probability
        action_prob = exp_preferences/reshaped_sum_of_experiences
        # to remove singleton dimensions
        action_prob = action_prob.squeeze()
        return action_prob

    def greedy_selection(self, action_values, decay_step):
        """
            Predicts the next action selected by the agent
            Args:
                action_values: (Numpy array): A 2D array of shape (batch_size, num_actions). 
                               The action-values computed by an action-value network.
                decay_step: (int)
            Returns:
                selected actions
        """
        #print(action_values)
        # randomly generate paremter for expolitation, exploration
        exp_exp_tradeoff = self.rand_generator.rand()
        # epsilon greedy calculation
        explore_prob = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * decay_step)
        # epsilon greedy selection check
        if (explore_prob > exp_exp_tradeoff):
            # randomly select and type cast as int
            action = int(self.rand_generator.choice(np.arange(self.num_actions)))
        else:
            action = int(np.argmax(action_values))
        # return list of selected actions per row in batch
        return action 

    def policy(self, state):
        """
            Returns selected action in the given state
            Args:
                state: (array of floats) represents current state
                self.replay_steps: total number of steps from begining ot training
                name: type of update we are doing, expected_sara, Q-learning
            Return:
                action
        """
        # calculate the q_value for current state
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # set network to evaluation mode
        self.network.eval()
        # disable gradient calculation because we are inference
        with torch.no_grad():
            # run forward pass
            action_values = self.network(state)
        # set network back to train mode
        self.network.train()
        # convert to numpy array
        action_values = action_values.cpu().data.numpy()
        # if we are using greedy method
        if self.name == "Q_update":
            return self.greedy_selection(action_values, self.total_steps)
        # if using expected sarsa
        elif self.name == "Expected_sarsa":
            probs_state = self.softmax(action_values)
            action = self.rand_generator.choice(self.num_actions, p=probs_state.squeeze())

            return action

    def agent_start(self, state):
        """
            The first method called when the experiment starts, called after
            the environment starts.
            Args:
                state (Numpy array): the state from the
                    environment's evn_start function.
            Returns:
                The first action the agent takes.
        """
        self.total_steps += 1
        self.episode_steps += 1
        self.sum_rewards = 0
        self.last_state = state
        self.last_action = self.policy(self.last_state)
        

    def agent_step(self, reward, state, done):
        """
            A step taken by the agent.
            Args:
                reward (float): the reward received for taking the last action taken
                state (Numpy array): the state from the
                    environment's step based, where the agent ended up after the
                    last step
                done: flag for if we at a terminal state or not
            Returns:
                The action the agent is taking.
        """
        self.total_steps += 1
        self.sum_rewards += reward
        # get action in current state
        action = self.policy(self.last_state)
        # add to replay buffer
        self.replay_buffer.store(self.last_state, self.last_action, reward, state, done)

        # incremental mini updates
        self.weight_update_control()

        # number of times we use the replay buffer
        for _ in range(self.num_replay):               
            # optimize the network
            self.optimize_network()

        # update last state and last action
        self.last_state = state
        self.last_action = action
    

    def agent_end(self, reward, done):
        """
            A step taken by the agent.
            Args:
                reward (float): the reward received for taking the last action taken
                self.total_step (float): current episode step
            Returns:
                The action the agent is taking.
        """
        self.sum_rewards += reward
        # convert state to numpy array
        state = np.zeros_like(self.last_state)
        # add to replay buffer
        self.replay_buffer.store(self.last_state, self.last_action, reward, state, done)
        # increment the replay steps
        self.total_steps += 1

        # incremental mini updates
        self.weight_update_control()
        # perform replay steps
        for _ in range(self.num_replay):               
            #optimize the network
            self.optimize_network()

        

    def optimize_network(self):
        """
        Updating the weights of the local network. Using the fixed q_target scheme
        Args:
            experiences (Numpy array): The batch of experiences including the states, actions, 
                                       rewards, terminals, and next_states.
            self.discount (float): The discount factor.
            self.network (ActionValueNetwork): The latest state of the network that is getting replay updates.
            self.target (ActionValueNetwork): The fixed network used for computing the targets, 
                                            and particularly, the action-values at the next-states.
        """
        # get sample from the buffer
        tree_idxs, experiences, ISweights = self.replay_buffer.sample(self.batch_size)
        # returns numpy arrays
        states, actions, rewards, next_states, dones = self.replay_buffer.unwrap_experiences(experiences)
        """
        Replay_buffer outputs explained:
            states (Numpy array): The batch of states with the shape (batch_size, state_dim).
            next_states (Numpy array): The batch of next states with the shape (batch_size, state_dim).
            actions (Numpy array): The batch of actions with the shape (batch_size,).
            rewards (Numpy array): The batch of rewards with the shape (batch_size,).
            discount (float): The discount factor.
            dones (Numpy array): The batch of terminals with the shape (batch_size,).
            network (DuelingNetwork): The latest state of the network that is getting replay updates.
            target_q (DuelingNetwork): The fixed network used for computing the targets, 
                                            and particularly, the action-values at the next-states.
            ISweights (float): importance sampling weights for mean square error loss
            tree_idxs (int): marks locations where samples were taken from useful for priority update layer
        """   
        # convert into torch tensor
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        ISweights = torch.from_numpy(ISweights).float().to(device)
        """
            Implementing the Double DQN logic to calculate the q(s',a') targets:
                - use the dqn network to select action to take and next state(a')
                - use target network to calculate the q_val of Q(s', a')
        """
        # target storage
        q_target = None

        # forward pass on local network and target network for next state to get q_values
        q_local_next = self.network(next_states).detach()
        q_targets_next = self.target(next_states).detach()
        # check which update type
        if self.name == "Q_update":
            # extract actions
            max_actions = q_local_next.max(1)[1].unsqueeze(1)
            # gather from target along the column axis using the maximum actions from local network
            q_targets = torch.gather(q_targets_next, 1, max_actions)

        # if using expected sarsa
        elif self.name == "Expected_sarsa":
            # convert to numpy
            q_target_numpy = q_targets_next.cpu().data.numpy()
            # get action probability for all batch states
            probs_list = np.array(self.softmax(q_target_numpy))           
            # calculated expected sarsa target
            weighted_max = probs_list * q_target_numpy
            # calculate action values
            q_targets = np.sum(weighted_max, axis=1).reshape((-1, 1))
            # convert to numberpy
            q_targets = torch.from_numpy(q_targets).float().to(device)
            
        # calculate target
        q_targets = rewards + self.discount * q_targets * (1 - dones)

        # we have the target and expected 
        q_expected = self.network(states).gather(1, actions)

        # priority update and loss calculation
        loss, abs_error = self.loss_calculations(q_targets, q_expected, ISweights)

        # update the sum trees
        self.replay_buffer.batch_update(tree_idxs, abs_error.squeeze())

        # minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()     

    def loss_calculations(self, q_target, q_expected, ISweights):
        """
            Calculates the weighted mean square error, and absolute difference between the target and expected states. 
            This is for updating the sum tree - the higher the difference the more likely it will be sampled as there 
            is a higher opportunity of learning from such regions

        """
        diff = q_target - q_expected
        # calculate absolute error: conver to numpy
        abs_error = torch.abs(diff).cpu().detach().numpy()
        # calculate weighted mean square error
        loss = torch.mean(ISweights * (diff**2))

        # return the loss and absolute difference
        return loss, abs_error

    # controls how the weights are updated
    def weight_update_control(self):
        if self.episode_steps % 100 == 0:
            self.soft_update(self.network, self.target, tau=1.0)
        else:
            # update weights
            self.soft_update(self.network, self.target)


    def soft_update(self, local_model, target_model, tau=0.001):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()

        # Epsilon-greedy action selection
        if self.rand_generator.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return int(self.rand_generator.choice(np.arange(self.num_actions)))