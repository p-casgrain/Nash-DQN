import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nashRL_netlib import *


# Define object for estimated elements via NN
# ***NOTE*** All elements are tensors
class FittedValues(object):
    def __init__(self, value_vector, v_value, num_players):
        """

        :param value_vector:
        :param v_value:
        :param num_players:
        """
        self.num_players = num_players  # number of players in game

        self.V = v_value  # nash value vector

        self.mu = value_vector[0:self.num_players]  # mean of each player
        value_vector = value_vector[self.num_players:]

        self.P1 = (value_vector[0]) ** 2  # P1 matrix for each player, exp transformation to ensure positive value
        value_vector = value_vector[1:]

        self.a = value_vector[0:self.num_players]  # a of each player
        value_vector = value_vector[self.num_players:]

        # p2 vector
        self.P2 = value_vector[0]

        self.P3 = value_vector[1]

class NashNN():
    def __init__(self, input_dim, output_dim, nump, t):
        self.num_players = nump
        self.T = t
        self.main_net = DQN(input_dim, output_dim, self.num_players)
        # self.target_net = copy.deepcopy(self.main_net)
        self.num_sim = 5000
        # Define optimizer used (SGD, etc)
        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),
                                       lr=0.01)
        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        self.counter = 0

        # Predicts resultant values, input a State object, outputs a FittedValues object

    def slice(self,X,i):
        "Return all items in array except for i^th item"
        return torch.cat([X[0:i], X[i + 1:]])

    def predict(self, input):
        a, b = self.main_net.forward(self.stateTransform(input))
        return self.tensorTransform(a, b)

    #    def predict_targetNet(self, input):
    #        a,b = self.target_net.forward(self.stateTransform(input))
    #        return self.tensorTransform(a,b)

    # Transforms state object into tensor
    def stateTransform(self, s):
        return torch.tensor(s.getNormalizedState()).float()

    # Transforms output tensor into FittedValues Object
    def tensorTransform(self, output1, output2):
        return FittedValues(output1, output2, self.num_players)

    # takes a tuple of transitions and outputs loss
    def compute_Loss(self, state_tuple):
        currentState, action, nextState, reward = state_tuple[0], torch.tensor(state_tuple[1]).float(), \
<<<<<<< HEAD
                                                  state_tuple[2], state_tuple[3]
=======
                                                          state_tuple[2], state_tuple[3]
>>>>>>> 57e3d5515f33013b5ecbf344fa5dbb8ab500ac50
        # Q = lambda u, uNeg, mu, muNeg, a, v, c1, c2, c3: v - 0.5*c1*(u-mu)**2 + c2*(u -a)*torch.sum(uNeg - muNeg) + c3*(uNeg - muNeg)**2
        nextVal = self.predict(nextState).V
        flag = 0

        # set next nash value to be 0 if last time step
        if nextState.t > self.T - 1:
            flag = 1

        curVal = self.predict(currentState)
        loss = []

        for i in range(0, self.num_players):
            r = lambda T: torch.cat([T[0:i], T[i + 1:]])
            A = lambda u, uNeg, mu, muNeg, a, c1, c2, c3: 0.5 * c1 * (u - mu) ** 2 + c2 * (u - a) * torch.sum(
                uNeg - muNeg) + c3 * (uNeg - muNeg) ** 2
            loss.append((1 - flag) * nextVal[i] + flag * nextState.q[i] * (nextState.p - 50 * nextState.q[i])
                        + reward[i]
                        + A(action[i], r(action), curVal.mu[i], r(curVal.mu), curVal.a[i], curVal.P1, curVal.P2,
                            curVal.P3)
                        - curVal.V[i])

        #        if all(isNash):
        #            for i in range(0,self.num_players):
        #                loss.append(nextVal[i] + reward[i] - curVal.V[i])
        #        else:
        #            #note that this assumes that at most one person did not take nash action
        #            for i in range(0,self.num_players):
        #                r = lambda T : torch.cat([T[0:i], T[i+1:]])
        #                if isNash[i]:
        #                    loss.append(nextVal[i] + reward[i] - curVal.V[i].detach() - curVal.P2*(action[i] -curVal.a[i])*torch.sum(r(action) - r(curVal.mu)) - curVal.P3*(torch.sum(r(action) - r(curVal.mu))**2))
        #                else:
        #                    loss.append(nextVal[i] + reward[i] - curVal.V[i].detach() + 0.5*curVal.P1*(action[i]-curVal.mu[i])**2)
        #                    #A(action[i],r(action),curVal.mu[i],r(curVal.mu),curVal.a[i],curVal.V[i].detach(),curVal.P1,curVal.P2,curVal.P3)

        return torch.sum(torch.stack(loss) ** 2)

    def updateLearningRate(self):
        self.counter += 1
        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),
                                       lr=0.01 - (0.01 - 0.003) * self.counter / self.num_sim)

class NashNN2:
    def __init__(self, num_players=2, control_size=1, state_size=2):

        # Store Relevant Variables
        self.num_players = num_players
        self.control_size = control_size
        self.state_size = state_size

        # Compute Output Size (V + mu + c1 + c2 + c3 + a)
        self.output_dim = 6

        # Compute Input Size : X = (t,S,Q1,...,QK)
        self.input_size = self.state_size + self.num_players*self.control_size

        # Initialize DQN
        self.NNmodel = DQN2(input_dim,output_dim)

        # # Initialize Permutation Invariant NN-Model. Takes arguments (S,t,Q)
        # self.NNmodel = \
        #     PermInvariantQNN(self.num_players * self.control_size,
        #                      2, 3 + self.num_players * 2, block_size=1, num_moments=5)

    def predict(self, input):
        """
        :param input: Tensor of State Vectors, assumed to be in order (t,p,q)
        :return:
        """
        self.NNmodel.forward()

