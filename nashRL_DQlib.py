import torch
import torch.nn as nn


# Define object for estimated elements via NN
# ***NOTE*** All elements are tensors
class FittedValues(object):
    # Initialized via a single numpy vector
    def __init__(self, value_vector, v_value, num_players):
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

# Defines basic network parameters and functions
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, nump):
        super(DQN, self).__init__()
        self.num_players = nump
        # Define basic fully connected network for parameters in Advantage function
        self.main = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 60),
            nn.ReLU(),
            nn.Linear(60, 160),
            nn.ReLU(),
            nn.Linear(160, 60),
            nn.ReLU(),
            nn.Linear(60, output_dim)
        )

        # Define basic fully connected network for estimating nash value of each state
        self.main_V = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, num_players)
        )

    # Since only single network, forward prop is simple evaluation of network
    def forward(self, input):
        return self.main(input), self.main_V(input)


class NashNN():
    def __init__(self, input_dim, output_dim, nump):
        self.num_players = nump

        self.main_net = DQN(input_dim, output_dim, num_players)
        self.target_net = copy.deepcopy(self.main_net)
        self.num_sim = 10000
        # Define optimizer used (SGD, etc)
        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),
                                       lr=0.001)
        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        self.counter = 0

        # Predicts resultant values, input a State object, outputs a FittedValues object

    def predict(self, input):
        a, b = self.main_net.forward(self.stateTransform(input))
        return self.tensorTransform(a, b)

    def predict_targetNet(self, input):
        a, b = self.target_net.forward(self.stateTransform(input))
        return self.tensorTransform(a, b)

    # Transforms state object into tensor
    def stateTransform(self, s):
        return torch.tensor(s.getNormalizedState()).float()

    # Transforms output tensor into FittedValues Object
    def tensorTransform(self, output1, output2):
        return FittedValues(output1, output2, self.num_players)

    # takes a tuple of transitions and outputs loss
    def compute_Loss(self, state_tuple):
        currentState, action, nextState, reward = state_tuple[0], torch.tensor(state_tuple[1]).float(), state_tuple[2], \
                                                  state_tuple[3]
        A = lambda u, uNeg, mu, muNeg, a, v, c1, c2, c3: v - 0.5 * c1 * (u - mu) ** 2 + c2 * (u - a) * torch.sum(
            uNeg - muNeg) + c3 * (uNeg - muNeg) ** 2
        nextVal = self.predict(nextState).V
        curVal = self.predict(currentState)
        loss = []
        for i in range(0, self.num_players):
            r = lambda T: torch.cat([T[0:i], T[i + 1:]])
            loss.append(
                nextVal[i] + reward[i] - A(action[i], r(action), curVal.mu[i], r(curVal.mu), curVal.a[i], curVal.V[i],
                                           curVal.P1, curVal.P2, curVal.P3))

        return torch.sum(torch.stack(loss) ** 2)

    def updateLearningRate(self):
        self.counter += 1
        self.optimizer = optim.RMSprop(list(self.main_net.main.parameters()) + list(self.main_net.main_V.parameters()),
                                       lr=0.001 - (0.001 - 0.0005) * self.counter / self.num_sim)


#    #ignore these functions for now... was trying to do batch predictions
#    def predict_batch(self, input):
#        return self.tensorsTransform(self.main_net(self.statesTransform(input),batch_size = 3))
#
#    # Transforms state object into tensor
#    def statesTransform(self, s):
#        print(np.array([st.getNormalizedState() for st in s]))
#        return Variable(torch.from_numpy(np.array([st.getNormalizedState() for st in s])).float()).view(1, -1)
#
#    # Transforms output tensor into FittedValues Object
#    def tensorsTransform(self, output):
#        return np.apply_along_axis(FittedValues(),1,output.data.numpy())
