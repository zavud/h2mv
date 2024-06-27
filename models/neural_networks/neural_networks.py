# this module contains neural networks models that can be used in conjunction with process based layers

# load the libraries
import torch

# fully-connected NN layer to reduce the dimensions of static input and predict static outputs
class FC_Static(torch.nn.Module):

    def __init__(self, num_static_input, num_static_out, hidden_size1 = 150, hidden_size2 = 12): # initialise the class

        # initialise the parent module from torch
        super().__init__()

        # fc1 takes the static inputs, and outputs higher-dimensional representations
        self.fc1 = torch.nn.Linear(in_features = num_static_input, out_features = hidden_size1)

        # fc2 takes the output from fc1 (100D) and reduces the dimension to 12
        self.fc2 = torch.nn.Linear(in_features = hidden_size1, out_features = hidden_size2)

        # fc3 takes the output from fc2 (12D) and predicts static outputs
        self.fc3 = torch.nn.Linear(in_features = hidden_size2, out_features = num_static_out)

    # define the forward run
    def forward(self, input_static):

        # get the higher-dimensional representation of the static input
        out_highD = self.fc1(input_static) # of shape (batch_size, hidden_size1)
        out_highD = torch.nn.functional.leaky_relu(out_highD) # activate the output using leaky relu

        # get the condensed representation of the static input
        out_condensed = self.fc2(out_highD) # of shape (batch_size, hidden_size2)
        out_condensed = torch.nn.functional.leaky_relu(out_condensed) # activate the output using leaky relu

        # get the static outputs (not activated)
        out_static = self.fc3(out_condensed) # of shape (batch_size, num_static_out)

        # return the 12D dimensional representation of the static input and static outputs
        return out_condensed, out_static

# LSTM layer that takes 12D dimensional representation of the static and meterological inputs and make intermediate predictions
class LSTM_Dynamic(torch.nn.Module):

    # initialise the class
    def __init__(self, num_features, num_temporal_outputs, hidden_size_lstm = 100):

        # initialise the parent module from torch
        super().__init__()

        # an LSTM cell that takes the 12D representation of static input and meteorological input
        self.lstm_cell = torch.nn.LSTMCell(input_size = num_features, hidden_size = hidden_size_lstm)

        # a fully-connected NN layer that takes the output of LSTM cell and maps it to temporal intermediate predictions
        self.fc = torch.nn.Linear(in_features = hidden_size_lstm, out_features = num_temporal_outputs)
    
    # define the forward run
    def forward(self, features_t, h_t, c_t):
        
        # get the hidden and cell states of LSTM cell
        h_t, c_t = self.lstm_cell(features_t, (h_t, c_t)) # of shape (batch_size, hidden_size_lstm)

        # get the temporal predictions by sending the hidden state through the fully connected layer
        # (not activated)
        output_t = self.fc(h_t) # of shape (batch_size, num_temporal_outputs)

        # return the temporal predictions
        return h_t, c_t, output_t