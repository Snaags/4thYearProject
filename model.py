import torch.nn as nn
import torch.nn.functional as F 
import torch



"""
# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        self.hidden.cuda()
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1),self.hidden)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

"""








class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,seq):
        super(LSTMModel, self).__init__()
        
        self.seq_len = seq
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=False,dropout = 0)
        #self.act = nn.ReLU()
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self):
        # Initialize hidden state with zeros
        self.h0 = torch.zeros(self.layer_dim, 1, self.hidden_dim).requires_grad_().cuda()

        # Initialize cell state
        self.c0 = torch.zeros(self.layer_dim, 1, self.hidden_dim).requires_grad_().cuda()


    def forward(self, x):
        
        out, (self.h0, self.c0) = self.lstm(x.view(self.seq_len,1,-1), (self.h0.detach(), self.c0.detach()))
        out = self.fc(out[-1,-1,:])
        #out = self.fc(out) 

        return out.view(1)

"""
class LSTM(nn.Module):
    def __init__(self,inputDimensions, hiddenDimensions, layersNumber,outputDimensions):
        super(LSTM, self).__init__()
        
        self.X_d = inputDimensions
        self.h_d = hiddenDimensions
        self.L_num = layersNumber
        self.y_d = outputDimensions


        self.LSTM = nn.LSTM(input_size = self.X_d,hidden_size =  self.h_d,num_layers = self.L_num, batch_first = False)

    def forward(self, x, h0):

        output, h1 = self.LSTM(x, h0)

        return output, h1



class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        
        return predictions[-1]


"""