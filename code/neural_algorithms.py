import torch
import torch.nn as nn
import torch.optim as optim


class DeepNN(nn.Module):
    def __init__(self, n_inputs=4, n_outputs=2):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, n_outputs)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation at the output (for regression)
        return x



class GRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=2, num_layers=3):
        super(GRU, self).__init__()

        # GRU Layer
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)

        # Fully connected layer to map from hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Ensure x is 3D
        if x.dim() == 2:  
            x = x.unsqueeze(1)

        # Initialize hidden state 
        h0 = torch.zeros(self.gru.num_layers, x.size(
            0), self.gru.hidden_size).to(x.device)

        # Forward propagate through GRU
        out, _ = self.gru(x, h0)

        # We are interested only in the output of the last time step
        out = out[:, -1, :]  # Get the output of the last time step

        # Pass the last hidden state through the fully connected layer
        out = self.fc(out)

        return out
