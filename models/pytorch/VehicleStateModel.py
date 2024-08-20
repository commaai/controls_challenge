import torch.nn as nn

class VehicleStateModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=5):
        super(VehicleStateModel, self).__init__()
        
        self.model = nn.ModuleDict({
            'lstm1': nn.LSTMCell(input_size=input_size, hidden_size=hidden_size),
            'layer_norm1': nn.LayerNorm(hidden_size),
            'lstm2': nn.LSTMCell(input_size=input_size, hidden_size=hidden_size),
            'layer_norm2': nn.LayerNorm(hidden_size),
            'linear': nn.Linear(hidden_size, hidden_size),
            'relu': nn.ReLU()
        })
        
    def forward(self, x):
        lstm_out, _ = self.model.lstm1(x)
        lstm_out = self.model.layer_norm1(lstm_out)
        lstm_out, _ = self.model.lstm2(lstm_out)
        lstm_out = self.model.layer_norm2(lstm_out)
        x += self.model.relu(self.model.linear(lstm_out))
        last_output = x[:, -1, :]
        output = self.output_layer(last_output)
        return output.squeeze(-1)