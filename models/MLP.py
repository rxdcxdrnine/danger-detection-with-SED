import torch


def fc_layer(in_plane, out_plane, dropout_rate):
    modules = [torch.nn.Linear(in_plane, out_plane),
               # torch.nn.BatchNorm1d(out_plane),
               torch.nn.ReLU(),
               torch.nn.Dropout(dropout_rate)]
    
    torch.nn.init.xavier_uniform_(modules[0].weight)
    
    return torch.nn.Sequential(*modules)


class FCNet3(torch.nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(FCNet3, self).__init__()
        
        self.fc1 = fc_layer(256, 256, dropout_rate=dropout_rate)
        self.fc2 = fc_layer(256, 256, dropout_rate=dropout_rate)
        self.fc3 = fc_layer(256, 256, dropout_rate=dropout_rate)
        
    def forward(self, X):
    
        out = self.fc1(X)
        out = self.fc2(out)
        out = self.fc3(out)
                
        return out


class FCNet5(torch.nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(FCNet5, self).__init__()
        
        self.fc1 = fc_layer(256, 256, dropout_rate=dropout_rate)
        self.fc2 = fc_layer(256, 256, dropout_rate=dropout_rate)
        self.fc3 = fc_layer(256, 256, dropout_rate=dropout_rate)
        self.fc4 = fc_layer(256, 256, dropout_rate=dropout_rate)
        self.fc5 = fc_layer(256, 256, dropout_rate=dropout_rate)
        
        
    def forward(self, X):
    
        out = self.fc1(X)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        
        return out