import torch
from models.DcaseNet import *
from models.GAT import *
from models.MLP import *
from models.se_resnet1d import *


### GAT

class Network(torch.nn.Module):
    def __init__(self, args, device):
        super(Network, self).__init__()
        
        self.dcase = DcaseNet_v3()
        self.dcase.load_state_dict(torch.load(args["weight_directory"]))
        self.dcase = self.dcase.to(device)
        
        for param in self.dcase.parameters():
            param.requires_grad = False
        
        self.graph = GATNet(256, n_block=4).to(device)
        self.linear = torch.nn.Linear(256, 1).to(device)
                
    def forward(self, x):
        x = self.dcase(x, mode="SED")["SED"]
        x = self.graph(x)
        x = self.linear(x).squeeze(-1)
        
        x, _ = torch.max(x, dim=-1)
        x = torch.sigmoid(x)
        
        return x



### se_resnet

# class Network(nn.Module):
#     def __init__(self, args, device):
#         super(Network, self).__init__()
        
#         self.dcase = DcaseNet_v3()
#         self.dcase.load_state_dict(torch.load(args["weight_directory"]))
#         self.dcase = self.dcase.to(device)
        
#         if args["weight_freeze"]:
#             for param in self.dcase.parameters():
#                 param.requires_grad = False

#         self.se_resnet = se_resnet().to(device)
#         self.fc_out = nn.Linear(128, 1).to(device)

#         nn.init.xavier_uniform_(self.fc_out.weight)

#     def forward(self, x):
#         x = self.dcase(x, mode="SED")["SED"]
        
#         x = self.se_resnet(x)
#         x = self.fc_out(x).squeeze(-1)
#         x = torch.sigmoid(x)
        
#         return x


### MLP

# class Network(nn.Module):
#     def __init__(self, args, device):
#         super(Network, self).__init__()
        
#         self.dcase = DcaseNet_v3()
#         self.dcase.load_state_dict(torch.load(args["weight_directory"]))
#         self.dcase = self.dcase.to(device)
        
#         if args["weight_freeze"]:
#             for param in self.dcase.parameters():
#                 param.requires_grad = False

#         self.fc_net = FCNet5().to(device)    
#         self.fc_out = nn.Linear(256, 1).to(device)         

#         nn.init.xavier_uniform_(self.fc_out.weight)

#     def forward(self, x):
#         x = self.dcase(x, mode="SED")["SED"]
#         x = self.fc_net(x)
#         x = self.fc_out(x).squeeze(-1)
        
#         x, _ = torch.max(x, dim=-1)
#         x = torch.sigmoid(x)
        
#         return x