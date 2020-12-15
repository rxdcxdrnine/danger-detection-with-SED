import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphUNetPooling(nn.Module):
    """graph node pooling module.
    A matrix was excluded.
    see here: (Paper, Graph U-Nets)
                https://arxiv.org/pdf/1905.05178.pdf
    """
    def __init__(self, topk, num_feature):
        super(GraphUNetPooling, self).__init__()
        self.topk = topk
        self.num_feature = num_feature

        self.p = nn.Parameter(torch.zeros((num_feature, 1)))
        nn.init.xavier_uniform_(self.p.data, gain=1.414)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # calculate y, index
        y = torch.matmul(x, self.p) / torch.norm(self.p.data)

        # top k index
        topk = torch.topk(y, self.topk, dim=1)

        y = topk[0]
        y = self.sigmoid(y)

        idx = topk[1]
        indices = idx.repeat(1, 1, self.num_feature)

        # pool top k
        x = torch.gather(x, 1, indices)

        # calculate x'
        x = x * y

        return x



class GATBlock(nn.Module):
    """
    Simple GAT layer 
    see here - https://arxiv.org/abs/1710.10903
    see also - https://github.com/Diego999/pyGAT/blob/master/layers.py
    """

    def __init__(self, in_features, out_features, dropout, negative_slope):
        super(GATBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope)


    def forward(self, x):
        """
        param
            input - 2d tensor (num_node, num_feature_of_each_node)
        """
        batch, N, _ = x.size()
        
        h = torch.matmul(x, self.W)
        temp1 = h.repeat(1, 1, N).view(batch, N * N, -1)
        temp2 = h.repeat(1, N, 1)
        a_input = torch.cat([temp1, temp2], dim=2).view(batch, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention)
        h_prime = torch.matmul(attention, h)
        h_prime = self.leakyrelu(h_prime)

        return h_prime
    

def GATNet(params, n_block):
    blocks = []
    n_features = params
    
    ### Bottleneck
    ### method 1
    # while n_features != 1:
    #     blocks.append(GATBlock(in_features=n_features, out_features=int(n_features / 2), 
    #                            dropout=.5, negative_slope=.2))
    #     n_features = int(n_features / 2)
    
    ### method 2-1
    # for _ in range(n_block):
    #     blocks.append(GATBlock(in_features=n_features, out_features=int(n_features / 2), 
    #                            dropout=.5, negative_slope=.2))
    #     n_features = int(n_features / 2)
        
    # pool = nn.Linear(n_features, 1)
    # blocks.append(pool)
    
    ### method 2-2
    # padding, kernel_size, stride = 0, 4, 2
        
    # for _ in range(n_block):
    #     blocks.append(GATBlock(in_features=n_features, out_features=n_features,
    #                            dropout=.5, negative_slope=.2))
        
    #     blocks.append(torch.nn.MaxPool1d(padding=padding, kernel_size=kernel_size, stride=stride))
    #     n_features = int((n_features + 2 * padding - kernel_size) / stride + 1)
        
    #     print(n_features)
        
    # pool = nn.AdaptiveAvgPool1d(1)
    # blocks.append(pool)
    
    
    ## Box
    for _ in range(n_block):
        blocks.append(GATBlock(in_features=n_features, out_features=n_features,
                               dropout=.5, negative_slope=.2))

    return nn.Sequential(*blocks)