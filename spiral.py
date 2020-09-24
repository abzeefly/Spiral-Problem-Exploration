# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        input_nodes = 2
        output_nodes = 1

        self.layer_one = nn.Linear(input_nodes, num_hid)
        self.tanh = nn.Tanh()
        self.layer_two = nn.Linear(num_hid, output_nodes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        def to_polar(i):
            output = i.numpy()
            for t in output:
                t[0],t[1] = math.sqrt((t[0]**2 + t[1]**2)) ,  math.atan2(t[1],t[0])
            output = torch.from_numpy(output)
            return output

        output = to_polar(input)
        output1 = self.layer_one(output)
        output2 = self.tanh(output1)
        output3 = self.layer_two(output2)
        output4 = self.sigmoid(output3)
        self.hidden_layers = [output2]
        return output4

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        input_nodes = 2
        output_nodes = 1
        self.layer_one = nn.Linear(input_nodes, num_hid)
        self.tanh = nn.Tanh()
        self.layer_two = nn.Linear(num_hid, num_hid)
        self.output_layer = nn.Linear(num_hid, output_nodes)
        self.sigmoid = nn.Sigmoid()
        ##init 0.25
    def forward(self, input):
        # CHANGE CODE HERE
        output = self.layer_one(input)
        output1 = self.tanh(output)
        output2 = self.layer_two(output1)
        output3 = self.tanh(output2)
        output4 = self.output_layer(output3)
        output5 = self.sigmoid(output4)
        self.hidden_layers = [output1, output3]
        return output5

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        # INSERT CODE HERE
        input_nodes = 2
        output_nodes = 1
        hid1 = num_hid 
        hid2 = hid1 * 2

        self.layer_one = nn.Linear(input_nodes, hid1)
        self.tanh = nn.Tanh()
        self.layer_two = nn.Linear(hid1, hid2)
        self.output_layer = nn.Linear(hid2, output_nodes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        # CHANGE CODE HERE
        output = self.layer_one(input)
        output1 = self.tanh(output)
        output2 = self.layer_two(output1)
        output3 = self.tanh(output2)
        output4 = self.output_layer(output3)
        output5 = self.sigmoid(output4)
        self.hidden_layers = [output1, output2]
        return output5
        
        ## hid1 = 9
        #hid2  = 18
        #init = 0.25

def graph_hidden(net, layer, node):
    # INSERT CODE HERE
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval() 
        output = net(grid)
        # print(output.size())       # toggle batch norm, dropout

        output = net.hidden_layers[layer - 1][:, node]
        # print(output.size())       # toggle batch norm, dropout

        
        net.train() # toggle batch norm, dropout back again

        pred = (output >= 0.0).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
