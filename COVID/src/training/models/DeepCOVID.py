import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.float
lr = 0.001

# code from: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

def buildNetwork(layers, norm_layer=False, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        lm = nn.Linear(layers[i-1], layers[i])
        torch.nn.init.xavier_normal_(lm.weight) 
        if i==1 and norm_layer: # only at first layer
            net.append(nn.BatchNorm1d(num_features=layers[i-1],affine=True))
        net.append(lm)
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="leakyReLU":
            net.append(nn.LeakyReLU())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

class DeepCOVID(nn.Module):
    def __init__(self, input_dim, layers, norm_layer=False):
        super(DeepCOVID, self).__init__()
        self.ffn = buildNetwork([input_dim] + layers + [1], norm_layer=norm_layer).to(device)
        # torch.nn.init.xavier_normal_(self.ffn.data)
        # self.ffn.apply(torch.nn.init.xavier_normal_)
        # self.ffn.append(nn.layers.leakyReLU.Sigmoid())

    def predict(self,X):
        X = torch.tensor(X, dtype=dtype).to(device)
        pred = self.ffn(X)
        # convert to numpy
        pred = pred.cpu().detach().numpy()
        return pred 
    
    def fit(self,X,y):
        X = torch.tensor(X, dtype=dtype).to(device)
        y = torch.tensor(y, dtype=dtype).to(device)
        # import time
        # start=time.time()
        self.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        es = EarlyStopping(patience=10, min_delta=0.1)
        for epoch in range(1000):
            # print('X',X)
            pred = self.ffn(X)
            # print('pred',pred)
            loss = F.mse_loss(pred, y,reduction='mean')
            optimizer.zero_grad()
            loss.backward()
            # print('loss',(loss**0.5).item())
            optimizer.step()
            if es.step(loss):
                # print('====BROKE===')
                break  # early stop criterion is met, we can stop now
        # end=time.time()
        self.eval()
        # print('train time: ',end-start)
        # time.sleep(5)
        # print('pred,y',pred,y)
        return (loss**0.5).cpu().detach().numpy().item()