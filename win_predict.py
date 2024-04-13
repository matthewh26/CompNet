import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class MatchData(Dataset):
    '''Pytorch dataset class to put into dataloader
    
    Attributes:
        X: the X data.
        y: the y data.
        len: the length of the data.
    '''

    def __init__(self, X_train, y_train) -> None:
        #inits the Dataset with X,y data and length
        super().__init__()
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        #returns the data point at a given index
        return self.X[index], self.y[index]
    
    def __len__(self):
        #returns the length
        return self.len
    

class BinaryNN(nn.Module):
    '''Pytorch feed-forward neural network for binary classification.
    Current network configuration is:
        2 hidden layers
        100 hidden nodes
        0.1 dropout layer in between
        ReLu activation function
    
    Attributes are all layers of the network.
    '''

    def __init__(self, NUM_FEATURES):
        #inits dataset with input size
        super().__init__()
        self.lin1 = nn.Linear(NUM_FEATURES, 100)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self,x):
        #forward pass through the network
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        return x



if __name__ == '__main__':

    #load data and scale 
    data = pd.read_csv('data/stats1.csv')

    Scaler = StandardScaler()

    X_data = data.iloc[:,9:].to_numpy()
    X_data = Scaler.fit_transform(X_data)
    y_data = data['win'].to_numpy()

    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.1)

    #convert dtype for neural net
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    #create dataset instance and loader in torch dataloader
    match_data = MatchData(X_train= X_train, y_train= y_train)
    Loader = DataLoader(match_data, batch_size = 256)

    #create model instance
    model = BinaryNN(NUM_FEATURES = match_data.X.shape[1])

    #model parameters
    lr = 0.01
    optimizer=torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    #training loop
    EPOCHS = 50
    losses = []
    for epoch in range(EPOCHS):
        mean_losses = []
        for batch, (x, y) in enumerate(Loader):
        
            # initialize gradients
            optimizer.zero_grad()

            # forward pass
            y_pred = model(x)
            
            # calculate losses
            loss = criterion(y_pred, y.reshape([-1,1]).float())
            
            mean_losses.append(loss.item())
            
            # calculate gradients
            loss.backward()

            # update parameters 
            optimizer.step()

        print('epoch: ' + str(epoch))
        losses.append(np.mean(mean_losses))
        print('loss: ' + str(losses[epoch]))

    print('training complete')

    #test accuracy
    X_test_torch = torch.from_numpy(X_test)
    with torch.no_grad():
        y_test_pred = model(X_test_torch).round()

    print('test accuracy: ')
    print(accuracy_score(y_test_pred,y_test))


    #train accuracy
    X_train_torch = torch.from_numpy(X_train)
    with torch.no_grad():
        y_train_pred = model(X_train_torch).round()

    print('train accuracy: ')
    print(accuracy_score(y_train_pred,y_train))



    #print(y_test_pred.sum())
    #print(y_test_pred.shape)


    #naive accuracy 
    naive_acc = max(sum(y_test),(len(y_test)-sum(y_test)))/len(y_test)
    print('naive accuracy: ')
    print(naive_acc)

    #plot losses 
    sns.lineplot(x= range(len(losses)), y = losses)
    
    #save model state
    torch.save(model.state_dict(), 'model_state_dict.pth')

