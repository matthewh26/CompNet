
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import seaborn as sns


data = pd.read_csv('data/Lol_matchs.csv')
data = data.iloc[:,1:302]
X_data = data.iloc[:,:300].to_numpy()
y_data = data.iloc[:,300].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


class CompData(Dataset):
    def __init__(self, X_train, y_train) -> None:
        super().__init__()
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len


comp_data = CompData(X_train= X_train, y_train= y_train)
CompLoader = DataLoader(comp_data, batch_size = 32)


class BinaryNN(nn.Module):
    def __init__(self, NUM_FEATURES, HIDDEN_FEATURES):
        super().__init__()
        self.lin1 = nn.Linear(NUM_FEATURES, 100)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(100, HIDDEN_FEATURES)
        self.lin3 = nn.Linear(HIDDEN_FEATURES, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.sigmoid(x)
        return x
    

NUM_FEATURES = comp_data.X.shape[1]
HIDDEN_FEATURES = 20

if __name__ == '__main__':

    model = BinaryNN(NUM_FEATURES=NUM_FEATURES,HIDDEN_FEATURES=HIDDEN_FEATURES)


    lr = 0.01
    optimizer=torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    EPOCHS = 100
    losses = []
    for epoch in range(EPOCHS):
        mean_losses = []
        for batch, (x, y) in enumerate(CompLoader):
        
            # initialize gradients
            optimizer.zero_grad()

            # forward pass
            y_pred = model(x)
            
            # calculate losses
            loss = criterion(y_pred, y.reshape([-1,1]).float())
            
            #mean_losses.append(float(loss.data.numpy()))
            
            # calculate gradients
            loss.backward()

            # update parameters 
            optimizer.step()

        print('epoch: ' + str(epoch))
        #losses.append(sum(mean_losses)/comp_data.X.shape[0])
        losses.append(float(loss.data.detach().numpy()))

    print('training complete')


    X_test_torch = torch.from_numpy(X_test)
    with torch.no_grad():
        y_test_pred = model(X_test_torch).round()

    print('test accuracy: ')
    print(accuracy_score(y_test_pred,y_test))



    X_train_torch = torch.from_numpy(X_train)
    with torch.no_grad():
        y_train_pred = model(X_train_torch).round()

    print('train accuracy: ')
    print(accuracy_score(y_train_pred,y_train))



    print(y_test_pred.sum())
    print(y_test_pred.shape)

    # naive accuracy
    naive = np.ones(y_test.shape[0])
    naive_acc = sum((naive+y_test)-1)/naive.shape[0]
    print(naive_acc)


    sns.lineplot(x= range(len(losses)), y = losses)
    
    torch.save(model.state_dict(), 'model_state_dict.pth')

