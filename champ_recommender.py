import numpy as np
from compnet import BinaryNN
import torch
import pandas as pd

matches = pd.read_csv('data/Lol_matchs.csv')
columns = list(matches.columns)
del columns[0]
columns = [i.capitalize() for i in columns]


print('find out what champ best fits your comp! \nSimply input the champs on your team and the enemy team \nseparated by commas')

#initialize X to put through model and win probability array
x = np.zeros(300)
prob_win = np.zeros(150)

#recieve team info
team = input('input allied champs:')
enemy_team = input('input enemy champs:')

#create lists of champs on allied/enemy team 
team_list = team.split(sep=',')
team_list = [champ.strip() for champ in team_list]
team_list = [i.capitalize() for i in team_list]
enemy_list = enemy_team.split(sep=',')
enemy_list = [champ.strip() for champ in enemy_list]
enemy_list = [i.capitalize() for i in enemy_list]

#initialise model, load trained state
NUM_FEATURES = 300
HIDDEN_FEATURES = 20
model = BinaryNN(NUM_FEATURES=NUM_FEATURES,HIDDEN_FEATURES=HIDDEN_FEATURES)
model.load_state_dict(torch.load('model_state_dict.pth'))


#change x to represent current allied/enemy comps
for champ in team_list:
    x[columns.index(champ)] = 1
for champ in enemy_list:
    x[columns.index(champ)+150] = 1


#loop through all possible picks, generate win probability for each
for i in range(150):
    if x[i] != 1:
        x[i] = 1
        x_in = torch.from_numpy(x.astype('float32'))
        with torch.no_grad():
            y = model(x_in)
        prob_win[i] = y
        x[i] = 0


#determine which champ is the best pick
print('\n')
print('your recommended pick is...')
print(columns[np.argmax(prob_win)])
print('win probability: ')
print(prob_win[np.argmax(prob_win)])
        


#output pick recommendation to user
print(prob_win)




