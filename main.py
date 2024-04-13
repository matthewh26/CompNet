import numpy as np
from win_predict import BinaryNN, Scaler
import torch
import pandas as pd

df = pd.read_csv('match_data.csv')
df = df.drop('win',axis=1)
df = df.replace(r'\N', 0)
df = df.apply(pd.to_numeric)
champs = pd.read_csv('data/champs.csv')

print('find out what champ best fits your comp!')


#recieve comp/role info
#current iteration ONLY works for allied jungler
role = input('input your role: (top, jungle, mid, duo_carry, duo_support)').upper()
team_mate = input('current allied jungler:')
team_mate = int(champs[champs['name']==team_mate]['id']) #turn team mate champ name to id

#initialise model, load trained state
model = BinaryNN()
model.load_state_dict(torch.load('model_state_dict.pth'))

#only data with team mate champ in game
ddf = df[df['JUNGLE_blue_champ']==team_mate] 

#only include champs with enough data
champs = ddf['TOP_blue_champ'].value_counts()
top_champs = champs[champs>10].index 

#aggregate by player's role mean values
average_df_blue = ddf.groupby(['TOP_blue_champ']).mean()
columns = [col for col in list(average_df_blue.columns) if 'TOP_blue' in col]
average_df_blue = average_df_blue[columns].iloc[:,1:]

#find the predicted win prob from the model for each champ in player's role
win_probs = []
poss_champs = []
for champ in top_champs:
    row = average_df_blue[average_df_blue.index == champ]
    row = row.to_numpy().astype('float32')
    if row.shape[0] != 0:
        row = Scaler.transform(row)
        with torch.no_grad():
            win_probs.append(model(torch.from_numpy(row)).item())
            poss_champs.append(champ)

#output champ id of top 10 recommended champs
idxs = np.argsort(win_probs)

for i in range(10):
    print(poss_champs[idxs[-i-1]])
    print('win prob: ' + str(win_probs[idxs[-i-1]]))



'''
lots of improvement to be made here: 
    - Only looks at blue side, can look at red side too by simply flipping the data and appending
    - Not sure why I dont pass all of the average_df_blue through the model at the same time? QOL change
    - Model currently uses average of all team's stats, this NEEDS to be changed to just the role of the user, e.g. just 
    TOP stats for a top lane user are used to calc win prob
    - Make it work for all teammates, not just allied jungler!
'''
