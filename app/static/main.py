import winrates as winrates
import pandas as pd
import numpy as np

#import data
df = pd.read_csv('../data/match_data.csv')
df = df.replace(r'\N', 0)
champs_df = pd.read_csv('../data/champs.csv')
champs_dict = dict(zip(champs_df['name'],champs_df['id']))

print('find out what champ best fits your comp!')

#user's role
role = input('input your role: (top, jungle, mid, duo_carry, duo_support)').upper()

#initialise lists
teams = []
roles = []
champs = []

#build up lists of information about other champs in the game
while True:
    teams.append(input('ally or enemy?'))
    roles.append(input('input role: (top, jungle, mid, duo_carry, duo_support)').upper())
    champs.append(input('input champion selection'))
    cond = input('input another player? (y/n)')
    if cond == 'n':
        break

#change the notation to be correct (matching the column names in the data)
teams = ['blue' if x=='ally' else 'red' for x in teams]
champs = [champs_dict[x] for x in champs]

winrate_dicts = []
avg_winrates = dict()

#get winrates for all other champs in game
for i in range(len(champs)):
    wr = winrates.get_winrates(df,role,roles[i],teams[i],champs[i])
    print(wr)
    winrate_dicts.append(wr)

#find averages of all these winrates
for id in champs_df['id']:
    winrate_list = []
    for dictionary in winrate_dicts:
        if id in dict:
            winrate_list.append(dict[id])
    if winrate_list:
        avg_winrates[id] = np.mean(winrate_list)

#order largest to smallest winrates
avg_winrates = dict(sorted(avg_winrates.items(), key=lambda item: item[1]))


print(avg_winrates)
print(max(avg_winrates, key=avg_winrates.get))