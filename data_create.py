import pandas as pd
import numpy as np

#necessary csv files, combine stats to one dataframe
participants = pd.read_csv('data/participants.csv')
stats1 = pd.read_csv('data/stats1.csv')
stats2 = pd.read_csv('data/stats2.csv')

stats_total = pd.concat([stats1, stats2])
stats = stats_total.drop(['win','item1','item2','item3','item4','item5','item6','trinket'],axis=1)

#initialise dataframe with matchids
games = np.arange(10,187588)
df = pd.DataFrame(games, columns = ['matchid'])


#blue side topmidjung
for position in ['JUNGLE','TOP','MID']:
    pos_df = participants[participants['position']==position]
    pos_df = pos_df[pos_df['player'] < 6]
    pos_df = pos_df.loc[:,['id','matchid','championid']]
    pos_df = pos_df.rename(columns = {'id': position + '_blue_id', 'championid': position + '_blue_champ'})
    pos_df = pos_df.apply(pd.to_numeric)
    df = pd.merge(df, pos_df,on='matchid',suffixes = ('_', '__'))  


#blue side botsup
for role in ['DUO_CARRY', 'DUO_SUPPORT']:
    pos_df = participants[participants['role']==role]
    pos_df = pos_df[pos_df['player'] < 6]
    pos_df = pos_df.loc[:,['id','matchid','championid']]
    pos_df = pos_df.rename(columns = {'id': role + '_blue_id', 'championid': role + '_blue_champ'})
    pos_df = pos_df.apply(pd.to_numeric)
    df = pd.merge(df, pos_df,on='matchid',suffixes = ('_', '__'))  

#red side topmidjung
for position in ['JUNGLE','TOP','MID']:
    pos_df = participants[participants['position']==position]
    pos_df = pos_df[pos_df['player'] > 5]
    pos_df = pos_df.loc[:,['id','matchid','championid']]
    pos_df = pos_df.rename(columns = {'id': position + '_red_id', 'championid': position + '_red_champ'})
    pos_df = pos_df.apply(pd.to_numeric)
    df = pd.merge(df, pos_df,on='matchid',suffixes = ('_', '__'))  

#red side botsup
for role in ['DUO_CARRY', 'DUO_SUPPORT']:
    pos_df = participants[participants['role']==role]
    pos_df = pos_df[pos_df['player'] > 5]
    pos_df = pos_df.loc[:,['id','matchid','championid']]
    pos_df = pos_df.rename(columns = {'id': role + '_red_id', 'championid': role + '_red_champ'})
    pos_df = pos_df.apply(pd.to_numeric)
    df = pd.merge(df, pos_df,on='matchid',suffixes = ('_', '__'))  



#add match stats for each role
columns = list(stats.drop('id',axis=1).columns)

for ROLE_side in ['JUNGLE_blue','TOP_blue','MID_blue','DUO_CARRY_blue','DUO_SUPPORT_blue','JUNGLE_red',
                  'TOP_red','MID_red','DUO_CARRY_red','DUO_SUPPORT_red']:
    df = pd.merge(df,stats,left_on=(ROLE_side + '_id'),right_on='id')
    df = df.drop('id',axis=1)
    new_names = [(i,ROLE_side + '_' + i) for i in columns]
    df.rename(columns = dict(new_names), inplace=True)


#add result (did blue/red side team win)
results_df = stats_total.iloc[:,:2]
df = pd.merge(df, results_df, left_on = 'JUNGLE_blue_id', right_on = 'id')
df = df.drop(['id'], axis=1)
df['win'] = df['win'].replace({1: 'blue', 0: 'red'})


#check
print(df.head())
print(df.shape)

#to csv (ready for model!)
df.to_csv('match_data.csv')
