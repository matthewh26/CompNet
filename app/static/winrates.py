import pandas as pd
import numpy as np

def get_winrates(data,role_player,role_other,team_other,champ_other):
    '''Get the winrates for every combination of your champ in your role
    and every other champ for another role.
    
    :param data: the dataset to calculate the winrate from
    :param role_player: string containing the role the user is playing
    :param role_other: string containing the role the teammate/enemy is playing
    :param team_other: string containing either blue or red, corresponding to either teammate/enemy
    :param champ_player: the champion id of the champion the user is playing
    :returns: a dictionary of all the champs and the corresponding winrates with/against the user's champ'''
    
    winrates = dict()
    player_col = role_player + '_blue_champ'
    other_col = role_other + '_' + team_other + '_champ'


    total_matches = data[data[other_col]==champ_other]
    counts = total_matches[player_col].value_counts()

    for champ in counts[counts>15].index:
        specific_matches = total_matches[total_matches[player_col]==champ]
        winrates[champ] = len(specific_matches[specific_matches['win']=='blue'])/len(specific_matches)
    
    return winrates

if __name__ == '__main__':

    data = pd.read_csv('../data/match_data.csv')
    winrates = get_winrates(data,'TOP','JUNGLE','blue',64)
    print(len(winrates))
    print(winrates)

