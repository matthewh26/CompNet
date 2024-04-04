# %% imports
import requests
import time
import pandas as pd
# %% initialise api key
api_key = ''
#%% create api url for request to get all chall players
api_url = 'https://euw1.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5'
api_url = api_url + '?api_key=' + api_key

# %% save api response
req = requests.get(api_url)
response = req.json()
challenger_summs = response['entries']

# %% create list of summoner IDs
chall = []
for i in range(len(challenger_summs)):
    chall.append(challenger_summs[i]['summonerId'])
# %% get puuids for each summoner
chall_puuids = []
for i in range(len(chall)):
    api_url = 'https://euw1.api.riotgames.com/lol/summoner/v4/summoners/' + chall[i] + '?api_key=' + api_key
    req = requests.get(api_url)
    if req.status_code != 200:
        print('bad api call')
        print(req.status_code)
    info = req.json()
    chall_puuids.append(info['puuid'])
    print(i)
    if (i+1)%90 == 0:
        time.sleep(120) #to avoid exceeding rate limit
    
#%% initialise a dataframe
cols = ['id','al_top','al_jgl','al_mid','al_bot','al_sup',
        'en_top','en_jgl','en_mid','en_bot','en_sup','win']
df = pd.DataFrame(columns=cols)

# %% for each puuid get last 100 ranked games and add 
    #game info to dataframe
counter=0
match_list = []
for puuid in chall_puuids:
    api_url = api_url = 'https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/' + puuid +  '/ids?queue=420&type=ranked&start=0&count=100&api_key=' + api_key
    req = requests.get(api_url)
    match_list.extend(list(req.json()))
    time.sleep(2)

    print(counter)
    counter += 1
match_list = set(match_list)


#%%
counter=0
for match in match_list:
        #get comp information and winning team for each match, add row to dataframe
    row = []
    row.append(match)
    print(match)
    api_url = 'https://europe.api.riotgames.com/lol/match/v5/matches/' + match + '/?api_key=' + api_key 
    req = requests.get(api_url)
    game_info = req.json()
    print(req.status_code)
    if req.status_code != 200:
        pass
    elif len(game_info['info']['participants']) == 10:
        for i in range(10):
            row.append(game_info['info']['participants'][i]['championName'])
        row.append(game_info['info']['teams'][0]['win'])
        df.loc[len(df)] = row

        print(counter)
        counter += 1
    time.sleep(2) #pause for rate limit
    
#%%
df.head()
# %%
df.to_csv('chall_matches.csv')
# %%
