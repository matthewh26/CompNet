from flask import Flask, render_template, request
from static import winrates
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/", methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/list/',methods = ['GET', 'POST'])
def list():

    df = pd.read_csv('../data/match_data.csv')
    df = df.replace(r'\N', 0)
    champs_df = pd.read_csv('../data/champs.csv')
    champs_dict = dict(zip(champs_df['name'],champs_df['id']))
    ids_dict = dict(zip(champs_df['id'],champs_df['name']))


    role = request.form.get("role")
    a_top = request.form.get("a_top")
    a_jgl = request.form.get("a_jgl")
    a_mid = request.form.get("a_mid")
    a_bot = request.form.get("a_bot")
    a_sup = request.form.get("a_sup")
    ally_team = [a_top,a_jgl,a_mid,a_bot,a_sup]
    roles = ['TOP','JUNGLE','MID','DUO_CARRY','DUO_SUPPORT']

    winrate_dicts = []
    avg_winrates = dict()

    for champ_other,role_other in zip(ally_team,roles):
        if champ_other:
            wr = winrates.get_winrates(df,role,role_other,'blue',int(champ_other))
            winrate_dicts.append(wr)

    for id in champs_df['id']:
        winrate_list = []
        for dictionary in winrate_dicts:
            if id in dictionary:
                winrate_list.append(dictionary[id])
        if winrate_list:
            avg_winrates[id] = np.mean(winrate_list)

    pick = max(avg_winrates, key=avg_winrates.get,default=0)
    prob = round(100*avg_winrates[pick],3)
    pick = ids_dict[pick]

    return render_template('list.html', prob=prob, pick=pick)