# CompNet

Application to recommend your champ selection based on yours + enemies' team comp

Data Used: https://www.kaggle.com/datasets/paololol/league-of-legends-ranked-matches
(currently)

## FILES

- data_create.py: This uses all of the files (located above) to create a dataset I can use for modelling purposes
- win_predict.py: The file containing the neural net framework
- main.py: The file you need to run to actually get your custom champion recommendations!

### Naive - folder for method 1 (naive model)

- winrates.py: contains function for calculating winrates

## PLAN (as a reminder to myself so that I don't forget)

The kaggle dataset used is old but is large and contains a lot of information so is useful to build up a method that works well as a proof of concept.
First I would like to build 2 different models and test these to see how well they work before looking for potential improvements:
  1. 'Naive' model, checks the win rates of every champ in your team with all of the possible champs you can play, then averages this.
      (And the same for every champ in enemy team's win rates vs. every champ you can play). Champ with the highest win rate is what you pick.
  2.  Create/optimise a neural net that will predict based on game stats (gold earned, kills, etc) if the match was won or lost. Then, from the dataset,
      for every champ, find an average of these stats across all games played with you team's champs (and enemy champs) and insert into the model. The sigmoid
      activation output can be treated as a 'win probability' for this champ combination. Across all champs, the win prob is averaged and the champ with the highest
      average win prob is what you pick.

These are my initial ideas, and also it will be quite hard to test how effective they are. I can see if the approach of averaging is any good by holding out some data and 
finding the average euclidean distance between averaged match stats and actual (i.e. if the averaged stats are averaging <2 kills away from actual this is pretty good).
Can then research more/ increase complexity of this if it is good with heuristics etc.

Naive model I can test by using it??? To play a lot of games, and see what the win rate is like with/without it but that is quite time consuming. This model is probably more meant 
as a baseline anyway.

Would also be extremely beneficial to add in heuristics regarding the individual player, such as champion proficiency (so for example me if I am using it)
However for testing, I can just play the highest 'rated' champ that is in my champ pool (of like 7 champs lol)

End goal is to get Riot approval and create a functioning web application that I can use to get me to diamond ;) 
