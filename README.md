# CompNet

Application to recommend your champ selection based on yours + enemies' team comp

Data Used:
(currently)

PLAN (as a reminder to myself so that I don't forget)

The kaggle dataset used is old but is large and contains a lot of information so is useful to build up a method that works well as a proof of concept.
First I would like to build 2 different models and test these to see how well they work before looking for potential improvements:
  1. 'Naive' model, checks the win rates of every champ in your team with all of the possible champs you can play, then averages this.
      (And the same for every champ in enemy team's win rates vs. every champ you can play). Champ with the highest win rate is what you pick.
  2.  Create/optimise a neural net that will predict based on game stats (gold earned, kills, etc) if the match was won or lost. Then, from the dataset,
      for every champ, find an average of these stats across all games played with you team's champs (and enemy champs) and insert into the model. The sigmoid
      activation output can be treated as a 'win probability' for this champ combination. Across all champs, the win prob is averaged and the champ with the highest
      average win prob is what you pick.

These are pretty basic ideas but will hold out a bit of the data for validation and see if I am onto something with any of these methods. Can then research more/ increase
complexity of these if they are good with heuristics etc.

End goal is to get Riot approval and create a functioning web application that I can use to get me to diamond ;) 
