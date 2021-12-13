import pandas as pd

df = pd.read_csv('EPL1819.csv', header=0)
print(df.head())

unnecessary_columns = ['birthday',
                       'birthday_GMT',
                       'league',
                       'season',
                       'rank_in_league_top_defenders',
                       'rank_in_league_top_attackers',
                       'rank_in_league_top_midfielders']

df_relevant = df.drop(unnecessary_columns, axis=1)

# print((df['rank_in_league_top_defenders'] == -1).count())

# only players who played at least 180 mins over the season
df_minsplayed = df_relevant.loc[df_relevant['minutes_played_overall'] > 180]

print(df_minsplayed.dtypes)

X = df_minsplayed.drop('position', axis=1)  # data for classification
y = df_minsplayed['position'].copy()  # columns to predict
print(y)

