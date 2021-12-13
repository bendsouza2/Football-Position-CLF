import pandas as pd

df = pd.read_csv('EPL1819.csv', header=0)
print(df.head())


df_relevant = df.drop(['birthday', 'birthday_GMT', 'league', 'season'], axis=1)
print(df_relevant.dtypes)
print('defenders rank below')
print(df_relevant['rank_in_league_top_defenders'].unique())
print('-1 appears: \n')
print((df_relevant['rank_in_league_top_defenders'] == -1).count())
