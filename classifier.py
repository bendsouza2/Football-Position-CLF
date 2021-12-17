import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv('EPL1819.csv', header=0)
print(df.head())

unnecessary_columns = ['birthday',
                       'birthday_GMT',
                       'league',
                       'season',
                       'rank_in_league_top_defenders',
                       'rank_in_league_top_attackers',
                       'rank_in_league_top_midfielders',
                       'full_name',
                       'nationality',
                       'Current Club']

df_relevant = df.drop(unnecessary_columns, axis=1)

# print((df['rank_in_league_top_defenders'] == -1).count())

# only players who played at least 180 mins over the season
df_minsplayed = df_relevant.loc[df_relevant['minutes_played_overall'] > 180]

print(df_minsplayed.dtypes)

X = df_minsplayed.drop('position', axis=1)  # data for classification
y = df_minsplayed['position'].copy()  # columns to predict
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf_dt = DecisionTreeClassifier()
clf_dt = clf_dt.fit(X_train, y_train)


plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt, filled=True, feature_names=df_minsplayed.columns)

# class_names=['Goalkeeper', 'Forward', 'Midfielder', 'Defender'],

cm = ConfusionMatrixDisplay.from_estimator(clf_dt, X_test, y_test)

# Cross validation and pruning
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]  # removing max value

# Optimal alpha with cross-validation
alpha_loop_vals = []  # array to store results of each fold
for alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_vals.append([alpha, np.mean(scores), np.std(scores)])

# Plotting graph of accuracy scores
alpha_results = pd.DataFrame(alpha_loop_vals, columns=['alpha', 'mean_accuracy', 'std'])
alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')

plt.show()

