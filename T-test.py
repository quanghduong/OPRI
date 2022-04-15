df_score = df_score.reset_index().rename_axis(None, axis=1)

#balanced accuracy
df_score_pivot = df_score[['Balanced Accuracy','Model']].pivot(columns='Model', values='Balanced Accuracy')
columns = df_score_pivot.columns
pivot_score = pd.DataFrame()
for i in columns:
  new = df_score_pivot[[i]].dropna().reset_index(drop=True)
  pivot_score = pd.concat([pivot_score, new], axis=1)

import scipy
import itertools
list(itertools.combinations_with_replacement(score_pivot.columns))
a = df_score['Balanced Accuracy'].loc[df_score['Model'] == 'Random Forest']
b = df_score['Balanced Accuracy'].loc[df_score['Model'] == 'Support Vector Machine']
result = scipy.stats.ttest_ind(a, b)
