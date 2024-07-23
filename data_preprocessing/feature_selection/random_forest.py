import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import os


def aggregate_two_sorted_rankings(dataframe, first_col_name, second_col_name):
    # print("\n[function aggregate_two_sorted_rankings()]")

    df_sorted_by_first = dataframe.sort_values(by=first_col_name, ascending=False)
    df_sorted_by_second = dataframe.sort_values(by=second_col_name, ascending=False)

    df_sorted_by_first['firstColPos'] = np.arange(1, len(df_sorted_by_first) + 1)
    df_sorted_by_second['secondColPos'] = np.arange(1, len(df_sorted_by_second) + 1)

    merged_df = pd.merge(df_sorted_by_first[[first_col_name, 'firstColPos']],
                        df_sorted_by_second[[second_col_name, 'secondColPos']],
                        left_index=True, right_index=True, how='outer')

    merged_df['posSum'] = merged_df['firstColPos'] + merged_df['secondColPos']
    merged_df_sorted = merged_df.sort_values(by='posSum')
    merged_df_sorted['finalPos'] = np.arange(1, len(merged_df_sorted) + 1)

    return merged_df_sorted

def rf_feature_selection(dataframe, target_col, print_fs):
  no_features = int(os.getenv('min_features'))
  # Feature importance and ranking aggregation over multiple random forest executions
  execution_number = 100
  final_rankings = pd.DataFrame()

  for i in range(execution_number):
      if print_fs:
        print(f"\n\nExecution number {i + 1}")

      # Random Forest
      rf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
      features = [col for col in dataframe.columns if col != target_col]
      rf.fit(dataframe[features], dataframe[target_col])
      importances = pd.DataFrame(rf.feature_importances_, index=features, columns=['MeanDecreaseAccuracy'])

      importances['MeanDecreaseGini'] = rf.feature_importances_  # Placeholder for actual Gini importance calculation

      # Aggregate rankings
      aggregated_rank = aggregate_two_sorted_rankings(importances, 'MeanDecreaseAccuracy', 'MeanDecreaseGini')

      if i == 0:
          final_rankings = aggregated_rank
      else:
          final_rankings['MeanDecreaseAccuracy'] += aggregated_rank['MeanDecreaseAccuracy']
          final_rankings['MeanDecreaseGini'] += aggregated_rank['MeanDecreaseGini']
          final_rankings['finalPos'] += aggregated_rank['finalPos']

  # Average the results over all executions
  final_rankings['MeanDecreaseAccuracy'] /= execution_number
  final_rankings['MeanDecreaseGini'] /= execution_number
  final_rankings['finalPos'] /= execution_number
  
  if print_fs:
    print("\nFinal ranking after 100 executions:")
    print(final_rankings[['finalPos', 'MeanDecreaseAccuracy', 'MeanDecreaseGini']])

    plt.figure(figsize=(10, 8))
    plt.barh(final_rankings.index, final_rankings['MeanDecreaseAccuracy'], color='blue', label='Mean Decrease Accuracy')
    plt.barh(final_rankings.index, final_rankings['MeanDecreaseGini'], color='red', alpha=0.6, label='Mean Decrease Gini')
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.legend()
    plt.show()

  final_rankings_features = final_rankings.index.to_list()

  return final_rankings_features[: no_features]