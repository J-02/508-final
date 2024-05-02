import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('/Users/tadeozuniga/PycharmProjects/508-final/data/Alasak_cleaned.csv')

# Prepare the data
X = data.drop(['species'], axis=1)
y = data['species']
X_encoded = pd.get_dummies(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest model
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
rf_importances = forest.feature_importances_ / np.sum(forest.feature_importances_)

# XGBoost model
dtrain = xgb.DMatrix(X_train, label=pd.factorize(y_train)[0])
params = {'objective': 'multi:softmax', 'num_class': len(np.unique(y_train)), 'max_depth': 6, 'learning_rate': 0.1, 'seed': 42}
bst = xgb.train(params, dtrain, num_boost_round=10)
xgb_importances = {int(k[1:]): v for k, v in bst.get_score(importance_type='weight').items()}
total_xgb_importance = sum(xgb_importances.values())
xgb_importances_normalized = {k: v / total_xgb_importance for k, v in xgb_importances.items()}
full_xgb_importances = [xgb_importances_normalized.get(i, 0) for i in range(len(X_encoded.columns))]

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Random Forest Importance': rf_importances,
    'XGBoost Importance': full_xgb_importances
})
comparison_df['RF Rank'] = comparison_df['Random Forest Importance'].rank(ascending=False, method='min')
comparison_df['XGB Rank'] = comparison_df['XGBoost Importance'].rank(ascending=False, method='min')

# Plotting feature importances
fig, ax = plt.subplots(figsize=(12, 8))
index = np.arange(len(X_encoded.columns))
bar_width = 0.35
ax.bar(index - bar_width/2, rf_importances, bar_width, label='Random Forest')
ax.bar(index + bar_width/2, full_xgb_importances, bar_width, label='XGBoost', align='center')
ax.set_xlabel('Features')
ax.set_ylabel('Importance')
ax.set_title('Feature Importance Comparison')
ax.set_xticks(index)
ax.set_xticklabels(X_encoded.columns, rotation=90)
ax.legend()
plt.savefig('/Users/tadeozuniga/PycharmProjects/508-final/data/feature_importance_plot.png')
plt.show()

# Sorting the DataFrame for table display
comparison_df_sorted = comparison_df.sort_values(by='Random Forest Importance', ascending=False)

# Plotting the formatted table
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=comparison_df_sorted.values, colLabels=comparison_df_sorted.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
# Bold top 5 features and their values in both algorithms
for i, (index, row) in enumerate(comparison_df_sorted.iterrows()):
    for j, key in enumerate(row.index):
        if (key == 'RF Rank' or key == 'Random Forest Importance') and row['RF Rank'] <= 5:
            table[(i + 1, j)].get_text().set_weight('bold')
        if (key == 'XGB Rank' or key == 'XGBoost Importance') and row['XGB Rank'] <= 5:
            table[(i + 1, j)].get_text().set_weight('bold')
plt.savefig('/Users/tadeozuniga/PycharmProjects/508-final/data/feature_importance_table.png', bbox_inches='tight', dpi=300)
plt.show()
