import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

train_data = pd.read_csv("C:/Users/Admin/Downloads/cumulativefiltered.csv")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_data = train_data[train_data['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].dropna()

feature_cols = ['koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth',
    'koi_prad', 'koi_model_snr', 'koi_steff', 'koi_srad',  'koi_kepmag']

X = train_data[feature_cols]
y = train_data['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)

print("Accuracy:", round(accuracy_score(y_test, rf_model.predict(X_test)), 2))
print(classification_report(y_test, rf_model.predict(X_test)))
