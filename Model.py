from Graph_states import sample_density_matrix, gell_mann_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
import numpy as np


X, y = [], []

while len(X) < 30000:
    out = sample_density_matrix()
    if out is None:
        continue

    rho, label, alpha, alpha_ppt = out
    features = gell_mann_features(rho)

    X.append(features)
    y.append(label)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

'''scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


best_svm = SVC(
    kernel='rbf',
    C=5.0,
    gamma=0.1,
    class_weight='balanced'
)

best_svm.fit(X_train, y_train)

y_pred = best_svm.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))'''



'''best_rf = RandomForestClassifier(
    n_estimators=600,                 # or 800 if you want extra stability
    max_depth=20,
    max_features=0.5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))'''

gb = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
