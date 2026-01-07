from Graph_states import sample_density_matrix, gell_mann_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

X, y = [], []
alphas, alphas_ppt = [], []

while len(X) < 30000:
    out = sample_density_matrix()
    if out is None:
        continue

    rho, label, alpha, alpha_ppt = out
    X.append(gell_mann_features(rho))
    y.append(label)
    alphas.append(alpha)
    alphas_ppt.append(alpha_ppt)

X = np.array(X)
y = np.array(y)
delta = np.array(alphas) - np.array(alphas_ppt)

X_train, X_test, y_train, y_test, delta_train, delta_test = train_test_split(
    X, y, delta,
    test_size=0.25,
    stratify=y,
    random_state=42
)

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

correct = (y_pred == y_test)

'''plt.figure(figsize=(7,4))

plt.scatter(
    delta_test[correct],
    np.zeros_like(delta_test[correct]),
    s=12, alpha=0.4, label="Correct"
)

plt.scatter(
    delta_test[~correct],
    np.zeros_like(delta_test[~correct]),
    s=25, color='red', label="Misclassified"
)

plt.axvline(0, color='k', linestyle='--', label=r'$\alpha=\alpha_{\rm PPT}$')
plt.yticks([])
plt.xlabel(r'$\Delta = \alpha - \alpha_{\rm PPT}$')
plt.title("Misclassifications cluster near the PPT boundary")
plt.legend()
plt.show()'''

bins = np.linspace(0, np.max(np.abs(delta_test)), 20)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

error_rate = []

for i in range(len(bins)-1):
    mask = (np.abs(delta_test) >= bins[i]) & (np.abs(delta_test) < bins[i+1])
    if np.sum(mask) == 0:
        error_rate.append(np.nan)
    else:
        error_rate.append(np.mean(~correct[mask]))

error_rate = np.array(error_rate)

plt.figure(figsize=(6,4))
plt.plot(bin_centers, error_rate, marker='o')
plt.xlabel(r'$|\alpha - \alpha_{\rm PPT}|$')
plt.ylabel('Misclassification probability')
plt.title('Error rate vs distance to PPT boundary')
plt.grid(alpha=0.3)
plt.show()


