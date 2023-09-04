import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, RocCurveDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('Data/final_data.csv')

# Split the dataset
X = df.drop('repay_fail', axis=1)
y = df['repay_fail']
RAND_SEED = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RAND_SEED)

# Standardize features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
clf = LogisticRegression(class_weight='balanced', random_state=RAND_SEED, max_iter=1000).fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Logistic Regression Results:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve for Logistic Regression
y_scores = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.show()

auc_score = roc_auc_score(y_test, y_scores)
print("\nAUC-ROC Score:", auc_score)

# Random Forest
clf_rf = RandomForestClassifier(random_state=RAND_SEED)
clf_rf.fit(X_train, y_train)

y_pred_rf = clf_rf.predict(X_test)

print("\nRandom Forest Results:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# ROC Curve for Random Forest
y_scores_rf = clf_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_scores_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.show()

auc_score_rf = roc_auc_score(y_test, y_scores_rf)
print("\nAUC-ROC Score (Random Forest):", auc_score_rf)
