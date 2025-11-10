#Check class distribution
df['isFraud'].value_counts() #--> Data is excessively inbalanced fraud data is less that 0.01% of data

#Balance data for supervised model training 
legit = df[df.isFraud==0]
fraud = df[df.isFraud==1]

legit_sample = legit.sample(n=8500)
mdf = pd.concat([legit_sample, fraud], axis=0)
mdf.shape

#Split modified data
X = mdf.drop(['isFraud','isFlaggedFraud'], axis=1)
y = mdf['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Scale features 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
print("Logisitic Regression Results")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_lr))

#Random Forest Model 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=50,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Results")
print(classification_report(y_test, y_pred_rf))
print("ROC–AUC:", roc_auc_score(y_test, y_pred_rf))

#XG Boost Model
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("XGBoost Results")
print(classification_report(y_test, y_pred_xgb))
print("ROC–AUC:", roc_auc_score(y_test, y_pred_xgb))

#since model were trained on sample data,this tests xgb on full data to see how well it generalizes
X_full = df.drop(['isFraud','isFlaggedFraud'], axis=1)
y_full = df['isFraud']

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
X_test_full_scaled = scaler.transform(X_test_full)

y_pred = xgb.predict(X_test_full_scaled)
print(classification_report(y_test_full, y_pred))
print("ROC–AUC:", roc_auc_score(y_test_full, y_pred))

#Visualize Confusion matrices for supervised models
from sklearn.metrics import ConfusionMatrixDisplay
model = {'LogReg':log_reg, 'RandomForest':rf, 'XGBoost':xgb}
for name, m in model.items():
  disp = ConfusionMatrixDisplay.from_estimator(m, X_test_full_scaled, y_test_full, cmap='Blues')
  disp.ax_.set_title(name)
  plt.show()
