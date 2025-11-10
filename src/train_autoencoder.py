X_legit = X_train[y_train == 0]
X_fraud = X_train[y_train == 1]

X_legit_train, X_legit_val = train_test_split(X_legit, test_size=0.2, random_state=42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

input_dim = X_legit_train.shape[1]
encoding_dim = 8

input_layer = Input(shape=[input_dim])
x = Dense(64, activation="relu")(input_layer)
x = Dense(32, activation="relu")(x)
bottleneck = Dense(8, activation="relu")(x)
x = Dense(32, activation="relu")(bottleneck)
x = Dense(64, activation="relu")(x)
output = Dense(input_dim, activation="linear")(x)

autoencoder = Model(input_layer, output)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')
autoencoder.fit(
    X_legit_train, X_legit_train,
    validation_data=(X_legit_val, X_legit_val),
    epochs=20,
    batch_size=64,
    shuffle=True
)

reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.square(X_test - reconstructions), axis=1)
#Anomaly Threshold
threshold = np.percentile(
    np.mean(np.square(X_legit_val-autoencoder.predict(X_legit_val)), axis=1), 99)
#Flagging Anomalies
auto_preds = (mse > threshold).astype(int)

#Evaluate the autoencoder model 
print(classification_report(y_test, auto_preds))
print("ROC-AUC:", roc_auc_score(y_test_full, mse))
