import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler

# 1. Load Data
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# 2. Preprocess Data (Neural networks work best with scaled data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Build the Model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) # Output layer: 1 neuron, NO activation (because it's regression)

# 4. Compile the Model
# Loss is Mean Squared Error (MSE) because it's a regression problem
model.compile(optimizer='adam', loss='mse', metrics=['mae']) 

# 5. Train the Model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# 6. Evaluate the Model
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error on Test Data: {mae}")
