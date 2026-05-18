import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load Data (Keep only the top 10,000 most frequent words)
vocab_size = 10000
max_length = 256
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# 2. Preprocess Data (Pad all reviews to exactly 256 words)
X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

# 3. Build the Model
model = Sequential()
model.add(Embedding(vocab_size, 16, input_length=max_length)) # Converts words to numerical vectors
model.add(GlobalAveragePooling1D()) # Averages the vectors
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Output layer: Sigmoid for Binary Classification

# 4. Compile the Model
# Loss is binary_crossentropy because it's a 2-class problem
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=512, validation_split=0.2, verbose=1)

# 6. Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

