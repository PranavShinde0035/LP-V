import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# 1. Load Data from local CSV file
csv_file_path = 'imdb_reviews.csv'

if not os.path.exists(csv_file_path):
    print(f"Error: {csv_file_path} not found! Please place your CSV file in the same folder.")
else:
    # Assumes CSV has 'review' (text) and 'sentiment' (0/1 or positive/negative) columns
    df = pd.read_csv(csv_file_path)

    # Convert labels to 0 and 1 if they are stored as strings (e.g. 'positive'/'negative')
    if df['sentiment'].dtype == 'object':
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X = df['review'].values
    y = df['sentiment'].values

    # Split into train/test sets (80% training, 20% testing)
    X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Tokenize and Pad the sequences
    vocab_size = 10000
    max_length = 256

    # Initialize and fit tokenizer on the training text reviews
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    # Convert text to sequences of numbers
    X_train = tokenizer.texts_to_sequences(X_train_text)
    X_test = tokenizer.texts_to_sequences(X_test_text)

    # Pad all sequences to exactly 256 words to match model expectations
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
