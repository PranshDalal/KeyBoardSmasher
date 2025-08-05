import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
data = pd.read_csv('data.csv')
texts = data['keyboard_smash'].astype(str).tolist()
labels = data['mood'].tolist()

# Tokenize character-level
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad to max length or median length
MAX_LEN = int(np.percentile([len(s) for s in sequences], 95))  # More dynamic than fixed max
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# Label encoding
label_enc = LabelEncoder()
y = label_enc.fit_transform(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)

# Build improved model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_enc.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Add early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Prediction function
def predict_mood(smash):
    seq = tokenizer.texts_to_sequences([smash])
    padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded_seq, verbose=0)
    mood = label_enc.inverse_transform([np.argmax(pred)])
    return mood[0]

# Test it
test_inputs = ["asdfghjkl", "AAAAAAAAA", "fjdsklfjdslk", "🥲🥲🥲", ".........", "LOLOLLOL"]
for smash in test_inputs:
    print(f'Input: "{smash}" → Mood: {predict_mood(smash)}')
