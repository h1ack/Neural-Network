import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder

questions = [
    "hello", "hi", "hey", "good morning", "how are you",
    "what is your name", "who are you", "tell me your name", "what can you do","who is Your Creator"
]

responses = [
    "Hello!", "Hi!", "Hey!", "Good morning!",
    "I'm fine, thank you!", "I'm an AI chatbot.", "I'm a virtual assistant.", "My name is Bot!", "I can chat with you!", "My creator is You"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
X_sequences = tokenizer.texts_to_sequences(questions)
X_padded = pad_sequences(X_sequences, padding='post')

label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(responses)

model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=X_padded.shape[1]),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(64, activation="relu"),
    Dense(len(responses), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_padded, np.array(Y_encoded), epochs=100, verbose=1)

def chatbot_response(user_input):
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=X_padded.shape[1], padding='post')
    
    prediction = model.predict(padded_sequence)
    predicted_index = np.argmax(prediction)

    return label_encoder.inverse_transform([predicted_index])[0]

while True:
    user_input = input("You: ").lower()
    if user_input in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"Bot: {response}")
