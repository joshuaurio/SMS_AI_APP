import json
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

SEQ_LEN = 40
VOCAB_SIZE = 3000

data = pd.read_csv("sms_data.csv", encoding="latin-1")
texts = data["text"].astype(str).values
labels = data["label"].values


vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LEN,
)
vectorizer.adapt(texts)

# Save vocabulary for Flutter tokenization
vocab = vectorizer.get_vocabulary()
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

# Convert all texts to integer sequences for training
x = vectorizer(texts)

# Classifier that accepts integer token IDs (no TextVectorization inside)
classifier = tf.keras.Sequential(
    [
        layers.Input(shape=(SEQ_LEN,), dtype=tf.int32),
        layers.Embedding(VOCAB_SIZE, 16),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
classifier.fit(x, labels, epochs=300, validation_split=0.2)

classifier.save("sms_model_int.keras")
print("Saved sms_model_int.keras and vocab.json")