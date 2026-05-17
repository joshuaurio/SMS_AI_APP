import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("sms_data.csv", encoding="latin-1")
texts = data["text"].astype(str).values
labels = data["label"].values
print(data["label"].value_counts())
print(data["label"].unique())

SEQ_LEN = 40
VOCAB_SIZE = 3000

# Text processing
vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=SEQ_LEN
)
vectorizer.adapt(texts)

# Convert texts to integers FIRST (same as train_model_int.py)
x = vectorizer(texts)

# Build model (integer input, no vectorizer inside)
model = tf.keras.Sequential([
    layers.Input(shape=(SEQ_LEN,), dtype=tf.int32),
    layers.Embedding(VOCAB_SIZE, 16),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(x, labels, epochs=300, validation_split=0.2)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('SMS Fraud Detection Accuracy')
plt.show()

# Save model
model.save("sms_model.keras")
print("MODEL TRAINED SUCCESSFULLY")