import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("sms_data.csv", encoding="latin-1")
texts = data["text"].values
labels = data["label"].values
print(data["label"].value_counts())
print(data["label"].unique())

# Text processing
vectorizer = layers.TextVectorization(
    max_tokens=3000,
    output_sequence_length=40
)
vectorizer.adapt(texts)

# Build model
model = tf.keras.Sequential([
    vectorizer,
    layers.Embedding(3000, 16),
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
history = model.fit(texts, labels, epochs=300, validation_split=0.2)

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
