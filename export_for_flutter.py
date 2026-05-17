import json
import tensorflow as tf


MAX_TOKENS = 3000
SEQ_LEN = 40


def main():
    # Load your trained model (the one that includes TextVectorization)
    model = tf.keras.models.load_model("sms_model.keras")
    vectorizer = model.layers[0]  # TextVectorization
    embedding = model.layers[1]
    pool = model.layers[2]
    dense1 = model.layers[3]
    dense2 = model.layers[4]

    # Save vocabulary so Flutter can tokenize the same way
    vocab = vectorizer.get_vocabulary()
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # Build an "int-input" model: input is [1, SEQ_LEN] int32
    int_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(SEQ_LEN,), dtype=tf.int32),
            tf.keras.layers.Embedding(MAX_TOKENS, 16),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Copy weights from trained model
    int_model.layers[0].set_weights(embedding.get_weights())
    int_model.layers[1].set_weights(pool.get_weights())
    int_model.layers[2].set_weights(dense1.get_weights())
    int_model.layers[3].set_weights(dense2.get_weights())

    # Convert to TFLite (BUILTINS only, Flutter-friendly)
    converter = tf.lite.TFLiteConverter.from_keras_model(int_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("sms_model_flutter.tflite", "wb") as f:
        f.write(tflite_model)

    print("DONE:")
    print("- vocab.json")
    print("- sms_model_flutter.tflite")


if __name__ == "__main__":
    main()