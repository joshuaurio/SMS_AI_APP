import tensorflow as tf
import numpy as np
import json

SEQ_LEN = 40

# Load model
model = tf.keras.models.load_model("sms_model_int.keras")

# Load vocabulary
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Recreate vectorizer using saved vocab
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=len(vocab),
    output_mode="int",
    output_sequence_length=SEQ_LEN,
)

vectorizer.set_vocabulary(vocab)

# Test SMS messages
test_messages = [
    "Umeshinda ela, tuma namba 12345 kupata zawadi",
    "Tafadhali tuma namba yako ya simu kupata zawadi",
    "Nakuja nyumbani saa moja",
    "Bonyeza link kuthibitisha akaunti yako",
    "Umepokea pesa kutoka kwa mama",
    "WASILIANA NAMGANGA MZEE VIKONA ANATOWA MARI BILA KAFARA MVUTO KESI BIASHARA NYOTA PETE MAGONJWA PESA ZA MAJINI KUTULIZA MKE MME PG 0622528503 AU 0691176752 ",
    "MPIGIE MGANGA WA JADI Mzee JIMSON ANATOWA MARI BILA KAFARA MVUTO KESI BIASHARA NYOTA PETE MAGONJW PESA ZA MAJINI KUTULIZA MKE MME pg 0621597991 au 0619541686",
    "Utanitumia kwa hi'i Airtel 0785358469 jina ALIKO MWAKIGONJOLA.",
    "WASILIANA NAMGANGA MZEE VIKONA ANATOWA MARI BILA KAFARA MVUTO KESI BIASHARA NYOTA PETE MAGONJWA PESA ZA MAJINI KUTULIZA MKE MME PG 0622528503 AU 0691176752",
    "Habari, Salio lako halitoshi! Jibu YES kwenye SMS hii kupata mkopo wa muda wa maongezi mpaka Tsh 6000 na ulipe utakapoongeza salio au piga *149*01*99# kwa mikopo zaidi."
]

test_messages = np.array(test_messages)

# Convert to integer sequences
x_test = vectorizer(test_messages)

# Predict
predictions = model.predict(x_test)

# Show results
for msg, pred in zip(test_messages, predictions):
    confidence = float(pred[0])
    label = "FRAUD ⚠️" if confidence > 0.5 else "SAFE ✅"

    print("SMS:", msg)
    print("Prediction:", label)
    print("Confidence:", round(confidence, 3))
    print("-" * 50)
