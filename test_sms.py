import tensorflow as tf

# Load trained model (includes TextVectorization)
model = tf.keras.models.load_model("sms_model.keras")

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


# Convert to Tensor
test_messages = tf.constant(test_messages)

# Predict
predictions = model.predict(test_messages)

# Show results
for msg, pred in zip(test_messages.numpy().astype(str), predictions):
    label = "FRAUD ⚠️" if pred[0] > 0.5 else "SAFE ✅"
    print(f"SMS: {msg}")
    print(f"Prediction: {label}\n")
