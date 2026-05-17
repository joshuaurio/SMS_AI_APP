# 📱 SMS Fraud Detector — Swahili AI

An AI-powered SMS fraud detection system trained on Swahili text messages. It classifies SMS as **FRAUD ⚠️** or **SAFE ✅** using a TensorFlow text classification model. Includes a Flutter-ready TFLite export for mobile deployment.

---

## 🧠 What It Does

- Trains a neural network on labeled Swahili SMS data
- Detects fraud patterns like fake prizes, witchdoctor scams, phishing links, and M-Pesa fraud
- Exports a `.tflite` model and `vocab.json` for use in a Flutter mobile app
- Supports both string-input and integer-input model variants

---

## 📁 Project Structure

```
SMS_AI_APP/
├── sms_data.csv           # Training data (SMS text + label)
├── vocab.json             # Vocabulary list (auto-generated)
├── train_model.py         # Train model with built-in TextVectorization
├── train_model_int.py     # Train integer-input model (Flutter-compatible)
├── test_sms.py            # Test the string-input model
├── test_model_int.py      # Test the integer-input model
├── export_for_flutter.py  # Export TFLite model + vocab for Flutter
├── sms_model.keras        # Trained model (string input) — not in repo
├── sms_model_int.keras    # Trained model (int input) — not in repo
└── sms_model_flutter.tflite  # TFLite model — not in repo
```

> **Note:** `.keras` and `.tflite` files are excluded from the repo (see `.gitignore`) because they are large binary files. Generate them by running the training scripts.

---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- matplotlib

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SMS_AI_APP.git
cd SMS_AI_APP
```

### 2. Install dependencies

```bash
pip install tensorflow pandas numpy matplotlib
```

### 3. Prepare your data

Make sure `sms_data.csv` is in the project folder. It must have two columns:

| text | label |
|------|-------|
| Umeshinda zawadi... | 1 |
| Nakuja nyumbani saa moja | 0 |

- `1` = Fraud
- `0` = Safe/Legitimate

---

## 🏋️ Training the Model

### Option A — String-input model (simpler, not Flutter-compatible)

```bash
python train_model.py
```

Outputs: `sms_model.keras`

### Option B — Integer-input model (recommended, Flutter-compatible)

```bash
python train_model_int.py
```

Outputs: `sms_model_int.keras` and `vocab.json`

---

## 🧪 Testing the Model

### Test string-input model:

```bash
python test_sms.py
```

### Test integer-input model:

```bash
python test_model_int.py
```

Example output:

```
SMS: Umeshinda ela, tuma namba 12345 kupata zawadi
Prediction: FRAUD ⚠️
Confidence: 0.981
--------------------------------------------------
SMS: Nakuja nyumbani saa moja
Prediction: SAFE ✅
Confidence: 0.042
```

---

## 📦 Exporting for Flutter

```bash
python export_for_flutter.py
```

Outputs:
- `sms_model_flutter.tflite` — optimized model for mobile
- `vocab.json` — vocabulary for tokenizing SMS in Flutter/Dart

---

## 📊 Model Architecture

```
Input (integer token IDs, length 40)
  → Embedding (vocab_size=3000, dim=16)
  → GlobalAveragePooling1D
  → Dense(16, relu)
  → Dense(1, sigmoid)
```

- **Loss:** Binary Crossentropy
- **Optimizer:** Adam
- **Epochs:** 300
- **Validation split:** 20%

---

## 🌍 Language

This model is trained on **Swahili (Kiswahili)** SMS messages, targeting fraud patterns common in Tanzania, including:

- Fake prize/lottery scams
- Witchdoctor (mganga) solicitations
- M-Pesa phishing
- Suspicious loan offers
- Fake recruitment messages

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Joshua Urio**  
Built with TensorFlow · Designed for Tanzania 🇹🇿
