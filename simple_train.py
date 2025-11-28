"""
Ultra Simple Training - No Pandas Required
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("SIMPLE MODEL TRAINING (No Pandas)")
print("=" * 60)

# Ensure directories exist
os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Create sample syscall data directly
print("\n📂 Creating sample data...")
syscalls = []
patterns = [
    ['open', 'read', 'close'],
    ['open', 'write', 'close'],
    ['open', 'fstat', 'read', 'close'],
    ['stat', 'open', 'read', 'write', 'close'],
    ['open', 'mmap', 'munmap', 'close']
]

# Generate realistic patterns
for _ in range(100):
    for pattern in patterns:
        syscalls.extend(pattern)

print(f"✅ Created {len(syscalls)} syscalls")

# Encode syscalls
print("\n🔄 Encoding syscalls...")
encoder = LabelEncoder()
encoded = encoder.fit_transform(syscalls)

print(f"✅ Found {len(encoder.classes_)} unique syscalls: {list(encoder.classes_)}")

# Create sequences
print("\n🔄 Creating sequences...")
sequence_length = 10
X, y = [], []

for i in range(len(encoded) - sequence_length):
    X.append(encoded[i:i + sequence_length])
    y.append(encoded[i + sequence_length])

X = np.array(X)
y = np.array(y)

print(f"✅ Created {len(X)} sequences")

# Split data (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   Training: {len(X_train)}")
print(f"   Testing: {len(X_test)}")

# Train model
print("\n🚀 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=50,  # Fewer trees for speed
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\n✅ Training completed!")
print(f"   Training Accuracy: {train_acc*100:.2f}%")
print(f"   Testing Accuracy: {test_acc*100:.2f}%")

# Save model
print("\n💾 Saving files...")
with open('models/saved_models/rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Saved: rf_model.pkl")

# Save encoder
with open('models/saved_models/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
print("✅ Saved: label_encoder.pkl")

# Save mappings
syscall_to_id = {name: idx for idx, name in enumerate(encoder.classes_)}
id_to_syscall = {idx: name for name, idx in syscall_to_id.items()}

mappings = {
    'syscall_to_id': syscall_to_id,
    'id_to_syscall': id_to_syscall,
    'n_syscalls': len(encoder.classes_)
}

with open('models/saved_models/syscall_mappings.pkl', 'wb') as f:
    pickle.dump(mappings, f)
print("✅ Saved: syscall_mappings.pkl")

# Also save a CSV for the dashboard
print("\n💾 Creating CSV data...")
with open('data/processed/parsed_syscalls.csv', 'w') as f:
    f.write("timestamp,syscall,duration,category,return_value,source_file\n")
    for i, sc in enumerate(syscalls[:500]):  # Save first 500
        category = 'file' if sc in ['open', 'read', 'write', 'close'] else 'memory'
        f.write(f"2024-01-01 00:00:{i%60:02d},{sc},{np.random.uniform(0.0001, 0.01):.6f},{category},0,trace_1.txt\n")
print("✅ Saved: parsed_syscalls.csv")

# Test prediction
print("\n" + "=" * 60)
print("TESTING PREDICTION")
print("=" * 60)

test_sequence = X_test[0]
actual = y_test[0]

print(f"\n📝 Input sequence:")
for syscall_id in test_sequence:
    print(f"   - {id_to_syscall[syscall_id]}")

# Predict
predicted_proba = model.predict_proba([test_sequence])[0]
predicted_id = model.predict([test_sequence])[0]
confidence = predicted_proba[predicted_id]

predicted_name = id_to_syscall[predicted_id]
actual_name = id_to_syscall[actual]

print(f"\n🎯 Prediction:")
print(f"   Predicted: {predicted_name}")
print(f"   Confidence: {confidence*100:.2f}%")
print(f"   Actual: {actual_name}")
print(f"   {'✅ Correct!' if predicted_name == actual_name else '❌ Incorrect'}")

# Show top 3 predictions
print(f"\n📊 Top 3 predictions:")
top_3_idx = np.argsort(predicted_proba)[-3:][::-1]
for i, idx in enumerate(top_3_idx, 1):
    print(f"   {i}. {id_to_syscall[idx]:10s} ({predicted_proba[idx]*100:.1f}%)")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\n📁 Created files:")
print("   ✅ models/saved_models/rf_model.pkl")
print("   ✅ models/saved_models/label_encoder.pkl")
print("   ✅ models/saved_models/syscall_mappings.pkl")
print("   ✅ data/processed/parsed_syscalls.csv")
print("\n🚀 Now you can run:")
print("   python models/predict.py")
print("   streamlit run dashboard/app.py")