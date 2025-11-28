"""
Machine Learning Model Training
Trains LSTM model to predict system calls
"""

import numpy as np
import pickle
import os
from prepare_data import DataPreparator

# Try to import TensorFlow
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available, using simple ML model instead")
    from sklearn.ensemble import RandomForestClassifier
    TENSORFLOW_AVAILABLE = False

class SyscallPredictor:
    def __init__(self, sequence_length=10):
        """Initialize the predictor"""
        self.sequence_length = sequence_length
        self.model = None
        self.model_type = "LSTM" if TENSORFLOW_AVAILABLE else "RandomForest"
    
    def build_lstm_model(self, n_syscalls):
        """Build LSTM neural network model"""
        print(f"\n🏗️  Building LSTM model...")
        
        model = Sequential([
            # Embedding layer to convert syscall IDs to vectors
            Embedding(input_dim=n_syscalls, output_dim=32, 
                     input_length=self.sequence_length),
            
            # LSTM layer to learn patterns
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            
            LSTM(32),
            Dropout(0.2),
            
            # Output layer - predict next syscall
            Dense(n_syscalls, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ LSTM model built successfully")
        model.summary()
        
        return model
    
    def build_random_forest_model(self):
        """Build Random Forest model (fallback)"""
        print(f"\n🌲 Building Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        print("✅ Random Forest model built successfully")
        return model
    
    def train(self, X_train, y_train, X_test, y_test, n_syscalls):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            n_syscalls: Number of unique system calls
        """
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        if TENSORFLOW_AVAILABLE:
            self.model = self.build_lstm_model(n_syscalls)
            
            # Callbacks for training
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint('models/saved_models/best_model.h5', 
                              save_best_only=True)
            ]
            
            print(f"\n🚀 Training LSTM model...")
            print(f"   Epochs: 20")
            print(f"   Batch size: 32")
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=20,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"\n✅ Training completed!")
            print(f"   Test Accuracy: {test_acc*100:.2f}%")
            print(f"   Test Loss: {test_loss:.4f}")
            
            return history
        
        else:
            self.model = self.build_random_forest_model()
            
            print(f"\n🚀 Training Random Forest model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_acc = self.model.score(X_train, y_train)
            test_acc = self.model.score(X_test, y_test)
            
            print(f"\n✅ Training completed!")
            print(f"   Training Accuracy: {train_acc*100:.2f}%")
            print(f"   Testing Accuracy: {test_acc*100:.2f}%")
            
            # Save model
            with open('models/saved_models/rf_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            return None
    
    def predict_next_syscall(self, sequence, id_to_syscall):
        """
        Predict next system call given a sequence
        
        Args:
            sequence: List of syscall IDs
            id_to_syscall: Mapping from ID to syscall name
        
        Returns:
            Predicted syscall name and probability
        """
        if len(sequence) != self.sequence_length:
            print(f"❌ Sequence must be {self.sequence_length} syscalls")
            return None, 0
        
        # Prepare input
        X = np.array([sequence])
        
        # Predict
        if TENSORFLOW_AVAILABLE:
            predictions = self.model.predict(X, verbose=0)[0]
            predicted_id = np.argmax(predictions)
            confidence = predictions[predicted_id]
        else:
            predictions = self.model.predict_proba(X)[0]
            predicted_id = np.argmax(predictions)
            confidence = predictions[predicted_id]
        
        predicted_syscall = id_to_syscall[predicted_id]
        
        return predicted_syscall, confidence
    
    def predict_sequence(self, initial_sequence, n_predictions, id_to_syscall):
        """
        Predict a sequence of future system calls
        
        Args:
            initial_sequence: Starting sequence
            n_predictions: Number of calls to predict
            id_to_syscall: Mapping dictionary
        
        Returns:
            List of predicted syscalls
        """
        current_sequence = list(initial_sequence)
        predictions = []
        
        for _ in range(n_predictions):
            # Get last N syscalls
            input_seq = current_sequence[-self.sequence_length:]
            
            # Predict next
            next_syscall, confidence = self.predict_next_syscall(
                input_seq, id_to_syscall
            )
            
            predictions.append({
                'syscall': next_syscall,
                'confidence': confidence
            })
            
            # Add to sequence for next prediction
            # Find the ID of predicted syscall
            syscall_id = [k for k, v in id_to_syscall.items() 
                         if v == next_syscall][0]
            current_sequence.append(syscall_id)
        
        return predictions


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("SYSTEM CALL PREDICTION MODEL - TRAINING")
    print("=" * 60)
    
    # Step 1: Prepare data
    preparator = DataPreparator(sequence_length=10)
    data = preparator.prepare_all()
    
    if data is None:
        print("\n❌ Data preparation failed. Exiting.")
        return
    
    # Step 2: Train model
    predictor = SyscallPredictor(sequence_length=10)
    predictor.train(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['n_syscalls']
    )
    
    # Step 3: Test predictions
    print("\n" + "=" * 60)
    print("TESTING PREDICTIONS")
    print("=" * 60)
    
    # Load mappings
    with open('models/saved_models/syscall_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    
    # Test on a sample sequence
    sample_sequence = data['X_test'][0]
    actual_next = data['y_test'][0]
    
    print(f"\n📝 Input sequence:")
    for syscall_id in sample_sequence:
        print(f"   - {mappings['id_to_syscall'][syscall_id]}")
    
    predicted, confidence = predictor.predict_next_syscall(
        sample_sequence, mappings['id_to_syscall']
    )
    
    actual = mappings['id_to_syscall'][actual_next]
    
    print(f"\n🎯 Prediction:")
    print(f"   Predicted: {predicted} (confidence: {confidence*100:.2f}%)")
    print(f"   Actual: {actual}")
    print(f"   {'✅ Correct!' if predicted == actual else '❌ Incorrect'}")
    
    # Predict sequence
    print(f"\n🔮 Predicting next 5 system calls:")
    future_calls = predictor.predict_sequence(
        sample_sequence, 5, mappings['id_to_syscall']
    )
    
    for i, pred in enumerate(future_calls, 1):
        print(f"   {i}. {pred['syscall']} ({pred['confidence']*100:.1f}%)")
    
    print("\n✅ Training and testing completed!")
    print("📁 Model saved in: models/saved_models/")


if __name__ == "__main__":
    main()