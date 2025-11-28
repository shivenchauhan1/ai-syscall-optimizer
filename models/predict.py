"""
Prediction Module
Uses trained model to predict system calls - optimized for large datasets
"""

import numpy as np
import pickle
import os
import sys

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False

class SyscallPredictorInference:
    def __init__(self, model_dir="models/saved_models"):
        """Load trained model and encoders"""
        self.model_dir = model_dir
        self.model = None
        self.mappings = None
        self.sequence_length = 10
        self.load_model_and_mappings()
    
    def load_model_and_mappings(self):
        """Load the trained model and mappings"""
        try:
            # Load mappings
            mappings_path = os.path.join(self.model_dir, "syscall_mappings.pkl")
            
            if not os.path.exists(mappings_path):
                print(f"❌ Mappings file not found: {mappings_path}")
                print("   Please train the model first: python models/train.py")
                return
            
            with open(mappings_path, 'rb') as f:
                self.mappings = pickle.load(f)
            
            print(f"✅ Loaded mappings: {self.mappings['n_syscalls']} unique syscalls")
            
            # Load model
            if TENSORFLOW_AVAILABLE:
                model_path = os.path.join(self.model_dir, "best_model.h5")
                if os.path.exists(model_path):
                    self.model = load_model(model_path)
                    print(f"✅ Loaded LSTM model from {model_path}")
                else:
                    print(f"⚠️  LSTM model not found, trying Random Forest...")
                    self._load_random_forest()
            else:
                self._load_random_forest()
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("   Make sure you've trained the model first!")
            import traceback
            traceback.print_exc()
    
    def _load_random_forest(self):
        """Load Random Forest model"""
        rf_path = os.path.join(self.model_dir, "rf_model.pkl")
        if os.path.exists(rf_path):
            with open(rf_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Loaded Random Forest model from {rf_path}")
        else:
            print(f"❌ No model found at {rf_path}")
    
    def predict(self, syscall_names):
        """
        Predict next syscall from a sequence of names
        
        Args:
            syscall_names: List of syscall names (strings), must be exactly 10
        
        Returns:
            Predicted syscall name and confidence
        """
        if self.model is None or self.mappings is None:
            print("❌ Model or mappings not loaded")
            return None, 0
        
        if len(syscall_names) != self.sequence_length:
            print(f"❌ Sequence must be exactly {self.sequence_length} syscalls, got {len(syscall_names)}")
            return None, 0
        
        # Convert names to IDs
        try:
            syscall_ids = []
            for name in syscall_names:
                if name in self.mappings['syscall_to_id']:
                    syscall_ids.append(self.mappings['syscall_to_id'][name])
                else:
                    # Use most common syscall as fallback
                    print(f"⚠️  Unknown syscall '{name}', using 'open' as fallback")
                    syscall_ids.append(self.mappings['syscall_to_id'].get('open', 0))
                    
        except KeyError as e:
            print(f"❌ Unknown syscall: {e}")
            return None, 0
        
        # Predict
        X = np.array([syscall_ids])
        
        try:
            if TENSORFLOW_AVAILABLE and hasattr(self.model, 'predict'):
                predictions = self.model.predict(X, verbose=0)[0]
            else:
                # Random Forest
                predictions = self.model.predict_proba(X)[0]
            
            predicted_id = np.argmax(predictions)
            confidence = float(predictions[predicted_id])
            
            predicted_name = self.mappings['id_to_syscall'][predicted_id]
            
            return predicted_name, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return None, 0
    
    def predict_batch(self, sequences):
        """
        Predict next syscalls for multiple sequences (optimized for large datasets)
        
        Args:
            sequences: List of lists, each inner list is a sequence of syscall names
        
        Returns:
            List of (predicted_syscall, confidence) tuples
        """
        if self.model is None or self.mappings is None:
            return [(None, 0)] * len(sequences)
        
        results = []
        
        # Convert all sequences to IDs
        X_batch = []
        valid_indices = []
        
        for i, seq in enumerate(sequences):
            if len(seq) != self.sequence_length:
                results.append((None, 0))
                continue
            
            try:
                syscall_ids = [self.mappings['syscall_to_id'].get(name, 0) for name in seq]
                X_batch.append(syscall_ids)
                valid_indices.append(i)
            except:
                results.append((None, 0))
        
        if not X_batch:
            return results
        
        # Batch prediction
        X_batch = np.array(X_batch)
        
        try:
            if TENSORFLOW_AVAILABLE and hasattr(self.model, 'predict'):
                predictions = self.model.predict(X_batch, verbose=0)
            else:
                predictions = self.model.predict_proba(X_batch)
            
            # Process predictions
            batch_results = []
            for pred in predictions:
                predicted_id = np.argmax(pred)
                confidence = float(pred[predicted_id])
                predicted_name = self.mappings['id_to_syscall'][predicted_id]
                batch_results.append((predicted_name, confidence))
            
            # Merge results
            final_results = []
            batch_idx = 0
            for i in range(len(sequences)):
                if i in valid_indices:
                    final_results.append(batch_results[batch_idx])
                    batch_idx += 1
                else:
                    final_results.append((None, 0))
            
            return final_results
            
        except Exception as e:
            print(f"❌ Batch prediction error: {str(e)}")
            return [(None, 0)] * len(sequences)
    
    def predict_next_n(self, initial_sequence, n=5):
        """
        Predict next N system calls iteratively
        
        Args:
            initial_sequence: Starting sequence (list of syscall names)
            n: Number of future calls to predict
        
        Returns:
            List of dictionaries with 'syscall' and 'confidence'
        """
        if len(initial_sequence) < self.sequence_length:
            print(f"❌ Initial sequence too short. Need at least {self.sequence_length}")
            return []
        
        current_sequence = list(initial_sequence[-self.sequence_length:])
        predictions = []
        
        for i in range(n):
            predicted, confidence = self.predict(current_sequence)
            
            if predicted is None:
                break
            
            predictions.append({
                'step': i + 1,
                'syscall': predicted,
                'confidence': confidence
            })
            
            # Update sequence for next prediction
            current_sequence = current_sequence[1:] + [predicted]
        
        return predictions
    
    def get_top_k_predictions(self, syscall_names, k=5):
        """
        Get top K most likely next syscalls
        
        Args:
            syscall_names: Sequence of syscall names
            k: Number of top predictions to return
        
        Returns:
            List of (syscall, probability) tuples
        """
        if self.model is None or self.mappings is None:
            return []
        
        if len(syscall_names) != self.sequence_length:
            return []
        
        try:
            syscall_ids = [self.mappings['syscall_to_id'].get(name, 0) for name in syscall_names]
            X = np.array([syscall_ids])
            
            if TENSORFLOW_AVAILABLE and hasattr(self.model, 'predict'):
                predictions = self.model.predict(X, verbose=0)[0]
            else:
                predictions = self.model.predict_proba(X)[0]
            
            # Get top K indices
            top_k_indices = np.argsort(predictions)[-k:][::-1]
            
            results = []
            for idx in top_k_indices:
                syscall = self.mappings['id_to_syscall'][idx]
                prob = float(predictions[idx])
                results.append((syscall, prob))
            
            return results
            
        except Exception as e:
            print(f"❌ Error getting top-k: {str(e)}")
            return []


# Test if run directly
if __name__ == "__main__":
    print("=" * 60)
    print("SYSTEM CALL PREDICTION - INFERENCE TEST")
    print("=" * 60)
    
    # Initialize predictor
    predictor = SyscallPredictorInference()
    
    if predictor.model is None:
        print("\n❌ No model loaded. Please train the model first:")
        print("   python models/train.py")
        sys.exit(1)
    
    print("\n✅ Model loaded successfully!\n")
    
    # Test 1: Single prediction
    print("=" * 60)
    print("TEST 1: Single Prediction")
    print("=" * 60)
    
    test_sequence = ['open', 'fstat', 'read', 'write', 'close', 
                    'open', 'read', 'write', 'close', 'fstat']
    
    print(f"\n📝 Input sequence (last 10 syscalls):")
    for i, syscall in enumerate(test_sequence, 1):
        print(f"   {i:2d}. {syscall}")
    
    predicted, confidence = predictor.predict(test_sequence)
    
    if predicted:
        print(f"\n🎯 Prediction:")
        print(f"   Next syscall: {predicted}")
        print(f"   Confidence: {confidence*100:.2f}%")
    
    # Test 2: Top-K predictions
    print("\n" + "=" * 60)
    print("TEST 2: Top-5 Most Likely Next Syscalls")
    print("=" * 60 + "\n")
    
    top_k = predictor.get_top_k_predictions(test_sequence, k=5)
    
    for i, (syscall, prob) in enumerate(top_k, 1):
        bar = "█" * int(prob * 50)
        print(f"   {i}. {syscall:12s} {bar} {prob*100:5.2f}%")
    
    # Test 3: Predict sequence
    print("\n" + "=" * 60)
    print("TEST 3: Predict Next 5 Syscalls")
    print("=" * 60 + "\n")
    
    future = predictor.predict_next_n(test_sequence, n=5)
    
    for pred in future:
        bar = "█" * int(pred['confidence'] * 30)
        print(f"   Step {pred['step']}: {pred['syscall']:12s} {bar} ({pred['confidence']*100:.1f}%)")
    
    # Test 4: Batch prediction (for large datasets)
    print("\n" + "=" * 60)
    print("TEST 4: Batch Prediction (1000 sequences)")
    print("=" * 60 + "\n")
    
    import time
    
    # Generate 1000 test sequences
    sequences = []
    syscalls = list(predictor.mappings['syscall_to_id'].keys())
    
    for _ in range(1000):
        seq = [syscalls[i % len(syscalls)] for i in range(10)]
        sequences.append(seq)
    
    start_time = time.time()
    results = predictor.predict_batch(sequences)
    end_time = time.time()
    
    successful = sum(1 for r in results if r[0] is not None)
    
    print(f"   Total sequences: 1000")
    print(f"   Successful predictions: {successful}")
    print(f"   Time taken: {end_time - start_time:.3f} seconds")
    print(f"   Throughput: {1000 / (end_time - start_time):.1f} predictions/sec")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 60)