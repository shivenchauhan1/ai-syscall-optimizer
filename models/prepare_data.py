"""
Data Preparation for ML Model
Prepares system call sequences for training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

class DataPreparator:
    def __init__(self, sequence_length=10):
        """
        Initialize data preparator
        
        Args:
            sequence_length: Number of syscalls in each sequence
        """
        self.sequence_length = sequence_length
        self.label_encoder = LabelEncoder()
        self.syscall_to_id = {}
        self.id_to_syscall = {}
    
    def load_data(self, csv_file="data/processed/parsed_syscalls.csv"):
        """Load parsed system call data"""
        print(f"\n📂 Loading data from: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(df)} system calls")
            return df
        except FileNotFoundError:
            print(f"❌ File not found: {csv_file}")
            print("   Run tracer and parser first!")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def create_sequences(self, df):
        """
        Create sequences of system calls for training
        
        Args:
            df: DataFrame with system calls
        
        Returns:
            X (sequences), y (next syscall)
        """
        print(f"\n🔄 Creating sequences (length={self.sequence_length})...")
        
        # Get system call names
        syscalls = df['syscall'].values
        
        # Encode syscalls as numbers
        encoded_syscalls = self.label_encoder.fit_transform(syscalls)
        
        # Create mappings
        self.syscall_to_id = {name: idx for idx, name in 
                             enumerate(self.label_encoder.classes_)}
        self.id_to_syscall = {idx: name for name, idx in 
                             self.syscall_to_id.items()}
        
        print(f"📝 Found {len(self.syscall_to_id)} unique system calls")
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(encoded_syscalls) - self.sequence_length):
            # Input: sequence of syscalls
            sequence = encoded_syscalls[i:i + self.sequence_length]
            # Output: next syscall
            next_syscall = encoded_syscalls[i + self.sequence_length]
            
            X.append(sequence)
            y.append(next_syscall)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Created {len(X)} sequences")
        print(f"   Input shape: {X.shape}")
        print(f"   Output shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into training and testing sets"""
        print(f"\n✂️  Splitting data (test size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"✅ Training samples: {len(X_train)}")
        print(f"✅ Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_encoders(self, output_dir="models/saved_models"):
        """Save label encoders for later use"""
        os.makedirs(output_dir, exist_ok=True)
        
        encoder_path = os.path.join(output_dir, "label_encoder.pkl")
        mappings_path = os.path.join(output_dir, "syscall_mappings.pkl")
        
        # Save encoder
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save mappings
        mappings = {
            'syscall_to_id': self.syscall_to_id,
            'id_to_syscall': self.id_to_syscall,
            'n_syscalls': len(self.syscall_to_id)
        }
        
        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings, f)
        
        print(f"\n💾 Saved encoders to: {output_dir}")
    
    def prepare_all(self):
        """Complete data preparation pipeline"""
        print("=" * 60)
        print("DATA PREPARATION PIPELINE")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        if df is None:
            return None
        
        # Create sequences
        X, y = self.create_sequences(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Save encoders
        self.save_encoders()
        
        print("\n✅ Data preparation completed!")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'n_syscalls': len(self.syscall_to_id)
        }


# Test if run directly
if __name__ == "__main__":
    preparator = DataPreparator(sequence_length=10)
    data = preparator.prepare_all()
    
    if data:
        print("\n📊 Data Summary:")
        print(f"   Training sequences: {len(data['X_train'])}")
        print(f"   Testing sequences: {len(data['X_test'])}")
        print(f"   Unique syscalls: {data['n_syscalls']}")