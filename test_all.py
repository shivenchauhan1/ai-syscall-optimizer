"""
Quick Test Script
Tests all modules quickly
"""

import os

print("=" * 60)
print("TESTING ALL MODULES")
print("=" * 60)

# Test 1: Tracer
print("\n[1/4] Testing Tracer...")
result = os.system("python tracer/syscall_tracer.py")
print("✅ Tracer OK" if result == 0 else "❌ Tracer FAILED")

# Test 2: Parser
print("\n[2/4] Testing Parser...")
result = os.system("python tracer/parser.py")
print("✅ Parser OK" if result == 0 else "❌ Parser FAILED")

# Test 3: Data Preparation
print("\n[3/4] Testing Data Preparation...")
result = os.system("python models/prepare_data.py")
print("✅ Data Prep OK" if result == 0 else "❌ Data Prep FAILED")

# Test 4: Model Training
print("\n[4/4] Testing Model Training...")
result = os.system("python models/train.py")
print("✅ Training OK" if result == 0 else "❌ Training FAILED")

print("\n" + "=" * 60)
print("TESTING COMPLETE!")
print("=" * 60)
print("\nIf all tests passed, run:")
print("  streamlit run dashboard/app.py")