"""
Main Orchestrator Script
Runs the complete AI-Enhanced System Call Optimization pipeline
"""

import os
import sys
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    result = os.system(command)
    if result != 0:
        print(f"⚠️  Warning: Command returned code {result}")
    return result

def main():
    """Main execution pipeline"""
    print_header("AI-ENHANCED SYSTEM CALL OPTIMIZATION")
    print("Complete Pipeline Execution\n")
    print("This will:")
    print("  1. Collect system call traces")
    print("  2. Parse and process data")
    print("  3. Train ML model")
    print("  4. Launch dashboard")
    print("\nPress Ctrl+C at any time to stop\n")
    
    input("Press Enter to start...")
    
    # Step 1: Data Collection
    print_header("STEP 1: SYSTEM CALL TRACING")
    
    try:
        from tracer.syscall_tracer import SystemCallTracer
        
        tracer = SystemCallTracer()
        trace_files = tracer.generate_sample_traces()
        
        if not trace_files:
            print("❌ Failed to generate traces")
            return
        
        print(f"\n✅ Generated {len(trace_files)} trace files")
        time.sleep(2)
    except Exception as e:
        print(f"❌ Error in Step 1: {str(e)}")
        return
    
    # Step 2: Data Parsing
    print_header("STEP 2: PARSING TRACES")
    
    try:
        from tracer.parser import TraceParser
        
        parser = TraceParser()
        df = parser.parse_multiple_files(trace_files)
        
        if df.empty:
            print("❌ No data parsed")
            return
        
        parser.save_parsed_data(df)
        parser.get_statistics(df)
        time.sleep(2)
    except Exception as e:
        print(f"❌ Error in Step 2: {str(e)}")
        return
    
    # Step 3: Model Training
    print_header("STEP 3: TRAINING ML MODEL")
    print("Training AI prediction model...")
    print("This may take 5-10 minutes...\n")
    
    try:
        result = run_command("python models/train.py", "Training model")
        if result == 0:
            print("\n✅ Model training completed!")
        time.sleep(2)
    except Exception as e:
        print(f"❌ Error in Step 3: {str(e)}")
        print("Continuing anyway...")
    
    # Step 4: Dashboard
    print_header("STEP 4: LAUNCHING DASHBOARD")
    print("Starting interactive dashboard...\n")
    print("🌐 Dashboard will open in your browser at: http://localhost:8501")
    print("📊 Explore all tabs: Overview, Predictions, Performance, Optimization\n")
    print("⚠️  Press Ctrl+C in this terminal to stop the dashboard\n")
    
    input("Press Enter to launch dashboard...")
    
    try:
        os.system("streamlit run dashboard/app.py")
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Pipeline stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()