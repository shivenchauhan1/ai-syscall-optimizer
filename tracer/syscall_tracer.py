"""
System Call Tracer Module
This module captures system calls from running processes
"""

import subprocess
import os
import time
from datetime import datetime

class SystemCallTracer:
    def __init__(self, output_dir="data/raw_traces"):
        """Initialize the tracer"""
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def trace_command(self, command, trace_name="trace"):
        """
        Trace system calls for a given command
        
        Args:
            command: Command to trace (e.g., "ls -la")
            trace_name: Name for the trace file
        
        Returns:
            Path to the trace file
        """
        print(f"\n🔍 Starting trace for command: {command}")
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"{trace_name}_{timestamp}.txt")
        
        try:
            # For Windows (use different approach)
            if os.name == 'nt':
                print("⚠️  Note: Full strace not available on Windows")
                print("Running command and collecting basic info...")
                
                # Run command and time it
                start_time = time.time()
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                end_time = time.time()
                
                # Save basic info
                with open(output_file, 'w') as f:
                    f.write(f"Command: {command}\n")
                    f.write(f"Execution Time: {end_time - start_time:.4f} seconds\n")
                    f.write(f"Return Code: {result.returncode}\n")
                    f.write(f"\nStdout:\n{result.stdout}\n")
                    f.write(f"\nStderr:\n{result.stderr}\n")
                
                print(f"✅ Basic trace saved to: {output_file}")
                
            # For Linux/Mac (use strace/dtrace)
            else:
                # Try strace first (Linux)
                strace_cmd = f"strace -o {output_file} -tt -T {command}"
                
                result = subprocess.run(
                    strace_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                print(f"✅ Trace saved to: {output_file}")
            
            return output_file
            
        except subprocess.TimeoutExpired:
            print("❌ Command timeout (30 seconds exceeded)")
            return None
        except FileNotFoundError:
            print("❌ strace not found. Please install it:")
            print("   Linux: sudo apt-get install strace")
            print("   Mac: Use dtrace (built-in) or install strace")
            return None
        except Exception as e:
            print(f"❌ Error during tracing: {str(e)}")
            return None
    
    def trace_multiple_commands(self, commands):
        """
        Trace multiple commands
        
        Args:
            commands: List of command strings
        
        Returns:
            List of trace file paths
        """
        trace_files = []
        
        for i, cmd in enumerate(commands, 1):
            print(f"\n📝 Tracing command {i}/{len(commands)}")
            trace_file = self.trace_command(cmd, f"trace_{i}")
            if trace_file:
                trace_files.append(trace_file)
            time.sleep(1)  # Small delay between commands
        
        return trace_files
    
    def generate_sample_traces(self):
        """Generate sample traces for testing"""
        print("\n🎯 Generating sample system call traces...")
        
        # Sample commands that make various system calls
        sample_commands = [
            "echo Hello World",
            "ls",
            "pwd",
            "date",
            "whoami"
        ]
        
        trace_files = self.trace_multiple_commands(sample_commands)
        
        print(f"\n✅ Generated {len(trace_files)} trace files")
        return trace_files


# Test the tracer if run directly
if __name__ == "__main__":
    print("=" * 60)
    print("SYSTEM CALL TRACER - TEST MODE")
    print("=" * 60)
    
    # Create tracer instance
    tracer = SystemCallTracer()
    
    # Generate sample traces
    tracer.generate_sample_traces()
    
    print("\n✅ Tracer test completed!")
    print(f"Check the 'data/raw_traces' folder for trace files")