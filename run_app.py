#!/usr/bin/env python3
"""
Voice Agent App Runner
Starts both backend and frontend services
"""

import os
import sys
import time
import subprocess
import threading
import signal
import atexit

# Global process handles
backend_process = None
frontend_process = None

def start_backend():
    """Start the backend server"""
    global backend_process
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    print("🚀 Starting Voice Agent Backend...")
    
    # Use virtual environment if it exists
    python_cmd = "python3"
    if os.path.exists("env/bin/python"):
        python_cmd = "env/bin/python"
    elif os.path.exists("venv/bin/python"):
        python_cmd = "venv/bin/python"
    
    backend_process = subprocess.Popen(
        [python_cmd, "-m", "backend.app.main"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Start thread to read output
    def read_backend_output():
        for line in backend_process.stdout:
            print(f"\033[34m[BACKEND]\033[0m {line.strip()}")
    
    threading.Thread(target=read_backend_output, daemon=True).start()
    time.sleep(2)  # Give backend time to start

def start_frontend():
    """Start the frontend development server"""
    global frontend_process
    
    print("🚀 Starting Voice Agent Frontend...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Start thread to read output
    def read_frontend_output():
        for line in frontend_process.stdout:
            print(f"\033[32m[FRONTEND]\033[0m {line.strip()}")
    
    threading.Thread(target=read_frontend_output, daemon=True).start()

def cleanup():
    """Clean up processes on exit"""
    print("\n🛑 Shutting down Voice Agent...")
    
    if backend_process:
        print("Stopping backend...")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
    
    if frontend_process:
        print("Stopping frontend...")
        frontend_process.terminate()
        try:
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_process.kill()
    
    print("✅ Shutdown complete")

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    cleanup()
    sys.exit(0)

if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start services
    start_backend()
    start_frontend()
    
    print("\n✅ Voice Agent is running!")
    print("   Backend API: http://localhost:8000")
    print("   Frontend UI: http://localhost:3000 (or next available port)")
    print("\nPress Ctrl+C to stop all services\n")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass  # Cleanup will be handled by signal handler 