import os
import sys
import subprocess

def main():
    # Define paths to streamlit inside the local virtual environment
    is_windows = os.name == 'nt'
    if is_windows:
        streamlit_path = os.path.join("venv", "Scripts", "streamlit.exe")
    else:
        streamlit_path = os.path.join("venv", "bin", "streamlit")

    # If the venv streamlit is found, use it; otherwise fallback to the system command
    if os.path.exists(streamlit_path):
        cmd = [streamlit_path, "run", "app.py"]
    else:
        print("Warning: Local virtual environment (venv) not found at current directory.")
        print("Attempting to run using global/system 'streamlit'...")
        cmd = ["streamlit", "run", "app.py"]

    print(f"Launching dashboard: {' '.join(cmd)}")
    
    try:
        # Launch the streamlit dashboard subprocess
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("\n[ERROR] Streamlit could not be found.")
        print("Please make sure you have installed streamlit using:")
        print("  venv\\Scripts\\pip install streamlit")
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print("\nDashboard server stopped by user.")

if __name__ == '__main__':
    main()
