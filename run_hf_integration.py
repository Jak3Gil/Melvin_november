#!/usr/bin/env python3
"""
🚀 QUICK RUNNER FOR HUGGING FACE INTEGRATION
===========================================
Simple script to run Hugging Face integration with Melvin brain.
Use this on your Jetson device via COM8/PuTTY.
"""

import sys
import os
import subprocess
import time

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['transformers', 'datasets', 'torch', 'cv2', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(missing_packages):
    """Install missing packages"""
    if not missing_packages:
        return True
        
    print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
    
    try:
        # Install requirements file
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_hf.txt'])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def main():
    print("🤗 MELVIN + HUGGING FACE INTEGRATION")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists('melvin_global_brain.py'):
        print("❌ Error: melvin_global_brain.py not found!")
        print("   Make sure you're running this from the melvin-unified-brain directory")
        return 1
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n📦 Missing packages detected: {missing}")
        print("🔧 Installing dependencies...")
        
        if not install_missing_packages(missing):
            print("❌ Failed to install dependencies. Please install manually:")
            print("   pip install -r requirements_hf.txt")
            return 1
    
    print("\n✅ All dependencies available!")
    
    # Import and run integration
    try:
        print("🚀 Starting Hugging Face integration...")
        from huggingface_integration import main as hf_main
        return hf_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure melvin_global_brain.py is in the current directory")
        return 1
    except Exception as e:
        print(f"❌ Error during integration: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
