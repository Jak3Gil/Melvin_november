#!/bin/bash
# Script to run INSIDE the VM after Ubuntu is installed
# Copy this to the VM and run it

echo "Installing SDK Manager in VM..."

sudo apt update
sudo apt upgrade -y

# Install SDK Manager
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install sdkmanager

echo "SDK Manager installed!"
echo "Run: sdkmanager"
echo "Then flash JetPack 6.x on your Jetson"

