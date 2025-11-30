# Linux VM Setup Guide for Melvin Development

This guide will help you set up a Linux virtual machine on your Mac to run Melvin with full EXEC capabilities (which are restricted on macOS).

## Why a Linux VM?

macOS enforces W^X (Write XOR Execute) security, which prevents executing code from file-backed memory mappings. Linux allows this, so we can test Melvin's machine code execution features there.

## Quick Start

### Step 1: Install UTM

```bash
brew install --cask utm
```

Or download from: https://mac.getutm.app/

### Step 2: Download Ubuntu Server ISO

```bash
chmod +x setup_linux_vm.sh
./setup_linux_vm.sh
```

This will download Ubuntu Server 22.04 ARM64 (optimized for Apple Silicon).

### Step 3: Create VM in UTM

1. Open UTM from Applications
2. Click **"+"** or **"New"** to create a VM
3. Choose **"Virtualize"** (not Emulate)
4. Select **"Linux"**
5. Click **"Browse..."** and select the downloaded ISO:
   - Location: `~/melvin_linux_vm/ubuntu-22.04-server-arm64.iso`
6. Configure VM:
   - **RAM**: 4-8 GB (4096-8192 MB)
   - **CPU cores**: 4+ (allocate half your cores)
   - **Disk**: 20 GB minimum (40 GB recommended)
   - **Network**: Shared Network (NAT) - allows SSH from host
7. Click **"Save"** and start the VM

### Step 4: Install Ubuntu Server

1. Boot the VM
2. Follow the installation wizard:
   - Choose keyboard layout
   - Configure network (DHCP is fine)
   - **Install OpenSSH server** when prompted (important!)
   - Create a user account (remember the password)
   - Accept default disk partitioning
3. Wait for installation to complete
4. Reboot the VM

### Step 5: Find VM IP Address

After the VM reboots, you'll need its IP address to transfer files:

**Option A: From VM console**
```bash
ip addr show | grep 'inet ' | grep -v '127.0.0.1'
```

**Option B: From macOS host**
```bash
# UTM VMs typically appear as:
# 192.168.64.x or 10.0.2.x
# Check your network settings or UTM's network info
```

### Step 6: Transfer Melvin Codebase

From your Mac:

```bash
chmod +x transfer_melvin_to_vm.sh
./transfer_melvin_to_vm.sh <vm_ip> <vm_user>

# Example:
./transfer_melvin_to_vm.sh 192.168.64.2 ubuntu
```

### Step 7: Build and Test on Linux

SSH into the VM:

```bash
ssh ubuntu@<vm_ip>
```

Then inside the VM:

```bash
cd ~/melvin_november
chmod +x build_and_test.sh
./build_and_test.sh
```

Or manually:

```bash
gcc -o test_exec_stub test_exec_stub.c -lm -std=c11
./test_exec_stub
```

## Troubleshooting

### VM Won't Boot

- Make sure you selected **"Virtualize"** not **"Emulate"**
- Check that the ISO downloaded correctly
- Try a different Linux ISO (Ubuntu Desktop, Debian, etc.)

### Can't SSH to VM

1. Check that SSH server is installed:
   ```bash
   # Inside VM:
   sudo apt update
   sudo apt install openssh-server
   sudo systemctl start ssh
   sudo systemctl enable ssh
   ```

2. Check VM's IP address again
3. Verify network is set to "Shared" in UTM

### Transfer Script Fails

- Make sure `rsync` is installed on Mac: `brew install rsync`
- Or manually copy files using SCP:
  ```bash
  scp -r melvin.c melvin.h test_*.c ubuntu@<vm_ip>:~/melvin_november/
  ```

### Build Errors

Inside the VM, install build tools:

```bash
sudo apt update
sudo apt install build-essential gcc make
```

## Alternative: Docker

If you prefer containers over full VMs:

```bash
# On Mac
docker run -it -v $(pwd):/workspace ubuntu:22.04 bash

# Inside container:
apt update && apt install -y build-essential gcc make
cd /workspace
gcc -o test_exec_stub test_exec_stub.c -lm -std=c11
./test_exec_stub
```

## What to Test

Once the VM is running, you should be able to:

1. ✅ Run `test_exec_stub` - Should return `0x42` (not `0x40` like on macOS)
2. ✅ Run `test_run_20min` - 20-minute stability test
3. ✅ Run `test_pattern_reward` - Pattern learning with rewards
4. ✅ Run `test_exec_pattern_actor` - Full EXEC node functionality

## Performance Notes

- UTM uses QEMU under the hood - expect ~80-90% native performance
- Allocate enough RAM (4GB+ recommended)
- Network file sharing (NFS/SMB) can be faster than rsync for development
- Consider using shared folders if UTM supports them

## Next Steps

After the VM is set up:

1. Verify `test_exec_stub` returns `0x42` (EXEC works!)
2. Run the 20-minute stability test
3. Run the pattern reward test
4. Implement and test the full EXEC pattern actor

---

**Note**: This VM is specifically for testing Melvin's EXEC capabilities. For production deployment, consider:
- Native Linux systems
- Embedded Linux devices (Jetson, Raspberry Pi)
- Cloud Linux instances

