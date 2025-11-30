# 🚀 Linux VM Setup for Melvin - Complete Guide

## ✅ What's Ready

1. **UTM is installed** - Virtualization software ready
2. **Setup scripts created** - Automated VM management
3. **VM directory created** - `~/melvin_linux_vm/`
4. **Browser opened** - Ubuntu download page

## 📥 Step 1: Download Ubuntu ISO

**Download the Ubuntu Server ARM64 ISO:**

1. In the browser that just opened, click **"Download Ubuntu Server 24.04 LTS"** (ARM64 version)
   - Or visit: https://ubuntu.com/download/server/arm

2. When the download starts, **save it to:**
   ```
   ~/melvin_linux_vm/ubuntu-24.04.3-live-server-arm64.iso
   ```
   (Or use 22.04 if you prefer: `ubuntu-22.04-server-arm64.iso`)
   (The Finder window is already open at that location)

3. **Wait for download** (~1.2 GB, may take a few minutes)

4. **Verify** the file is ~1.2 GB (not just a few hundred bytes)

## 🖥️ Step 2: Create VM in UTM

**Follow these steps (detailed guide in `CREATE_VM_STEPS.md`):**

1. **Open UTM** (Applications → UTM)

2. **Click "+"** to create new VM

3. **Select "Virtualize"** (NOT Emulate)

4. **Choose "Linux"**

5. **Browse for ISO**: Select `~/melvin_linux_vm/ubuntu-22.04-server-arm64.iso`

6. **Configure VM:**
   - RAM: **4-8 GB** (4096-8192 MB)
   - CPU: **4 cores**
   - Disk: **20-40 GB**
   - Network: **Shared Network (NAT)**

7. **Start VM** and install Ubuntu:
   - Create user: `ubuntu` (recommended)
   - **IMPORTANT**: Check "Install OpenSSH server" during setup

## 📡 Step 3: Find VM IP Address

After Ubuntu is installed and rebooted:

1. Login to the VM
2. Run: `ip addr show | grep 'inet ' | grep -v '127.0.0.1'`
3. Note the IP (usually `192.168.64.x`)

## 📦 Step 4: Transfer Melvin Code

From your Mac terminal:

```bash
cd ~/melvin_november/Melvin_november
./transfer_melvin_to_vm.sh <vm_ip> ubuntu

# Example:
./transfer_melvin_to_vm.sh 192.168.64.3 ubuntu
```

This will automatically:
- Install build tools on the VM
- Transfer all Melvin source files
- Set up the development environment

## 🧪 Step 5: Test Melvin on Linux

SSH into your VM:

```bash
ssh ubuntu@<vm_ip>
```

Then run:

```bash
cd ~/melvin_november
./build_and_test.sh
```

**Expected Result**: `test_exec_stub` should return `0x42` (execution works!) instead of `0x40` (blocked on macOS).

## 📚 Documentation

- **`CREATE_VM_STEPS.md`** - Detailed step-by-step VM creation
- **`ISO_DOWNLOAD.md`** - Alternative download methods
- **`VM_SETUP_GUIDE.md`** - Complete reference guide
- **`transfer_melvin_to_vm.sh`** - Transfer script
- **`build_and_test.sh`** - Build/test script for Linux

## 🎯 Quick Commands Reference

```bash
# Download ISO (if automatic failed)
open https://ubuntu.com/download/server/arm

# Transfer Melvin to VM
./transfer_melvin_to_vm.sh <vm_ip> ubuntu

# SSH into VM
ssh ubuntu@<vm_ip>

# Build and test on VM
cd ~/melvin_november && ./build_and_test.sh
```

## 🔧 Troubleshooting

### ISO download failed?
- See `ISO_DOWNLOAD.md` for alternative methods
- Or use UTM's built-in download feature

### Can't SSH to VM?
- Make sure OpenSSH server was installed during Ubuntu setup
- Check: `sudo systemctl status ssh` (inside VM)

### Transfer script fails?
- Install rsync: `brew install rsync`
- Or manually copy files with `scp`

## 🎉 What You'll Achieve

Once the VM is running:
- ✅ Full EXEC capabilities (machine code execution)
- ✅ All Melvin tests working
- ✅ Pattern learning with rewards
- ✅ EXEC pattern actor implementation
- ✅ Complete development environment

---

**Ready to go!** Download the ISO, create the VM, and you're all set. 🚀

