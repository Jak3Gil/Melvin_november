# Step-by-Step: Create Linux VM in UTM

Follow these steps to create your Linux VM for Melvin development:

## Step 1: Open UTM

1. Open **UTM** from Applications (or press `Cmd+Space` and type "UTM")
2. If you see a welcome screen, click **"Create a New Virtual Machine"** or the **"+"** button

## Step 2: Choose Virtualization Type

1. Select **"Virtualize"** (NOT "Emulate")
   - This uses native virtualization (faster)
   - "Emulate" is for running x86 on ARM (much slower)

## Step 3: Select Operating System

1. Choose **"Linux"**

## Step 4: Select Installation Media

1. Click **"Browse..."**
2. Navigate to: `/Users/jakegilbert/melvin_linux_vm/`
3. Select: `ubuntu-24.04.3-live-server-arm64.iso` (or `ubuntu-22.04-server-arm64.iso` if you prefer 22.04)
4. Click **"Open"**

## Step 5: Configure Hardware

### Memory (RAM)
- **Recommended**: 4096 MB (4 GB) minimum
- **Better**: 8192 MB (8 GB) if you have 16+ GB total
- Move the slider or type in the box

### CPU Cores
- **Recommended**: 4 cores (half your available cores)
- You can allocate more if you have a powerful Mac

### Storage
- **Minimum**: 20 GB
- **Recommended**: 40 GB (gives room for development)
- Click **"New Drive"** if needed

### Network
- Select **"Shared Network"** (NAT)
- This allows the VM to access the internet and your Mac to SSH into it

## Step 6: Review and Create

1. Review your settings
2. Give your VM a name (e.g., "Melvin Linux")
3. Click **"Save"** or **"Create"**

## Step 7: Start the VM

1. Select your new VM in the UTM window
2. Click the **Play** button (▶️) or press `Cmd+R`
3. The VM will boot from the Ubuntu ISO

## Step 8: Install Ubuntu Server

Follow the installation wizard:

1. **Keyboard Layout**: Choose your layout (default is usually fine)

2. **Network**: 
   - Accept DHCP settings (automatic)

3. **Proxy**: 
   - Leave blank (unless you use one)

4. **Mirror**: 
   - Use default Ubuntu archive mirror

5. **Storage**: 
   - Use entire disk with guided setup
   - Confirm the disk selection

6. **Profile Setup**:
   - **Your name**: `melvin` (or your preference)
   - **Server name**: `melvin-vm` (or your preference)
   - **Username**: `ubuntu` (recommended - matches our scripts)
   - **Password**: Choose a strong password (you'll need this for SSH)
   - Confirm password

7. **SSH Setup** (IMPORTANT!):
   - ✅ **CHECK** "Install OpenSSH server"
   - This allows you to SSH into the VM from your Mac

8. **Snaps**: 
   - Select any you want, or skip

9. **Installation**:
   - Wait for installation to complete (5-10 minutes)

10. **Reboot**:
    - When prompted, click "Reboot Now"
    - The VM will reboot into Ubuntu

## Step 9: Find Your VM's IP Address

After the VM reboots:

1. Login with your username and password
2. Run:
   ```bash
   ip addr show | grep 'inet ' | grep -v '127.0.0.1'
   ```
3. Note the IP address (usually starts with `192.168.64.` or `10.0.2.`)

**Example output:**
```
inet 192.168.64.3/24 brd 192.168.64.255 scope global dynamic enp0s1
```
Your IP is: `192.168.64.3`

## Step 10: Transfer Melvin Codebase

From your Mac terminal:

```bash
cd ~/melvin_november/Melvin_november
./transfer_melvin_to_vm.sh <vm_ip_address> ubuntu
```

**Example:**
```bash
./transfer_melvin_to_vm.sh 192.168.64.3 ubuntu
```

This will:
- Install build tools on the VM
- Transfer all Melvin source files
- Set up the build environment

## Step 11: Test Melvin on Linux

SSH into your VM:

```bash
ssh ubuntu@<vm_ip_address>
```

Then inside the VM:

```bash
cd ~/melvin_november
chmod +x build_and_test.sh
./build_and_test.sh
```

Or test the EXEC stub directly:

```bash
gcc -o test_exec_stub test_exec_stub.c -lm -std=c11
./test_exec_stub
```

**Expected Result**: You should see `0x42` (not `0x40` like on macOS) - this proves EXEC works!

## Troubleshooting

### VM Won't Start
- Make sure you selected **"Virtualize"** not **"Emulate"**
- Check that virtualization is enabled in your Mac's settings
- Try restarting UTM

### Can't Find IP Address
- Make sure the VM is fully booted (logged in)
- Try: `hostname -I` (shows IP addresses)
- Check UTM's network settings

### SSH Connection Refused
- Make sure OpenSSH server was installed during setup
- Inside VM, check: `sudo systemctl status ssh`
- If not running: `sudo systemctl start ssh`

### Transfer Script Fails
- Make sure `rsync` is installed on Mac: `brew install rsync`
- Or manually use `scp`:
  ```bash
  scp -r melvin.c melvin.h test_*.c ubuntu@<vm_ip>:~/melvin_november/
  ```

## Next Steps

Once your VM is running:

1. ✅ Verify `test_exec_stub` returns `0x42`
2. ✅ Run the 20-minute stability test
3. ✅ Test pattern learning with rewards
4. ✅ Implement full EXEC pattern actor

---

**You're all set!** Your Linux VM is ready for full Melvin development. 🚀

