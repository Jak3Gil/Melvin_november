# Download Ubuntu Server ISO

The automatic download isn't working (Ubuntu's CDN redirects are tricky). Here's how to download it manually:

## Option 1: Direct Download (Recommended)

1. **Open your browser and go to:**
   ```
   https://ubuntu.com/download/server/arm
   ```

2. **Click "Download Ubuntu Server 22.04 LTS"** (ARM64 version)

3. **Save the file to:**
   ```
   ~/melvin_linux_vm/ubuntu-22.04-server-arm64.iso
   ```
   
   Or run this command to open the directory:
   ```bash
   open ~/melvin_linux_vm/
   ```
   Then drag the downloaded ISO into that folder.

4. **Rename it** to: `ubuntu-22.04-server-arm64.iso` if it has a different name

## Option 2: Use Terminal (Alternative Mirror)

Try this direct download command:

```bash
cd ~/melvin_linux_vm

# Try this mirror
curl -L -C - -o ubuntu-22.04-server-arm64.iso \
  "https://mirror.arizona.edu/ubuntu-releases/22.04/ubuntu-22.04.4-live-server-arm64.iso"
```

Or use `wget` if you have it:

```bash
cd ~/melvin_linux_vm
wget https://mirror.arizona.edu/ubuntu-releases/22.04/ubuntu-22.04.4-live-server-arm64.iso \
  -O ubuntu-22.04-server-arm64.iso
```

## Option 3: Use UTM's Built-in Download

UTM can download ISOs automatically:

1. Open UTM
2. Click "+" to create new VM
3. Choose "Virtualize" → "Linux"
4. Instead of browsing, click "Download Ubuntu Server 22.04"
5. UTM will download it automatically

## Verify Download

After downloading, verify the file:

```bash
cd ~/melvin_linux_vm
ls -lh ubuntu-22.04-server-arm64.iso

# Should be ~1.2 GB
# If it's only a few hundred bytes, the download failed
```

## After Download

Once the ISO is in `~/melvin_linux_vm/`, you can:

1. Follow the steps in `CREATE_VM_STEPS.md` to create the VM
2. Or run `./setup_linux_vm.sh` again (it will detect the ISO is already there)

