# Quick Start: Melvin in Linux VM

## You're in the VM! 🎉

### Step 1: Verify Environment

Run these commands in your VM shell:

```bash
# Check Linux kernel
uname -a

# Check IP address
ip addr show | grep 'inet ' | grep -v '127.0.0.1'

# Your VM IP should be: 192.168.64.2
```

### Step 2: Set Up Melvin

**Option A: Transfer from Mac (Recommended)**

From your Mac terminal:

```bash
cd ~/melvin_november/Melvin_november
./transfer_and_test.sh 192.168.64.2
```

This will:
- Install build tools
- Transfer all Melvin files
- Build and run the first test

**Option B: Manual Setup in VM**

In your VM shell:

```bash
# Install build tools
sudo apt update
sudo apt install -y build-essential gcc make git

# Create directory
mkdir -p ~/melvin_november
cd ~/melvin_november
```

Then from Mac, transfer files:

```bash
# From Mac
scp -r ~/melvin_november/Melvin_november/* ubuntu@192.168.64.2:~/melvin_november/
```

### Step 3: Build and Test

In the VM:

```bash
cd ~/melvin_november

# Build EXEC stub test
gcc -o test_exec_stub test_exec_stub.c -lm -std=c11

# Run test - should return 0x42 (not 0x40 like macOS!)
./test_exec_stub
```

### Step 4: Create and Test melvin.m Files

```bash
# Build the full test suite
gcc -o test_run_20min test_run_20min.c -lm -std=c11

# Run 20-minute stability test
./test_run_20min
```

## Expected Results

✅ **EXEC stub should return `0x42`** (proves machine code execution works)  
✅ **All tests should pass**  
✅ **Can create `.m` files and test them**  
✅ **Full EXEC capabilities enabled**

## Troubleshooting

### Can't transfer from Mac?
```bash
# In VM, install build tools first:
sudo apt update && sudo apt install -y build-essential

# Then from Mac:
rsync -avz ~/melvin_november/Melvin_november/ ubuntu@192.168.64.2:~/melvin_november/
```

### Build errors?
```bash
# Install dependencies
sudo apt install -y build-essential gcc make libc6-dev
```

### Need to exit VM?
```bash
exit  # or Ctrl+D
```

---

**You're all set!** Start testing Melvin's full capabilities on Linux! 🚀




