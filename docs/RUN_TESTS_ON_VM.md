# Running Theoretical Tests on Linux VM

Since macOS blocks RWX memory, these tests need to run on a Linux VM where RWX permissions work.

## Option 1: If Already SSH'd into VM

If you're already connected to the VM, just run:

```bash
cd ~/melvin_november
./run_theoretical_tests.sh
```

## Option 2: Transfer Files and Run

### Step 1: Make sure SSH is enabled on VM

Inside the VM terminal:
```bash
sudo systemctl start ssh
sudo systemctl enable ssh
```

### Step 2: Find VM IP (if needed)

Inside the VM terminal:
```bash
ip addr show | grep 'inet ' | grep -v '127.0.0.1'
```

### Step 3: Transfer files from Mac

From your Mac terminal:
```bash
# Transfer test files
scp test_self_modify.c test_code_evolution.c test_auto_exec.c test_meta_learning.c test_emergent_algo.c run_theoretical_tests.sh melvin.c melvin.h ubuntu@192.168.64.2:~/melvin_november/
```

Or use the automated script:
```bash
./run_theoretical_tests_on_vm.sh 192.168.64.2 ubuntu
```

### Step 4: Run tests on VM

SSH into VM:
```bash
ssh ubuntu@192.168.64.2
```

Then inside VM:
```bash
cd ~/melvin_november
chmod +x run_theoretical_tests.sh
./run_theoretical_tests.sh
```

## Option 3: Manual Transfer (if SSH not working)

If SSH isn't set up, you can:

1. Copy files manually through shared folder (if VM has one)
2. Or set up SSH first:
   ```bash
   # In VM
   sudo apt update
   sudo apt install openssh-server
   sudo systemctl start ssh
   sudo systemctl enable ssh
   ```

## Expected Results on Linux VM

On Linux, you should see:
- ✅ RWX permissions work (no "Permission denied" errors)
- ✅ EXEC nodes can execute properly
- ✅ More tests may pass than on macOS

The tests will show which theoretical capabilities actually work!

