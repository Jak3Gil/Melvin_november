# 🚀 Git LFS Setup Guide for Melvin Repository

## 🎯 What is Git LFS?

Git LFS (Large File Storage) is a Git extension that handles large files efficiently by storing them on a separate server and only downloading them when needed. This is perfect for your repository which contains large database files.

## 📊 Why Use Git LFS?

Your repository currently has:
- `global_memory.db` (~92MB) - Too large for regular Git
- Potential future large files (datasets, models, etc.)
- Better performance and storage efficiency

## 🔧 Installation Methods

### **Method 1: Using Homebrew (Recommended)**
```bash
# Install Homebrew first if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Git LFS
brew install git-lfs
```

### **Method 2: Direct Download**
```bash
# Download Git LFS for macOS
curl -L -o git-lfs.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v3.4.0/git-lfs-darwin-amd64-v3.4.0.tar.gz

# Extract and install
tar -xzf git-lfs.tar.gz
cd git-lfs-3.4.0
sudo ./install.sh
```

### **Method 3: Using Package Managers**
```bash
# For macOS with MacPorts
sudo port install git-lfs

# For Ubuntu/Debian
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

## 🚀 Setup Git LFS in Your Repository

### **Step 1: Install Git LFS**
```bash
# Check if Git LFS is installed
git lfs version

# If not installed, use one of the methods above
```

### **Step 2: Initialize Git LFS**
```bash
# Navigate to your repository
cd /Users/jakegilbert/melvin\ Repo/melvin-unified-brain

# Initialize Git LFS
git lfs install
```

### **Step 3: Track Large Files**
```bash
# Track database files
git lfs track "*.db"
git lfs track "*.sqlite"
git lfs track "*.sqlite3"

# Track large data files
git lfs track "*.csv"
git lfs track "*.json"
git lfs track "*.parquet"
git lfs track "*.h5"
git lfs track "*.hdf5"

# Track model files
git lfs track "*.pkl"
git lfs track "*.pickle"
git lfs track "*.joblib"
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.onnx"

# Track image/video files
git lfs track "*.jpg"
git lfs track "*.jpeg"
git lfs track "*.png"
git lfs track "*.gif"
git lfs track "*.mp4"
git lfs track "*.avi"
git lfs track "*.mov"

# Track audio files
git lfs track "*.mp3"
git lfs track "*.wav"
git lfs track "*.flac"
git lfs track "*.aac"

# Track compressed files
git lfs track "*.zip"
git lfs track "*.tar.gz"
git lfs track "*.rar"
git lfs track "*.7z"
```

### **Step 4: Add .gitattributes**
```bash
# Add the .gitattributes file to Git
git add .gitattributes
git commit -m "Add Git LFS tracking for large files"
```

### **Step 5: Migrate Existing Large Files**
```bash
# Remove large files from Git history (if needed)
git lfs migrate import --include="*.db,*.sqlite,*.csv,*.json"

# Or manually track existing files
git lfs track "melvin_global_memory/global_memory.db"
git add .gitattributes
git add melvin_global_memory/global_memory.db
git commit -m "Track large database file with Git LFS"
```

## 📋 Git LFS Commands Reference

### **Basic Commands**
```bash
# Check Git LFS status
git lfs status

# List tracked files
git lfs ls-files

# Pull LFS files
git lfs pull

# Push LFS files
git lfs push origin main

# Fetch LFS files
git lfs fetch --all
```

### **File Management**
```bash
# Track new file types
git lfs track "*.extension"

# Untrack file types
git lfs untrack "*.extension"

# Track specific files
git lfs track "path/to/specific/file.db"

# Check which files are tracked
git lfs track
```

### **Migration Commands**
```bash
# Import existing large files to LFS
git lfs migrate import --include="*.db,*.csv"

# Export LFS files back to regular Git
git lfs migrate export --include="*.db,*.csv"

# Check what would be migrated
git lfs migrate info --include="*.db,*.csv"
```

## 🔧 Configuration

### **Git LFS Configuration**
```bash
# Set LFS endpoint (if using custom server)
git config lfs.url "https://your-lfs-server.com"

# Set batch size for transfers
git config lfs.batchsize 100

# Set transfer mode
git config lfs.transfer.mode basic
```

### **Repository Configuration**
```bash
# Enable LFS for this repository
git lfs install

# Check LFS hooks
git lfs install --skip-smudge
```

## 📊 Best Practices

### **1. File Size Limits**
- **Git LFS**: Files > 50MB
- **Regular Git**: Files < 50MB
- **Consider LFS**: Files > 10MB (for performance)

### **2. File Types to Track**
```bash
# Database files
*.db, *.sqlite, *.sqlite3

# Data files
*.csv, *.json, *.parquet, *.h5

# Model files
*.pkl, *.pickle, *.pt, *.pth, *.onnx

# Media files
*.jpg, *.png, *.mp4, *.mp3

# Compressed files
*.zip, *.tar.gz, *.rar
```

### **3. Performance Tips**
```bash
# Pull only needed files
git lfs pull --include="*.db"

# Use batch operations
git lfs push --all

# Clean up unused LFS files
git lfs prune
```

## 🚨 Troubleshooting

### **Common Issues**

**Issue**: "git: 'lfs' is not a git command"
```bash
# Solution: Install Git LFS
brew install git-lfs  # macOS
sudo apt-get install git-lfs  # Ubuntu
```

**Issue**: Large files still in regular Git
```bash
# Solution: Migrate to LFS
git lfs migrate import --include="*.db"
git push --force-with-lease
```

**Issue**: LFS files not downloading
```bash
# Solution: Pull LFS files
git lfs pull
git lfs fetch --all
```

**Issue**: Slow LFS operations
```bash
# Solution: Configure batch size
git config lfs.batchsize 100
git config lfs.transfer.mode basic
```

## 📈 Benefits for Your Repository

### **Immediate Benefits**
1. **No more GitHub warnings** about large files
2. **Faster clone/pull operations**
3. **Better repository performance**
4. **Reduced storage costs**

### **Future Benefits**
1. **Scalable for large datasets**
2. **Efficient model file storage**
3. **Better collaboration**
4. **Professional repository management**

## 🎯 Next Steps

1. **Install Git LFS** using one of the methods above
2. **Initialize LFS** in your repository
3. **Track large files** (especially `global_memory.db`)
4. **Migrate existing files** to LFS
5. **Update documentation** to reflect LFS usage

## 📚 Additional Resources

- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub LFS Guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [Git LFS Best Practices](https://github.com/git-lfs/git-lfs/wiki/Best-Practices)

---

**Ready to optimize your repository with Git LFS! 🚀**
