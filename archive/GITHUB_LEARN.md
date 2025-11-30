# GitHub Auto-Learning System

Melvin can now automatically learn from GitHub repositories! Just give him a URL and he'll clone, parse, and learn from it.

## How to Use

### Method 1: Add URLs to File

Edit `github_urls.txt` and add one URL per line:

```
https://github.com/user/repo.git
https://github.com/another/project.git
```

Melvin will automatically process these when running.

### Method 2: Pipe URLs via Stdin

```bash
echo "https://github.com/user/repo.git" | ./melvin
```

Or multiple URLs:
```bash
cat urls.txt | ./melvin
```

### Method 3: Enter URLs While Running

Just type GitHub URLs into Melvin's stdin while it's running. It will detect them and queue them for cloning.

## How It Works

1. **Auto-Detection**: Melvin monitors:
   - `github_urls.txt` file (reads on startup and checks periodically)
   - stdin (non-blocking, detects URLs in real-time)

2. **Auto-Clone**: When a URL is detected:
   - Extracts repository name
   - Clones to `ingested_repos/<repo-name>/`
   - Skips if already cloned

3. **Auto-Parse**: After cloning:
   - Automatically triggers parsing
   - Scans all `.c`, `.cpp`, `.h`, `.hpp` files
   - Extracts functions, structures, relationships
   - Stores in `melvin.m` as nodes and edges

4. **Auto-Learn**: The knowledge becomes part of Melvin's graph-based brain in `melvin.m`

## Example

```bash
# Add URL to file
echo "https://github.com/Jak3Gil/melvin-unified-brain.git" >> github_urls.txt

# Run Melvin - he'll automatically clone and learn!
./melvin
```

Or give him a URL directly:
```bash
echo "https://github.com/user/cool-project.git" | ./melvin
```

Melvin will:
1. ✅ Detect the URL
2. ✅ Clone the repository
3. ✅ Parse all code files
4. ✅ Store knowledge in melvin.m
5. ✅ Ready to use that knowledge!

## Notes

- Melvin remembers what he's already cloned (won't duplicate)
- URLs can be full GitHub URLs or just `github.com/user/repo`
- The `.git` extension is added automatically if missing
- All repositories are cloned to `ingested_repos/` directory

