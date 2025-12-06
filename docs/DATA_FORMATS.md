# Data Formats & Input Guide

## Core Principle: **Raw Bytes Only**

Melvin accepts **any byte-addressable data**. No preprocessing required. The system learns patterns automatically.

---

## ✅ What Melvin Can Accept

### 1. **Text Data** (Any Format)
```c
// Plain text
melvin_feed_byte(g, 0, 'H', 0.5f);
melvin_feed_byte(g, 0, 'e', 0.5f);
// ...

// UTF-8 text (multi-byte characters)
// Feed each byte of UTF-8 encoding
uint8_t utf8_bytes[] = {0xE2, 0x82, 0xAC};  // € symbol
for (int i = 0; i < 3; i++) {
    melvin_feed_byte(g, 0, utf8_bytes[i], 0.5f);
}

// JSON, XML, CSV - all just bytes
// No parsing needed - system learns structure
```

### 2. **Binary Data**
```c
// Images (raw pixel bytes)
uint8_t *pixels = read_image("photo.jpg");
for (size_t i = 0; i < image_size; i++) {
    melvin_feed_byte(g, 10, pixels[i], 0.3f);  // Port 10 = vision
}

// Audio (raw samples)
int16_t *samples = read_audio("sound.wav");
for (size_t i = 0; i < audio_size; i++) {
    uint8_t byte = (uint8_t)(samples[i] & 0xFF);
    melvin_feed_byte(g, 20, byte, 0.3f);  // Port 20 = audio
}

// Machine code (compiled binaries)
uint8_t *code = read_binary("program.bin");
for (size_t i = 0; i < code_size; i++) {
    melvin_feed_byte(g, 30, code[i], 0.4f);  // Port 30 = code
}
```

### 3. **Structured Data** (As Bytes)
```c
// JSON - feed as raw bytes
const char *json = "{\"name\":\"Melvin\",\"age\":1}";
for (size_t i = 0; i < strlen(json); i++) {
    melvin_feed_byte(g, 0, json[i], 0.5f);
}

// CSV - feed as raw bytes
const char *csv = "name,age,city\nMelvin,1,Earth";
for (size_t i = 0; i < strlen(csv); i++) {
    melvin_feed_byte(g, 0, csv[i], 0.5f);
}

// Protocol buffers, MessagePack, etc.
// All just bytes - feed directly
```

### 4. **Mixed/Dirty Data**
```c
// Log files with mixed formats
FILE *log = fopen("app.log", "r");
int c;
while ((c = fgetc(log)) != EOF) {
    melvin_feed_byte(g, 0, (uint8_t)c, 0.3f);
}
// System learns patterns despite noise

// Corrupted data - system handles it
uint8_t corrupted[] = {0xFF, 0x00, 0xAA, 0x55, ...};
for (size_t i = 0; i < corrupted_size; i++) {
    melvin_feed_byte(g, 0, corrupted[i], 0.2f);
}
// Patterns still form around valid sequences
```

### 5. **Streaming Data** (Real-time)
```c
// Network packets
void on_packet_received(uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        melvin_feed_byte(g, 1, data[i], 0.4f);  // Port 1 = network
    }
}

// Sensor data
void on_sensor_update(float value) {
    uint8_t bytes[4];
    memcpy(bytes, &value, 4);
    for (int i = 0; i < 4; i++) {
        melvin_feed_byte(g, 2, bytes[i], 0.3f);  // Port 2 = sensor
    }
}
```

---

## 🎯 Data Format Requirements

### **NONE!** 

Melvin has **zero format requirements**:

- ❌ No JSON schema
- ❌ No XML validation
- ❌ No CSV headers required
- ❌ No preprocessing
- ❌ No cleaning
- ❌ No normalization

**Just feed raw bytes.**

---

## 📊 What to Expect

### 1. **Automatic Pattern Discovery**

The system automatically discovers patterns:

```c
// Feed: "hello world hello world"
// System discovers:
// - Pattern: "hello world" (repeated sequence)
// - Pattern: "hello" (subsequence)
// - Pattern: "world" (subsequence)
// - Pattern: "ll" (repeated in "hello")
```

### 2. **Structure Learning**

From unstructured data, structure emerges:

```c
// Feed messy log:
// "ERROR: file not found\nWARNING: retry\nERROR: timeout\n"
// System learns:
// - Pattern: "ERROR: " → error message
// - Pattern: "WARNING: " → warning message
// - Pattern: error types (file not found, timeout)
```

### 3. **Concept Formation**

Concepts form from co-occurrence:

```c
// Feed text corpus
// System learns:
// - "cat" and "dog" often appear together → related concepts
// - "hello" and "world" appear together → phrase pattern
// - Code patterns: "if (" → ")" → control flow concept
```

### 4. **Hierarchical Patterns**

Patterns build on patterns:

```c
// Feed code:
// "int add(int a, int b) { return a + b; }"
// System learns:
// - Byte-level: "int " pattern
// - Token-level: "int", "add", "(", ")" patterns
// - Statement-level: function definition pattern
// - Concept-level: "add" → addition concept
```

---

## 🔄 Data Flow Examples

### Example 1: Text Corpus
```
Input: "The quick brown fox jumps over the lazy dog."
       "The quick brown fox jumps over the lazy dog."

Process:
1. Feed each byte sequentially
2. System creates nodes for each byte value
3. System creates edges between sequential bytes
4. System discovers "The quick brown fox..." pattern (repeated)
5. System creates pattern node for this sequence
6. System learns word boundaries (spaces)
7. System learns word patterns ("the", "quick", "brown", etc.)

Output:
- Pattern nodes for repeated phrases
- Word pattern nodes
- Concept relationships (fox → animal, jumps → action)
```

### Example 2: Code Files
```
Input: C source code files

Process:
1. Feed each byte of source code
2. System learns syntax patterns:
   - "int " → type declaration
   - "()" → function call
   - "{}" → block structure
3. System learns semantic patterns:
   - Function definitions
   - Variable assignments
   - Control flow
4. System can compile and create EXEC nodes

Output:
- Syntax pattern nodes
- Semantic pattern nodes
- EXEC nodes (compiled functions)
- Code structure understanding
```

### Example 3: Mixed/Dirty Data
```
Input: Log file with errors, warnings, mixed formats

Process:
1. Feed all bytes (including noise)
2. System learns:
   - Valid patterns (repeated sequences)
   - Noise patterns (random bytes, less frequent)
3. Strong patterns emerge (valid data)
4. Weak patterns fade (noise)

Output:
- Strong pattern nodes (valid log entries)
- Weak/noise patterns (filtered out naturally)
- Structure emerges despite chaos
```

---

## 💡 Best Practices

### 1. **Use Port Nodes for Context**
```c
// Different ports = different contexts
melvin_feed_byte(g, 0, byte, energy);   // Port 0 = general text
melvin_feed_byte(g, 10, byte, energy);  // Port 10 = vision
melvin_feed_byte(g, 20, byte, energy);  // Port 20 = audio
melvin_feed_byte(g, 30, byte, energy);  // Port 30 = code
```

### 2. **Energy Levels Matter**
```c
// Higher energy = more important
melvin_feed_byte(g, 0, byte, 1.0f);  // Important data
melvin_feed_byte(g, 0, byte, 0.1f);  // Background noise
```

### 3. **Feed Sequences Together**
```c
// Feed related data in sequence
// System learns temporal relationships
for (size_t i = 0; i < data_len; i++) {
    melvin_feed_byte(g, port, data[i], energy);
    melvin_call_entry(g);  // Process after each byte (or batch)
}
```

### 4. **Let System Learn Structure**
```c
// Don't pre-parse - let system discover
// Feed raw JSON, XML, CSV
// System learns structure automatically
```

---

## ⚠️ What NOT to Do

### ❌ Don't Preprocess
```c
// BAD: Pre-parsing
JSON *parsed = parse_json(data);
for each field {
    melvin_feed_byte(...);  // Lost structure
}

// GOOD: Feed raw bytes
for (size_t i = 0; i < data_len; i++) {
    melvin_feed_byte(g, 0, data[i], 0.5f);
}
// System learns JSON structure automatically
```

### ❌ Don't Clean Data
```c
// BAD: Removing "noise"
clean_data = remove_noise(raw_data);
feed(clean_data);

// GOOD: Feed everything
feed(raw_data);
// System learns what's noise vs signal
```

### ❌ Don't Normalize
```c
// BAD: Normalizing formats
normalized = normalize(data);
feed(normalized);

// GOOD: Feed as-is
feed(data);
// System learns variations
```

---

## 🎯 Expected Outcomes

### Short-term (First 1000 bytes)
- Basic byte nodes created (0-255)
- Sequential edges form
- Simple patterns emerge

### Medium-term (10K-100K bytes)
- Word patterns discovered
- Common sequences identified
- Pattern nodes created

### Long-term (1M+ bytes)
- Hierarchical patterns
- Concept relationships
- EXEC nodes (from code)
- Semantic understanding

---

## 📝 Summary

**Input Format**: Any raw bytes
**Preprocessing**: None required
**Cleaning**: Not needed
**Structure**: Learned automatically
**Output**: Patterns, concepts, understanding

**The system is designed to handle dirty, unstructured, mixed data.**
**Just feed it bytes and let it learn.**

