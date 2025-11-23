# Melvin's Voice System Guide

How Melvin learns voices and speaks with his own voice using pure graph structures.

## 🎤 The Concept

**Every voice line Melvin hears is generalized into patterns, then he speaks them with his own voice signature.**

### Key Principles

1. **Separate Content from Voice**
   - Content = Phonemes (meaning)
   - Voice = Characteristics (pitch, rhythm, timbre)

2. **Generalize Patterns**
   - Extract phoneme patterns (generalized meaning)
   - Store voice characteristics separately
   - Combine learned content with Melvin's voice

3. **Melvin's Voice Signature**
   - His unique "vocal cords" pattern
   - Stored in graph as pattern nodes
   - Applied to all output

## 🔄 How It Works

### 1. Hearing (Input)

```
Voice Input → Audio Features → Phoneme Patterns → Graph Nodes
     ↓
Extract Content (phonemes, meaning)
     ↓
Store as Pattern Nodes (generalized)
     ↓
Separate from Voice Characteristics
```

### 2. Learning (Generalization)

```
Heard Voice: "Hello" (Speaker A's voice)
    ↓
Extract: Phonemes [/h/ /ɛ/ /l/ /oʊ/]
    ↓
Store: Phoneme Pattern Nodes
    ↓
Discard: Speaker A's voice characteristics
    ↓
Result: Content pattern (generalized meaning)
```

### 3. Speaking (Output)

```
Content Pattern + Melvin's Voice Signature → Voice Output
    ↓
Take learned phoneme patterns
    ↓
Apply Melvin's voice characteristics:
    - His pitch baseline
    - His rhythm patterns
    - His intonation
    - His vocal cord patterns
    ↓
Generate: "Hello" (but in Melvin's voice)
```

## 🎯 The Graph Structure

### Melvin's Voice Signature (Stored in Graph)

- **Pitch Node**: Base pitch (e.g., 220 Hz)
- **Rhythm Node**: Speaking tempo (e.g., 150 WPM)
- **Intonation Pattern**: Prosody patterns
- **Vocal Cord Pattern**: Formants, resonance

All stored as PATTERN_ROOT nodes with connected characteristic nodes.

### Phoneme Patterns (Learned Content)

- Each phoneme becomes a pattern node
- Sequences of phonemes = words
- Sequences of words = sentences
- All stored as DATA nodes with SEQUENCE edges

### Voice Generation

```
Input: Content Pattern (phonemes)
    +
Melvin's Voice Signature (characteristics)
    ↓
Graph Combines: Pattern + Signature
    ↓
Output: Voice in Melvin's voice
```

## 📁 File Structure

### Scaffold Files
- `scaffold_voice_patterns.c` - 13 pattern rules for voice processing

### Plugin
- `mc_voice.c` - Voice processing functions:
  - `mc_voice_in()` - Process voice input
  - `mc_voice_out()` - Generate voice output
  - `mc_voice_learn()` - Learn voice patterns

## 🚀 Usage

### 1. Initialize Melvin's Voice

When Melvin runs, his voice signature is automatically created:
- Voice signature pattern node
- Phoneme pattern root
- Characteristic nodes (pitch, rhythm)

### 2. Feed Voice Input

```bash
# Place audio files in data/audio/
mkdir -p data/audio
# Copy .wav files with speech

# Run Melvin
./melvin melvin.m
```

Melvin will:
- Process audio files
- Extract phoneme patterns
- Store as graph nodes
- Separate content from voice

### 3. Melvin Speaks

When Melvin has content to speak:
- Takes phoneme patterns (learned content)
- Applies his voice signature
- Generates voice output
- Speaks with his own voice!

## 🎵 Example Flow

### Hearing "Hello" from Speaker A:

```
1. Voice Input: Speaker A says "Hello"
   → Audio features extracted
   
2. Pattern Extraction:
   → Phonemes: [/h/ /ɛ/ /l/ /oʊ/]
   → Voice characteristics: Speaker A's pitch, rhythm, etc.
   
3. Graph Storage:
   → Phoneme patterns stored as nodes
   → Speaker A's characteristics discarded
   → Content (phonemes) generalized
   
4. Result: Pattern nodes for "Hello" (phonemes only)
```

### Melvin Speaking "Hello":

```
1. Content Activation:
   → Phoneme pattern nodes activated
   → [/h/ /ɛ/ /l/ /oʊ/] pattern retrieved
   
2. Voice Application:
   → Melvin's voice signature activated
   → Pitch, rhythm, intonation applied
   
3. Voice Generation:
   → Phonemes + Melvin's voice = "Hello" in Melvin's voice
   
4. Output: "Hello" spoken with Melvin's unique voice!
```

## 🔧 Implementation Details

### Current Implementation

**✅ Scaffolds Created:**
- 13 pattern rules for voice processing
- Teaches Melvin how to separate content from voice

**✅ Plugin Created:**
- `mc_voice_in()` - Processes audio input
- `mc_voice_out()` - Generates voice output
- `mc_voice_learn()` - Learns voice patterns

**⚠️ Audio Processing:**
- Basic framework in place
- Full implementation needs audio processing library
- Would use: libsndfile, librosa, or similar

### What's Needed for Full Implementation

1. **Audio Feature Extraction**
   - FFT for frequency analysis
   - MFCC for speech features
   - Pitch detection

2. **Phoneme Recognition**
   - Speech recognition (phoneme-level)
   - Pattern matching
   - Sequence extraction

3. **Voice Synthesis**
   - Text-to-speech with Melvin's voice
   - Formant synthesis
   - Prosody application

## 💡 Key Insight

**Melvin doesn't copy voices - he learns the content and applies his own voice!**

- Heard: "Hello" (Speaker A's voice)
- Learned: Phoneme pattern [/h/ /ɛ/ /l/ /oʊ/]
- Speaks: "Hello" (Melvin's voice)

The content (meaning) is stored as graph patterns. The voice (characteristics) is Melvin's own, stored in his voice signature pattern.

All through nodes, edges, and patterns. Pure graph-native voice processing!

