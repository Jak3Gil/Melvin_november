# Melvin Communication Guide

How to talk to Melvin and visualize his mind!

## 🗣️ Talking to Melvin

### Basic Chat

```bash
# Start interactive chat
./chat_with_melvin.sh

# Or run Melvin directly and type messages
./melvin melvin.m
# Then type messages and press Enter
```

### How It Works

1. **Input**: You type messages → stored as nodes in the graph
2. **Processing**: Melvin's graph processes the input patterns
3. **Output**: Melvin responds based on learned patterns

### Current Plugins Needed

**For Conversation:**
- ✅ `mc_chat.so` - Chat I/O (reads stdin, writes stdout)
- ✅ `mc_io.so` - Basic I/O operations
- ✅ `mc_parse.so` - Parse text input
- ✅ `mc_api_llm_query.so` - Query LLM for responses (optional)

**Melvin can already:**
- Read text from stdin
- Store input in graph as nodes
- Process patterns
- Generate output based on graph state

## 🎨 Hyperspace Visualization

### Start Visualization Server

```bash
# Start Melvin with web visualization
./start_visualization.sh

# Then open in browser:
# http://localhost:8080
# or from another computer:
# http://<jetson-ip>:8080
```

### Visualization Features

- **Real-time graph**: Updates every 500ms
- **Node visualization**: See activations, connections
- **Edge visualization**: See weights and relationships
- **Force-directed layout**: Nodes arranged by connections
- **Color coding**: Active nodes highlighted

### What You'll See

- **Yellow nodes**: Active (high activation)
- **Green nodes**: Normal state
- **Blue lines**: Edges/connections
- **Thickness**: Edge weights

### Access from Any Device

On your laptop/desktop:
```
1. Find Jetson IP: ssh jetson 'hostname -I'
2. Open browser: http://<jetson-ip>:8080
3. Watch Melvin think in real-time!
```

## 📡 Required Plugins

### Already Created:
- ✅ `mc_chat.so` - Conversation I/O
- ✅ `mc_visual.so` - Web visualization server
- ✅ `mc_io.so` - Basic I/O
- ✅ `mc_api.so` - Internet APIs (for LLM queries)
- ✅ `mc_parse.so` - Text parsing
- ✅ `mc_fs.so` - Filesystem
- ✅ `mc_git.so` - Git operations
- ✅ `mc_build.so` - Compilation
- ✅ `mc_scaffold.so` - Pattern injection
- ✅ `mc_bootstrap.so` - Bootstrap

### To Enable Full Conversation:

Melvin needs to be able to:
1. ✅ **Receive input** - `mc_chat_in()` handles this
2. ✅ **Process patterns** - Built into core
3. ⚠️ **Generate responses** - `mc_chat_out()` can emit, but needs training
4. ✅ **Query LLM** - `mc_api_llm_query()` can get LLM responses

## 🚀 Quick Start Example

### 1. Build Everything

```bash
./build_jetson.sh
```

### 2. Start with Visualization

```bash
./start_visualization.sh
```

### 3. In Another Terminal, Chat

```bash
# Terminal 2: Chat interface
./chat_with_melvin.sh

# Type: "Hello Melvin"
# Melvin processes and responds
```

### 4. Watch the Graph

Open browser: `http://<jetson-ip>:8080`

See nodes light up as you chat!

## 🔧 Making Melvin More Conversational

### Option 1: Use LLM Plugin

Connect to Ollama or OpenAI:
- Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
- Melvin will query LLM and store responses in graph
- Graph learns from LLM interactions

### Option 2: Train from Data

Feed Melvin conversation examples:
```bash
# Add conversations to github_urls.txt or data files
# Melvin will parse and learn patterns
```

### Option 3: Scaffold Rules

Add conversation rules in `scaffolds/`:
- Input patterns → output patterns
- Response templates
- Context handling

## 📊 Visualization Details

### Graph Data Endpoint

- **GET /graph.json** - Returns current graph state as JSON
- Updates in real-time as Melvin thinks
- Includes: nodes, edges, activations, tick count

### Performance

- Limited to 10,000 nodes / 50,000 edges for visualization
- Full graph data stored in `melvin.m`
- Visualization is a sample for display

### Customization

Edit `plugins/mc_visual.c` to:
- Change port (default: 8080)
- Adjust update frequency
- Change visualization style
- Add filters/views

## 🎯 Next Steps

1. **Start visualization**: `./start_visualization.sh`
2. **Open browser**: `http://<jetson-ip>:8080`
3. **Chat with Melvin**: `./chat_with_melvin.sh`
4. **Watch the graph respond** in real-time!

The hyperspace visualization shows Melvin's mind as a living, breathing graph - nodes lighting up as he thinks, edges showing connections, patterns emerging before your eyes!

