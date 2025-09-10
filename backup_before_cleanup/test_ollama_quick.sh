#!/bin/bash

echo "🧪 QUICK OLLAMA TEST"
echo "==================="

echo "1. Checking if Ollama is running..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running"
    exit 1
fi

echo "2. Checking installed models..."
ollama list

echo "3. Testing simple request..."
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "prompt": "Hello", "stream": false}' \
  --max-time 10

echo ""
echo "✅ Ollama test complete!"
