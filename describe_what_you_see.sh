#!/bin/bash
# DESCRIBE WHAT YOU SEE - Full system test
# Shows nodes, edges, patterns that form from camera input

cd ~/melvin

echo "=========================================="
echo "MELVIN: DESCRIBE WHAT YOU SEE"
echo "=========================================="
echo ""

# Create fresh brain for clean test
echo "1. Creating fresh brain..."
cat > /tmp/fresh_brain.c << 'CCODE'
#include "src/melvin.h"
#include <stdio.h>
int main() {
    Graph *g = melvin_open("brain_test.m", 1000, 50000, 256*1024);
    if (!g) return 1;
    printf("Created: %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, (unsigned long long)g->edge_count);
    melvin_close(g);
    return 0;
}
CCODE
gcc -std=c11 -O2 -I. -o /tmp/fresh_brain /tmp/fresh_brain.c src/melvin.c -lm -pthread 2>/dev/null
/tmp/fresh_brain

echo ""
echo "2. Capturing camera image..."
ffmpeg -f v4l2 -video_size 640x480 -i /dev/video0 -frames:v 1 /tmp/camera_frame.jpg -y 2>/dev/null
if [ -f /tmp/camera_frame.jpg ]; then
    SIZE=$(stat -c%s /tmp/camera_frame.jpg)
    echo "   ✓ Captured: $(( SIZE / 1024 )) KB"
else
    echo "   ⚠ Failed"
    exit 1
fi

echo ""
echo "3. Running Vision AI..."
python3 << 'PYCODE'
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import urllib.request
import subprocess

# Load image
img = Image.open('/tmp/camera_frame.jpg')
print(f"   Image: {img.size}")

# Load model
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.eval()

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)

# Get labels
labels = urllib.request.urlopen(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).read().decode().split('\n')

probs = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probs, 5)

print("\n   Vision identified:")
vision_labels = []
for i in range(5):
    label = labels[top5_catid[i]].replace(' ', '_').upper()
    prob = 100 * top5_prob[i].item()
    print(f"     {i+1}. {label}: {prob:.1f}%")
    vision_labels.append(label)

# Save labels for Melvin
with open('/tmp/vision_labels.txt', 'w') as f:
    f.write('\n'.join(vision_labels))

print(f"\n   Top label: {vision_labels[0]}")
PYCODE

echo ""
echo "4. Feeding to Melvin (vision + pixels)..."
cat > /tmp/feed_and_show.c << 'CCODE'
#include "src/melvin.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    Graph *g = melvin_open("brain_test.m", 0, 0, 0);
    if (!g) return 1;
    
    uint64_t start_nodes = g->node_count;
    uint64_t start_edges = g->edge_count;
    
    printf("   Start: %llu nodes, %llu edges\n\n", 
           (unsigned long long)start_nodes, (unsigned long long)start_edges);
    
    // Read vision labels
    FILE *f = fopen("/tmp/vision_labels.txt", "r");
    char labels[5][64];
    int label_count = 0;
    if (f) {
        while (label_count < 5 && fgets(labels[label_count], 64, f)) {
            // Remove newline
            labels[label_count][strcspn(labels[label_count], "\n")] = 0;
            label_count++;
        }
        fclose(f);
    }
    
    // Feed each label multiple times (learning)
    printf("   Feeding vision labels:\n");
    for (int rep = 0; rep < 10; rep++) {
        for (int i = 0; i < label_count; i++) {
            // Feed "I_SEE_" prefix
            const char *prefix = "I_SEE_";
            for (size_t j = 0; j < strlen(prefix); j++) {
                melvin_feed_byte(g, 100, (uint8_t)prefix[j], 0.4f);
            }
            // Feed label
            for (size_t j = 0; j < strlen(labels[i]); j++) {
                melvin_feed_byte(g, 100, (uint8_t)labels[i][j], 0.4f);
            }
            melvin_feed_byte(g, 100, ' ', 0.2f);
        }
        melvin_call_entry(g);
    }
    
    for (int i = 0; i < label_count; i++) {
        printf("     - %s\n", labels[i]);
    }
    
    // Feed some pixels
    printf("\n   Feeding image pixels...\n");
    FILE *img = fopen("/tmp/camera_frame.jpg", "rb");
    if (img) {
        unsigned char buf[2000];
        size_t n = fread(buf, 1, 2000, img);
        for (size_t i = 0; i < n; i++) {
            melvin_feed_byte(g, 1, buf[i], 0.3f);  // Port 1 = camera
        }
        fclose(img);
        melvin_call_entry(g);
        printf("     - Fed %zu bytes\n", n);
    }
    
    printf("\n   ==============================\n");
    printf("   RESULTS:\n");
    printf("   ==============================\n");
    printf("   Nodes: %llu → %llu (+%llu NEW)\n",
           (unsigned long long)start_nodes,
           (unsigned long long)g->node_count,
           (unsigned long long)(g->node_count - start_nodes));
    printf("   Edges: %llu → %llu (+%llu NEW)\n",
           (unsigned long long)start_edges,
           (unsigned long long)g->edge_count,
           (unsigned long long)(g->edge_count - start_edges));
    
    // Count patterns
    uint32_t pattern_count = 0;
    for (uint32_t i = 0; i < g->node_count && i < 10000; i++) {
        if (g->nodes[i].pattern_data_offset > 0) pattern_count++;
    }
    printf("   Patterns: %u\n", pattern_count);
    
    // Show sample edges
    printf("\n   Sample connections:\n");
    int shown = 0;
    for (uint64_t i = 0; i < g->edge_count && shown < 10; i++) {
        uint32_t src = g->edges[i].src;
        uint32_t dst = g->edges[i].dst;
        if (src < 256 && dst < 256 && g->edges[i].w > 0.05f) {
            char s = (src >= 32 && src < 127) ? (char)src : '?';
            char d = (dst >= 32 && dst < 127) ? (char)dst : '?';
            printf("     '%c' → '%c' (w=%.3f)\n", s, d, g->edges[i].w);
            shown++;
        }
    }
    
    melvin_close(g);
    return 0;
}
CCODE

gcc -std=c11 -O2 -I. -o /tmp/feed_and_show /tmp/feed_and_show.c src/melvin.c -lm -pthread 2>&1 | grep error || echo ""
/tmp/feed_and_show

echo ""
echo "5. Speaking description..."
# Get top label
TOP_LABEL=$(head -1 /tmp/vision_labels.txt | tr '_' ' ' | tr '[:upper:]' '[:lower:]')
DESCRIPTION="I see a $TOP_LABEL"
echo "   Generating: $DESCRIPTION"

espeak "$DESCRIPTION" -w /tmp/speech.wav 2>/dev/null
if [ -f /tmp/speech.wav ]; then
    echo "   Playing audio..."
    aplay -D default /tmp/speech.wav 2>/dev/null
    echo "   ✓ Spoke: $DESCRIPTION"
fi

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="

