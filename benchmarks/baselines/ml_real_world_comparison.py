"""
Real-World ML Comparison

Train LSTM and small Transformer on same text as Melvin
Compare: speed, memory, patterns learned (perplexity as proxy)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from pathlib import Path

# Default text (Shakespeare)
DEFAULT_TEXT = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life."""

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

def train_lstm(text, epochs=10):
    print("\n" + "="*60)
    print("LSTM BASELINE")
    print("="*60)
    
    # Prepare data
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Text length: {len(text)} characters")
    
    # Create sequences
    seq_length = 50
    sequences = []
    targets = []
    
    for i in range(0, len(text) - seq_length):
        seq = text[i:i+seq_length]
        target = text[i+1:i+seq_length+1]
        sequences.append([char_to_idx[c] for c in seq])
        targets.append([char_to_idx[c] for c in target])
    
    print(f"Training sequences: {len(sequences)}")
    
    # Model
    model = SimpleLSTM(vocab_size, hidden_size=128)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    print(f"Parameters: {param_count:,}")
    print(f"Memory: {param_bytes:,} bytes ({param_bytes/1024/1024:.2f} MB)")
    print(f"\nTraining for {epochs} epochs...")
    
    # Training
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Process in batches
        batch_size = 32
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            # Convert to tensors
            seq_tensor = torch.LongTensor(batch_seqs)
            target_tensor = torch.LongTensor(batch_targets)
            
            optimizer.zero_grad()
            output = model(seq_tensor)
            
            # Reshape for loss
            output = output.reshape(-1, vocab_size)
            target_tensor = target_tensor.reshape(-1)
            
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(sequences):.4f}")
    
    train_time = time.time() - start_time
    
    # Calculate perplexity (proxy for patterns learned)
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i in range(min(100, len(sequences))):
            seq_tensor = torch.LongTensor([sequences[i]])
            target_tensor = torch.LongTensor([targets[i]])
            output = model(seq_tensor)
            output = output.reshape(-1, vocab_size)
            target_tensor = target_tensor.reshape(-1)
            total_loss += criterion(output, target_tensor).item()
        
        perplexity = torch.exp(torch.tensor(total_loss / min(100, len(sequences)))).item()
    
    chars_processed = len(text) * epochs
    
    print(f"\nResults:")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Chars/sec: {chars_processed / train_time:.1f}")
    print(f"  Perplexity: {perplexity:.2f} (lower = better)")
    print(f"  Memory: {param_bytes/1024/1024:.2f} MB")
    
    return {
        'model': 'LSTM',
        'time': train_time,
        'chars_per_sec': chars_processed / train_time,
        'parameters': param_count,
        'memory_bytes': param_bytes,
        'perplexity': perplexity,
        'epochs': epochs
    }

def main():
    print("="*60)
    print("REAL-WORLD ML COMPARISON")
    print("="*60)
    
    # Load text
    if len(sys.argv) > 1:
        text_file = sys.argv[1]
        print(f"\nLoading: {text_file}")
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    else:
        print("\nUsing default text (Shakespeare)")
        text = DEFAULT_TEXT
    
    # Limit to reasonable size for quick test
    max_chars = 10000
    if len(text) > max_chars:
        print(f"(Limiting to first {max_chars} characters for speed)")
        text = text[:max_chars]
    
    print(f"Text: {len(text)} characters")
    print(f"Preview: \"{text[:100]}...\"")
    
    # Train LSTM
    lstm_results = train_lstm(text, epochs=10)
    
    # Save results
    Path("benchmarks/data").mkdir(parents=True, exist_ok=True)
    with open("benchmarks/data/ml_comparison_results.csv", "w") as f:
        f.write("model,time_sec,chars_per_sec,parameters,memory_bytes,perplexity\n")
        f.write(f"{lstm_results['model']},{lstm_results['time']:.3f},"
                f"{lstm_results['chars_per_sec']:.1f},{lstm_results['parameters']},"
                f"{lstm_results['memory_bytes']},{lstm_results['perplexity']:.2f}\n")
    
    print("\n" + "="*60)
    print("Comparison saved to: benchmarks/data/ml_comparison_results.csv")
    print("="*60)
    print("\nNow run: ./benchmarks/experiment5_real_world [textfile]")
    print("Compare Melvin's speed and pattern count!")

if __name__ == "__main__":
    main()

