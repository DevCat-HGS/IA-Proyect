import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        # Reshape output para que coincida con las dimensiones esperadas
        out = out.contiguous().view(batch_size * out.size(1), -1)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
                weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_())

# Preparación de datos
text = "hola como estas hola bien y tu hola genial"  
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

seq_length = 10
dataX = []
dataY = []
for i in range(0, len(text) - seq_length):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_idx[char] for char in seq_in])
    dataY.append(char_to_idx[seq_out])

X = torch.tensor(dataX, dtype=torch.long)
y = torch.tensor(dataY, dtype=torch.long)

embedding_dim = 64  # Reducido de 128
hidden_dim = 128   # Reducido de 256
num_layers = 2
dropout = 0.2
model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
batch_size = X.size(0)
for epoch in range(num_epochs):
    hidden = model.init_hidden(batch_size)
    model.train()
    optimizer.zero_grad()
    
    output, hidden = model(X, hidden)
    # Asegurar que las dimensiones coincidan
    y_reshaped = y.repeat(X.size(1))
    loss = criterion(output, y_reshaped)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "text_generation_model.pth")

def generate_text(model, start_string, length=100, temperature=1.0):
    model.eval()
    chars = [char_to_idx.get(ch, char_to_idx[' ']) for ch in start_string]
    while len(chars) < seq_length:
        chars = [char_to_idx[' ']] + chars
    
    hidden = model.init_hidden(1)
    generated_chars = list(chars)
    
    for _ in range(length):
        x = torch.tensor([chars[-seq_length:]], dtype=torch.long)
        output, hidden = model(x, hidden)
        
        # Tomar solo la última predicción
        output = output[-1, :] / temperature
        
        # Calcular probabilidades
        probs = torch.softmax(output, dim=-1)
        
        # Muestrear usando torch.multinomial
        next_idx = torch.multinomial(probs.view(-1), 1).item()
        
        generated_chars.append(next_idx)
        chars = chars[1:] + [next_idx]
    
    return ''.join([idx_to_char[i] for i in generated_chars])

start_string = "hola"
generated_text = generate_text(model, start_string, length=50)
print("Texto generado:", generated_text)