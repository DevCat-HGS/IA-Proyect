import torch
import torch.nn as nn
import torch.optim as optim
from model_example import TextGenerationModel
from utils import save_model

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

# Ajustar hiperparámetros
embedding_dim = 64  # Reducido
hidden_dim = 128   # Reducido
num_layers = 2
dropout = 0.3      # Aumentado para mejor regularización
learning_rate = 0.005  # Ajustado

model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Agregar early stopping
best_loss = float('inf')
patience = 5
patience_counter = 0

num_epochs = 100
batch_size = X.size(0)
for epoch in range(num_epochs):
    hidden = model.init_hidden(batch_size)
    model.train()
    optimizer.zero_grad()
    
    output, hidden = model(X, hidden)
    loss = criterion(output, y)
    
    loss.backward()
    optimizer.step()
    
    # Early stopping
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
        save_model(model, "best_model.pth")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

save_model(model, "text_generation_model.pth")
