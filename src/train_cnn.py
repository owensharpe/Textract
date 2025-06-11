import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import create_dataloaders
from tqdm import tqdm

class CNNEncoder(nn.Module):
    def __init__(self, encoder_dim=64):
        super(CNNEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, encoder_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        features = self.features(x)
        features = features.permute(0, 2, 3, 1)
        return features

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, encoder_out, captions):
        encoder_out = encoder_out.mean(dim=[1, 2])
        embeddings = self.embedding(captions)
        encoder_out = encoder_out.unsqueeze(1).expand(-1, captions.size(1), -1)
        decoder_input = embeddings + encoder_out
        hidden, _ = self.gru(decoder_input)
        outputs = self.fc(hidden)
        return outputs

class LatexCNN(nn.Module):
    def __init__(self, vocab_size):
        super(LatexCNN, self).__init__()
        self.encoder = CNNEncoder()
        self.decoder = Decoder(vocab_size)
        
    def forward(self, images, captions):
        encoder_out = self.encoder(images)
        outputs = self.decoder(encoder_out, captions)
        return outputs

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = batch['images'].to(device)
            captions = batch['latex'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, captions[:, :-1])
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = captions[:, 1:].reshape(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                captions = batch['latex'].to(device)
                outputs = model(images, captions[:, :-1])
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = captions[:, 1:].reshape(-1)
                val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}, Loss: {val_loss:.4f}')
        torch.save(model.state_dict(), 'model.pth')

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, _, vocab = create_dataloaders(
        'data/crohme_images/TC11_CROHME23/INKML',
        batch_size=32,
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.9], std=[0.1])
        ])
    )
    
    print(f"Vocabulary size: {len(vocab.token_to_idx)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Initialize model and training components
    model = LatexCNN(len(vocab.token_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device)

if __name__ == '__main__':
    main() 