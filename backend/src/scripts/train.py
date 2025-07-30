import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your model and dataset
from backend.src.model.handwriting_model import HandwritingRecognitionModel
from backend.src.preprocessing.dataset import HandwritingDataset

# Configuration
class Config:
    # Character set (example - adjust based on your dataset)
    CHAR_SET = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

def create_char_mapping(char_set):
    """Create character to index mapping"""
    return {char: i for i, char in enumerate(char_set)}

def main():
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char_mapping = create_char_mapping(Config.CHAR_SET)
    
    # Model
    model = HandwritingRecognitionModel(num_classes=len(Config.CHAR_SET)+1)  # +1 for CTC blank
    model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CTCLoss(blank=len(Config.CHAR_SET))
    
    # Dataset and DataLoader
    dataset = HandwritingDataset(split="train")
    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            
            # Convert text labels to numerical indices
            # Note: You'll need to implement this based on your dataset
            targets = torch.tensor([char_mapping[char] for char in labels], dtype=torch.long)
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
            
            # Forward pass
            outputs = model(images)
            input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long)
            
            # Compute loss
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} completed | Avg Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()