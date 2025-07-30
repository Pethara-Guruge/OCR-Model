from pathlib import Path
import pandas as pd
import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset

class HandwritingDataset(Dataset):
    def __init__(self, split="train"):
        # Path to the root of your project
        project_root = Path(__file__).parent.parent.parent.parent  # Goes up to "Project" folder
        
        # Path to the labels file (using backend/data structure)
        labels_path = project_root / "backend" / "data" / "splits" / split / "labels.csv"
        
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found at: {labels_path}\n"
                f"Please run preprocessing first!"
            )
        
        self.df = pd.read_csv(labels_path)
        
        # Verify text column exists
        if 'text' not in self.df.columns:
            raise KeyError("CSV file must contain a 'text' column")
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def __getitem__(self, idx):
        img_path = Path(self.df.iloc[idx]["image_path"])
        
        # Handle both relative and absolute paths
        if not img_path.exists():
            # Try prepending project root
            img_path = Path(__file__).parent.parent.parent.parent / img_path
        
        # Load image with OpenCV
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
        
        # Convert to PyTorch tensor
        img = (img / 255.0 * 2) - 1  # Convert to [-1,1] range
        text = self.df.iloc[idx]["text"] or ""  # Handle potential None values
        return torch.FloatTensor(img).unsqueeze(0), text

    def __len__(self):
        return len(self.df)
    
    def get_char_set(self):
        """Get all unique characters in the dataset"""
        all_text = ''.join(self.df['text'].astype(str).tolist())  # Ensure all items are strings
        return sorted(set(all_text))