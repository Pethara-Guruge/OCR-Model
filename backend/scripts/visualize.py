import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.src.preprocessing.dataset import HandwritingDataset
import matplotlib.pyplot as plt

def show_samples(n=3):
    dataset = HandwritingDataset()
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    
    for i in range(n):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_samples()