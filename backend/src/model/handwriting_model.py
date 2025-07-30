import torch
import torch.nn as nn
import torch.nn.functional as F

class HandwritingRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(HandwritingRecognitionModel, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        
        # BiLSTM layers
        self.lstm = nn.LSTM(input_size=128, 
                            hidden_size=256, 
                            num_layers=2, 
                            bidirectional=True,
                            batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1))
        
        # Output layer
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # CNN features
        features = self.cnn(x)
        
        # Reshape for LSTM (batch, timesteps, channels)
        b, c, h, w = features.size()
        features = features.view(b, c * h, w)  # (batch, 128*4, 8)
        features = features.permute(0, 2, 1)   # (batch, 8, 512)
        
        # BiLSTM
        lstm_out, _ = self.lstm(features)  # (batch, seq_len, 2*hidden_size)
        
        # Attention
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final output
        output = self.fc(context_vector)
        return output