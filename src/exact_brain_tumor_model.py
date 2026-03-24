import torch
import torch.nn as nn
import os

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_brain_tumor_model(model_path, load_mode='complete'):
    """
    Loads the Brain Tumor PyTorch model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load the dictionary or state dict
    data = torch.load(model_path, map_location=device)
    
    # Initialize the model structure
    metadata = {}
    if isinstance(data, dict):
        num_classes = data.get('num_classes', 4)
        model = BrainTumorCNN(num_classes=num_classes)
        
        # Load weights
        if 'model_state_dict' in data:
            model.load_state_dict(data['model_state_dict'])
        elif 'state_dict' in data:
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data) # Just in case it's pure state dict
            
        # Extract metadata
        metadata = {
            'class_names': data.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary']),
            'mean': data.get('mean', [0.485, 0.456, 0.406]),
            'std': data.get('std', [0.229, 0.224, 0.225]),
            'best_val_accuracy': data.get('best_val_accuracy', None)
        }
    else:
        # Fallback if it's somehow not a dictionary (unlikely)
        model = BrainTumorCNN(num_classes=4)
        model.load_state_dict(data)
        metadata = {
            'class_names': ['glioma', 'meningioma', 'notumor', 'pituitary'],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'best_val_accuracy': None
        }
        
    model = model.to(device)
    model.eval()
    
    return model, device, metadata
