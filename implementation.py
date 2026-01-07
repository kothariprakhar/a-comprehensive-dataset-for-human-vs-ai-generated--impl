import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import random

# ==========================================
# 1. Configuration & Constants
# ==========================================
CONFIG = {
    'batch_size': 8,
    'learning_rate': 0.001,
    'num_epochs': 2,
    'image_size': 224,
    'num_classes_multiclass': 6,  # Real + 5 Generators
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Mapping for Task 2 (Source Attribution)
LABEL_MAP = {
    0: 'Real',
    1: 'Stable Diffusion 3',
    2: 'Stable Diffusion 2.1',
    3: 'SDXL',
    4: 'DALL-E 3',
    5: 'MidJourney v6'
}

# ==========================================
# 2. Dummy Dataset Generation
# ==========================================
class MS_COCOAI_Dummy(Dataset):
    """
    Simulates the MS COCOAI dataset described in the paper.
    It generates random noise images to represent the 96,000 real 
    and synthetic datapoints.
    """
    def __init__(self, num_samples=100, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.data = []
        
        for _ in range(num_samples):
            # Randomly assign a source label (0 to 5)
            label_source = random.randint(0, 5)
            
            # Binary label: 0 if Real, 1 if AI-Generated
            label_binary = 0.0 if label_source == 0 else 1.0
            
            # Generate a random dummy image (3 channels, H, W)
            # In a real scenario, this would load an image from disk
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            self.data.append((dummy_image, label_binary, label_source))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_arr, binary_lbl, source_lbl = self.data[idx]
        
        # Convert to PIL-like structure for transforms (simulated via ToTensor)
        if self.transform:
            image = self.transform(img_arr)
        else:
            image = torch.tensor(img_arr).permute(2, 0, 1).float() / 255.0
            
        return image, torch.tensor(binary_lbl, dtype=torch.float32), torch.tensor(source_lbl, dtype=torch.long)

# ==========================================
# 3. Model Architecture
# ==========================================
class DeFactifyNet(nn.Module):
    """
    A Multi-Head Network based on ResNet50.
    - Shared Backbone: Extracts visual features.
    - Head 1: Binary Classification (Real vs. Fake).
    - Head 2: Multi-class Classification (Source Attribution).
    """
    def __init__(self, base_model='resnet50', frozen_backbone=False):
        super(DeFactifyNet, self).__init__()
        
        # Load Pre-trained ResNet
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        
        # Remove the original FC layer to use the features directly
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Task 1 Head: Binary Classification (1 output with Sigmoid logic implicit in BCEWithLogits)
        self.binary_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) 
        )

        # Task 2 Head: Multi-class Source Attribution (6 outputs)
        self.multiclass_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, CONFIG['num_classes_multiclass'])
        )

    def forward(self, x):
        features = self.backbone(x)
        
        # Branching logic
        binary_logits = self.binary_head(features)
        multiclass_logits = self.multiclass_head(features)
        
        return binary_logits, multiclass_logits

# ==========================================
# 4. Training Loop Implementation
# ==========================================
def train_model():
    print("Initializing MS COCOAI Detection System...")
    
    # Transforms matching standard ImageNet statistics
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize Dataset and DataLoader
    dataset = MS_COCOAI_Dummy(num_samples=64, transform=transform)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Initialize Model
    model = DeFactifyNet().to(CONFIG['device'])
    
    # Loss Functions
    criterion_binary = nn.BCEWithLogitsLoss()   # For Real vs Fake
    criterion_multi = nn.CrossEntropyLoss()     # For Source Attribution
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    print(f"Starting training on {CONFIG['device']} for {CONFIG['num_epochs']} epochs...")
    
    model.train()
    for epoch in range(CONFIG['num_epochs']):
        running_loss = 0.0
        
        for inputs, binary_labels, source_labels in dataloader:
            inputs = inputs.to(CONFIG['device'])
            binary_labels = binary_labels.to(CONFIG['device']).unsqueeze(1) # Shape [B, 1]
            source_labels = source_labels.to(CONFIG['device'])              # Shape [B]

            optimizer.zero_grad()

            # Forward Pass
            pred_binary, pred_multi = model(inputs)

            # Calculate Losses for both tasks
            loss_b = criterion_binary(pred_binary, binary_labels)
            loss_m = criterion_multi(pred_multi, source_labels)
            
            # Combined loss (weighted equally here, but can be tuned)
            total_loss = loss_b + loss_m
            
            # Backward Pass
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] Loss: {epoch_loss:.4f}")

    print("Training Complete.")
    return model

# ==========================================
# 5. Inference / Demo
# ==========================================
if __name__ == "__main__":
    trained_model = train_model()
    
    # Simulate inference on a single image
    trained_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(CONFIG['device'])
        b_logit, m_logit = trained_model(dummy_input)
        
        prob_fake = torch.sigmoid(b_logit).item()
        predicted_source_idx = torch.argmax(m_logit, dim=1).item()
        
        print("\n--- Inference Results ---")
        print(f"Probability of being AI Generated: {prob_fake:.4f}")
        print(f"Predicted Source Model: {LABEL_MAP[predicted_source_idx]}")