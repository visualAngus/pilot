import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class VehicleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Répertoire contenant les images et annotations
            transform (callable, optional): Transformations optionnelles
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Charger les annotations
        annotations_file = os.path.join(root_dir, 'annotations.json')
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                data = json.load(f)
                # Format les annotations comme une liste pour l'indexation
                self.annotations = []
                for image_path, image_data in data.get('images', {}).items():
                    annotation = {
                        'image_path': image_data.get('path', ''),
                        'depth_path': image_data.get('depth_map_path', ''),
                        'corners': [],
                        'orientation': 'avant'  # Valeur par défaut
                    }
                    
                    # Extraire les informations du premier véhicule (s'il existe)
                    vehicles = image_data.get('vehicles', {})
                    if vehicles and '1' in vehicles:
                        vehicle = vehicles['1']
                        annotation['corners'] = vehicle.get('corners', [])
                        annotation['orientation'] = vehicle.get('orientation', 'avant')
                    
                    self.annotations.append(annotation)
        else:
            print(f"Attention: fichier d'annotations {annotations_file} non trouvé")
            self.annotations = []
        
        print(f"Dataset chargé avec {len(self.annotations)} images")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if idx >= len(self.annotations):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.annotations)} items")
        
        annotation = self.annotations[idx]
        
        # Handling of image path to avoid duplication
        img_path = annotation['image_path']
        if self.root_dir in img_path:
            img_path = img_path
        else:
            img_path = os.path.join(self.root_dir, img_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
            img_width, img_height = image.size
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {img_path}: {str(e)}")
            img_width, img_height = 320, 180
            image = Image.new('RGB', (img_width, img_height), color='black')
        
        # Handling of depth path to avoid duplication
        depth_path = annotation['depth_path']
        if self.root_dir in depth_path:
            depth_path = depth_path
        else:
            depth_path = os.path.join(self.root_dir, depth_path)
        
        try:
            depth = Image.open(depth_path).convert('L')
            if depth.size != (img_width, img_height):
                depth = depth.resize((img_width, img_height))
        except Exception as e:
            print(f"Erreur lors du chargement de la profondeur {depth_path}: {str(e)}")
            depth = Image.new('L', (img_width, img_height), color='black')
        
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)
        else:
            image = torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32) / 255.0
            depth = torch.tensor(np.array(depth)[None, :, :], dtype=torch.float32) / 255.0
        
        corners = torch.tensor(
            [coord for corner in annotation['corners'] for coord in corner],
            dtype=torch.float32
        ) if annotation['corners'] else torch.zeros(8, dtype=torch.float32)
        
        orientation = annotation['orientation']
        orientations = ["avant", "arrière", "gauche", "droite"]
        orientation_idx = orientations.index(orientation) if orientation in orientations else 0
        orientation_onehot = torch.zeros(len(orientations), dtype=torch.float32)
        orientation_onehot[orientation_idx] = 1.0
        
        return image, depth, corners, orientation_onehot

def train_model(model, data_loader, val_loader=None, num_epochs=10, learning_rate=0.001, device='cuda'):
    """
    Fonction pour entraîner le modèle VehicleDepthNN
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    corners_criterion = nn.MSELoss()
    orientation_criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    print(f"Entraînement sur {device}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, depths, corners_targets, orientation_targets) in enumerate(data_loader):
            images = images.to(device)
            depths = depths.to(device)
            corners_targets = corners_targets.to(device)
            orientation_targets = orientation_targets.to(device)
            
            optimizer.zero_grad()
            corners_pred, orientation_pred = model(images, depths)
            
            corners_loss = corners_criterion(corners_pred, corners_targets)
            class_orientations = torch.argmax(orientation_targets, dim=1)
            orientation_loss = orientation_criterion(orientation_pred, class_orientations)
            loss = corners_loss + orientation_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:
                print(f'[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/10:.3f}')
                running_loss = 0.0
        
        epoch_loss = running_loss / len(data_loader)
        train_losses.append(epoch_loss)
        
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            
            with torch.no_grad():
                for images, depths, corners_targets, orientation_targets in val_loader:
                    images = images.to(device)
                    depths = depths.to(device)
                    corners_targets = corners_targets.to(device)
                    orientation_targets = orientation_targets.to(device)
                    
                    corners_pred, orientation_pred = model(images, depths)
                    
                    corners_loss = corners_criterion(corners_pred, corners_targets)
                    class_orientations = torch.argmax(orientation_targets, dim=1)
                    orientation_loss = orientation_criterion(orientation_pred, class_orientations)
                    val_running_loss += (corners_loss + orientation_loss).item()
            
            val_epoch_loss = val_running_loss / len(val_loader)
            val_losses.append(val_epoch_loss)
            print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}')
    
    torch.save(model.state_dict(), 'vehicle_depth_model.pth')
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    if val_loader:
        plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()
    
    return model, train_losses, val_losses

class VehicleDepthNN(nn.Module):
    def __init__(self, num_vehicles=2, input_height=180, input_width=320):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, 8 + num_vehicles)

    def forward(self, image, depth):
        x = torch.cat([image, depth], dim=1)
        feats = self.conv(x).view(x.size(0), -1)
        out = self.fc(feats)
        
        corners_pred = out[:, :8]
        orientation_logits = out[:, 8:]
        return corners_pred, orientation_logits

if __name__ == "__main__":
    model = VehicleDepthNN(num_vehicles=4)
    train_dataset = VehicleDataset(root_dir='./data/train')
    val_dataset = VehicleDataset(root_dir='./data/train')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    trained_model, train_losses, val_losses = train_model(
        model=model,
        data_loader=train_loader,
        val_loader=val_loader,
        num_epochs=800,
        learning_rate=0.001
    )