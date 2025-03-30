import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import os
from PIL import Image

# ----- 1. Modèle -----
class TrajectoryPredictor(nn.Module):
    def __init__(self, num_points=20, vocab_size=1000, text_embedding_dim=128, state_dim=10):
        super(TrajectoryPredictor, self).__init__()

        # Branche image (CNN ResNet18)
        self.cnn = models.resnet18(pretrained=True)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # On enlève la dernière FC pour garder les features

        # Branche texte (Embedding + LSTM)
        self.text_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=text_embedding_dim)
        self.text_rnn = nn.LSTM(input_size=text_embedding_dim, hidden_size=128, batch_first=True)
        self.text_fc = nn.Linear(128, 128)

        # Branche état du véhicule
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Fusion et prédiction de trajectoire
        fusion_dim = num_ftrs + 128 + 64
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_points * 2)  # Chaque point = (x, y)
        )

    def forward(self, image, text, state):
        image_features = self.cnn(image)
        embedded_text = self.text_embedding(text)
        _, (hn, _) = self.text_rnn(embedded_text)
        text_features = self.text_fc(hn[-1])
        state_features = self.state_fc(state)
        fusion_features = torch.cat((image_features, text_features, state_features), dim=1)
        out = self.fusion_fc(fusion_features)
        return out.view(out.size(0), -1, 2)  # [batch, num_points, 2]


class TrajectoryDataset(Dataset):
    def __init__(self, data_path="dataset.json", tags_path="tags.json", image_folder=None, max_samples=None, num_points=20, state_dim=5,num_samples=1000, vocab_size=1000):
        """
        Dataset pour charger et traiter les données d'annotation de trajectoire
        
        Args:
            data_path: Chemin vers le fichier JSON contenant les annotations
            tags_path: Chemin vers le fichier JSON contenant les tags disponibles
            image_folder: Dossier contenant les images (si None, utilisera le chemin relatif)
            max_samples: Nombre maximum d'échantillons à charger (None = tous)
            num_points: Nombre de points à générer pour chaque trajectoire
            state_dim: Dimension du vecteur d'état (vitesse, angle, accélération)
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.vocab_size = vocab_size
        self.state_dim = state_dim
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.num_points = num_points
        self.state_dim = state_dim
        
        # Chargement des données
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Chargement des tags
        with open(tags_path, 'r') as f:
            self.tags = json.load(f)
        
        self.vocab_size = len(self.tags)
        
        # Extraction des noms de fichiers d'images annotées
        self.image_files = list(self.data.keys())
        
        # Limitation du nombre d'échantillons si demandé
        if max_samples and max_samples < len(self.image_files):
            self.image_files = self.image_files[:max_samples]
        
        # Chemin vers les images
        self.image_folder = image_folder
        
        # Transformation pour les images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"Dataset chargé avec {len(self.image_files)} échantillons et {self.vocab_size} tags uniques")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        annotation = self.data[image_name]
        
        # Chargement de l'image
        if self.image_folder:
            image_path = os.path.join(self.image_folder, image_name)
        else:
            # Si aucun dossier n'est spécifié, on suppose que les images sont dans le même dossier
            image_path = image_name
            
        try:
            # Chargement et transformation de l'image
            img = Image.open(image_path).convert('RGB')
            image = self.transform(img)
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {image_path}: {str(e)}")
            # En cas d'erreur, retourner une image de bruit
            image = torch.randn(3, 224, 224)
        
        # Traitement des tags
        # Convertir les IDs des tags en un tenseur one-hot
        tag_ids = annotation.get("tag_ids", [])
        text = torch.zeros(self.vocab_size, dtype=torch.long)
        for tag_id in tag_ids:
            if 0 <= tag_id < self.vocab_size:
                text[tag_id] = 1
        
        # Traitement de l'état du véhicule (vitesse, angle, accélération)
        state_data = annotation.get("data", {})
        state = torch.zeros(self.state_dim)
        
        # Récupération des valeurs de vitesse, angle et accélération
        if "Vitesse (km/h)" in state_data:
            state[0] = float(state_data["Vitesse (km/h)"])
        if "Angle (°)" in state_data:
            state[1] = float(state_data["Angle (°)"])
        if "Accélération (m/s²)" in state_data:
            state[2] = float(state_data["Accélération (m/s²)"])
        
        # Traitement de la trajectoire
        raw_trajectory = annotation.get("trajectory", [])
        
        if len(raw_trajectory) > 0:
            # Conversion en tenseur
            traj_tensor = torch.tensor(raw_trajectory, dtype=torch.float)
            
            # Normalisation des coordonnées entre 0 et 1
            if traj_tensor.shape[0] > 0:
                # Trouver min et max sur chaque dimension
                min_vals, _ = torch.min(traj_tensor, dim=0)
                max_vals, _ = torch.max(traj_tensor, dim=0)
                # Éviter division par zéro
                range_vals = max_vals - min_vals
                range_vals[range_vals == 0] = 1.0
                # Normaliser
                traj_tensor = (traj_tensor - min_vals) / range_vals
            
            # Rééchantillonnage pour avoir exactement num_points
            if len(raw_trajectory) != self.num_points:
                # Si trop peu de points, interpoler
                if len(raw_trajectory) < self.num_points:
                    # Créer une grille linéaire pour l'interpolation
                    indices = torch.linspace(0, len(raw_trajectory)-1, self.num_points)
                    resampled_trajectory = []
                    
                    for i in range(self.num_points):
                        idx = indices[i].item()
                        if idx.is_integer():
                            resampled_trajectory.append(raw_trajectory[int(idx)])
                        else:
                            # Interpolation linéaire
                            idx_floor = int(idx)
                            idx_ceil = min(idx_floor + 1, len(raw_trajectory) - 1)
                            weight_ceil = idx - idx_floor
                            weight_floor = 1 - weight_ceil
                            
                            x = weight_floor * raw_trajectory[idx_floor][0] + weight_ceil * raw_trajectory[idx_ceil][0]
                            y = weight_floor * raw_trajectory[idx_floor][1] + weight_ceil * raw_trajectory[idx_ceil][1]
                            
                            resampled_trajectory.append([x, y])
                    
                    trajectory = torch.tensor(resampled_trajectory, dtype=torch.float)
                
                # Si trop de points, sous-échantillonner
                else:
                    indices = torch.linspace(0, len(raw_trajectory)-1, self.num_points).long()
                    trajectory = traj_tensor[indices]
            else:
                trajectory = traj_tensor
        else:
            # Si pas de trajectoire, générer une ligne droite aléatoire
            x_traj = torch.linspace(0, 1, self.num_points)
            y_traj = 0.5 * torch.ones(self.num_points)
            trajectory = torch.stack([x_traj, y_traj], dim=1)

        return image, text, state, trajectory

# ----- 2. Dataset Synthétique -----
class TrajectoryDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=20, vocab_size=1000, state_dim=10):
        self.num_samples = num_samples
        self.num_points = num_points
        self.vocab_size = vocab_size
        self.state_dim = state_dim
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Image aléatoire (bruit)
        image = torch.randn(3, 224, 224)

        # Texte aléatoire (séquence de tokens)
        text = torch.randint(0, self.vocab_size, (10,))

        # État du véhicule (vitesse, accélération...)
        state = torch.randn(self.state_dim)

        # Trajectoire simulée (exemple : ligne droite légèrement bruitée)
        trajectory = torch.linspace(0, 1, self.num_points)  # Génère 20 points entre 0 et 1
        x_traj = trajectory + 0.05 * torch.randn(self.num_points)
        y_traj = 0.5 * trajectory + 0.05 * torch.randn(self.num_points)
        trajectory = torch.stack([x_traj, y_traj], dim=1)

        return image, text, state, trajectory

# ----- 3. Entraînement -----
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device="cuda"):
    criterion = nn.MSELoss()  # Perte basée sur la distance des points de trajectoire
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, texts, states, trajectories in train_loader:
            images, texts, states, trajectories = images.to(device), texts.to(device), states.to(device), trajectories.to(device)

            optimizer.zero_grad()
            outputs = model(images, texts, states)
            loss = criterion(outputs, trajectories)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, texts, states, trajectories in val_loader:
                images, texts, states, trajectories = images.to(device), texts.to(device), states.to(device), trajectories.to(device)
                outputs = model(images, texts, states)
                loss = criterion(outputs, trajectories)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ----- 4. Exécution -----
if __name__ == '__main__':
    # Hyperparamètres
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.0005
    num_points = 20
    vocab_size = 1000d
    state_dim = 10

    # Chargement des données
    dataset = TrajectoryDataset(num_samples=5000, num_points=num_points, vocab_size=vocab_size, state_dim=state_dim)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instanciation du modèle
    model = TrajectoryPredictor(num_points=num_points, vocab_size=vocab_size, state_dim=state_dim)

    # Entraînement
    train_model(model, train_loader, val_loader, epochs=num_epochs, lr=learning_rate, device="cuda" if torch.cuda.is_available() else "cpu")

    # ----- 5. Sauvegarde du modèle -----
    torch.save(model.state_dict(), "model.pth")