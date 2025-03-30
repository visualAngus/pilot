import os
import json
import shutil
from pathlib import Path

def prepare_dataset(source_folder, output_folder):
    """
    Prépare les données pour l'entraînement en créant une structure standard
    
    Args:
        source_folder: Dossier contenant les images sources
        output_folder: Dossier où créer la structure de dataset
    """
    # Créer les dossiers nécessaires
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'depths'), exist_ok=True)
    
    # Charger les annotations existantes si elles existent
    annotations_file = os.path.join(output_folder, 'annotations.json')
    data = {'images': {}}
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            data = json.load(f)
    
    # Trouver toutes les images dans le dossier source
    source_path = Path(source_folder)
    image_files = list(source_path.glob('*.png')) + list(source_path.glob('*.jpg')) + list(source_path.glob('*.jpeg'))
    
    print(f"Traitement de {len(image_files)} images depuis {source_folder}")
    
    # Traiter chaque image
    for img_path in image_files:
        img_name = img_path.name
        rel_path = f"images/{img_name}"
        dest_path = os.path.join(output_folder, rel_path)
        
        # Copier l'image vers le dossier de destination
        shutil.copy2(img_path, dest_path)
        print(f"Image copiée: {img_path} -> {dest_path}")
        
        # Ajouter l'entrée dans les annotations si elle n'existe pas déjà
        if img_name not in data['images']:
            data['images'][img_name] = {
                'path': rel_path,
                'vehicles': {},
                'depth_map_path': f"depths/depth_{img_name}"
            }
    
    # Sauvegarder les annotations
    with open(annotations_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Annotations sauvegardées dans {annotations_file}")

if __name__ == "__main__":
    # Exemple d'utilisation
    source_folder = "l:/projet/IA/pilot/img/Driving Around New York City - Manhattan 4k Drive Through"
    output_folder = "l:/projet/IA/pilot/data/train"
    prepare_dataset(source_folder, output_folder)