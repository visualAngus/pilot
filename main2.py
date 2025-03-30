import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import time
import tkinter as tk
from tkinter import filedialog, Scale, Button, Label, Frame, StringVar, OptionMenu, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os

class DepthMapGenerator:
    def __init__(self, model_type="DPT_Hybrid"):
        # Charger le modèle MiDaS
        self.model_type = model_type
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        
        # Définir le device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        
        # Charger les transformations du modèle
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform
        
        # Stocker la profondeur originale
        self.raw_depth = None
    
    def compute_depth_map(self, image_path, output_path="./data/train/depths/", min_depth=None, max_depth=None, apply_colormap=False,save=False):
        # Charger l'image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Appliquer les transformations
        input_batch = self.transform(img).to(self.device)

        output_path = output_path + "_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
        
        # Faire la prédiction
        with torch.no_grad():
            prediction = self.model(input_batch)
        
        # Stocker la profondeur brute
        self.raw_depth = prediction.squeeze().cpu().numpy()
        
        # Appliquer la plage de profondeur si spécifiée
        if min_depth is not None and max_depth is not None:
            # Normaliser entre 0 et 1
            normalized_depth = (self.raw_depth - self.raw_depth.min()) / (self.raw_depth.max() - self.raw_depth.min())
            
            # Masquer les valeurs hors de la plage sélectionnée
            mask = (normalized_depth >= min_depth) & (normalized_depth <= max_depth)
            
            # Créer une image contrastée
            depth_map = np.zeros_like(normalized_depth)
            depth_map[mask] = normalized_depth[mask]
            
            # Re-normaliser pour utiliser toute la plage dynamique
            if depth_map.max() > depth_map.min():
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        else:
            # Normalisation standard
            depth_map = (self.raw_depth - self.raw_depth.min()) / (self.raw_depth.max() - self.raw_depth.min()) * 255
        
        depth_map = depth_map.astype(np.uint8)
        
        # Augmenter le contraste - faire ceci AVANT d'appliquer la colormap
        depth_map = cv2.equalizeHist(depth_map)
        
        # Appliquer éventuellement une colormap pour améliorer la visualisation
        if apply_colormap:
            depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        
        # Sauvegarder et afficher la carte de profondeur
        if save:
            cv2.imwrite(output_path, depth_map)
        return depth_map,output_path

class DepthMapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Générateur de carte de profondeur")
        self.root.geometry("1000x800")
        
        # Attributs principaux
        self.depth_generator = DepthMapGenerator()
        self.image_path = None
        self.current_depth_map = None
        self.original_image = None
        self.json_image_info = {}
        
        # Variables pour le traitement par lot
        self.image_folder = None
        self.image_files = []
        self.current_image_index = -1
        self.all_json_data = {}  # Pour stocker les données de toutes les images traitées
        
        # Variables pour le dessin du rectangle
        self.rect_start_x = None
        self.rect_start_y = None
        self.rectangle = None
        self.dragging = False
        self.drag_corner = None
        self.rectangles = []  # Liste des rectangles (pour les multiples véhicules)
        self.current_rect_id = None  # ID du rectangle en cours de manipulation
        self.vehicle_count = 0  # Compteur de véhicules
        self.draw_mode = False  # Mode de dessin actif
        self.current_vehicle_id = None  # ID du véhicule en cours d'édition
        self.ctrl_pressed = False  # Indique si la touche Ctrl est enfoncée
        
        # Frame principale
        main_frame = Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame pour les contrôles
        controls_frame = Frame(main_frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        # Bouton pour sélectionner un dossier d'images (au lieu d'une image unique)
        self.select_btn = Button(controls_frame, text="Sélectionner un dossier d'images", command=self.select_folder)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        # Bouton pour charger un fichier JSON existant
        self.load_json_btn = Button(controls_frame, text="Charger JSON", command=self.load_json)
        self.load_json_btn.pack(side=tk.LEFT, padx=5)
        
        # Slider pour min_depth
        self.min_depth_label = Label(controls_frame, text="Min:")
        self.min_depth_label.pack(side=tk.LEFT, padx=5)
        self.min_depth_slider = Scale(controls_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                       resolution=0.01, length=200, command=self.update_depth_range)
        self.min_depth_slider.pack(side=tk.LEFT, padx=5)
        self.min_depth_slider.set(0.0)
        
        # Slider pour max_depth
        self.max_depth_label = Label(controls_frame, text="Max:")
        self.max_depth_label.pack(side=tk.LEFT, padx=5)
        self.max_depth_slider = Scale(controls_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                       resolution=0.01, length=200, command=self.update_depth_range)
        self.max_depth_slider.pack(side=tk.LEFT, padx=5)
        self.max_depth_slider.set(1.0)
        
        # Checkbox pour la colormap
        self.use_colormap = tk.BooleanVar()
        self.colormap_check = tk.Checkbutton(controls_frame, text="Appliquer colormap", 
                                             variable=self.use_colormap, command=self.update_depth_range)
        self.colormap_check.pack(side=tk.LEFT, padx=5)
        
        # Frame pour les contrôles de véhicule
        vehicle_frame = Frame(main_frame)
        vehicle_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Bouton pour ajouter une voiture
        self.add_vehicle_btn = Button(vehicle_frame, text="Ajouter une voiture", command=self.add_vehicle)
        self.add_vehicle_btn.pack(side=tk.LEFT, padx=5)
        
        # Menu pour sélectionner l'orientation
        self.orientation_var = StringVar(root)
        self.orientation_var.set("avant")  # Valeur par défaut
        self.orientation_label = Label(vehicle_frame, text="Orientation:")
        self.orientation_label.pack(side=tk.LEFT, padx=5)
        self.orientation_menu = OptionMenu(vehicle_frame, self.orientation_var, "avant", "arrière", "gauche", "droite")
        self.orientation_menu.pack(side=tk.LEFT, padx=5)
        
        # Bouton pour appliquer l'orientation
        self.apply_orientation_btn = Button(vehicle_frame, text="Appliquer orientation", command=self.apply_orientation)
        self.apply_orientation_btn.pack(side=tk.LEFT, padx=5)
        
        # Bouton pour terminer l'édition du véhicule
        self.finish_vehicle_btn = Button(vehicle_frame, text="Terminer véhicule", command=self.finish_vehicle)
        self.finish_vehicle_btn.pack(side=tk.LEFT, padx=5)
        
        # Navigation buttons (previous and next)
        self.prev_image_btn = Button(vehicle_frame, text="Image précédente", command=self.previous_image)
        self.prev_image_btn.pack(side=tk.RIGHT, padx=5)
        
        # Bouton pour passer à l'image suivante
        self.next_image_btn = Button(vehicle_frame, text="Image suivante", command=self.next_image)
        self.next_image_btn.pack(side=tk.RIGHT, padx=5)
        
        # Étiquette d'information sur l'image actuelle
        self.image_info_var = StringVar(root)
        self.image_info_var.set("Aucune image chargée")
        self.image_info_label = Label(vehicle_frame, textvariable=self.image_info_var)
        self.image_info_label.pack(side=tk.RIGHT, padx=10)
        
        # Frame pour les images
        self.images_frame = Frame(main_frame)
        self.images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas pour l'image originale (remplace le Label pour permettre de dessiner)
        self.original_label = Label(self.images_frame, text="Image originale")
        self.original_label.grid(row=0, column=0, pady=5)
        self.original_canvas = tk.Canvas(self.images_frame, width=800, height=800, bg="black")
        self.original_canvas.grid(row=1, column=0, padx=10, pady=10)
        
        # Événements de souris pour le canvas
        self.original_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.original_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.original_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        # Ajouter l'événement pour le double-clic
        self.original_canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        
        # Événements clavier pour détecter Ctrl
        self.root.bind("<KeyPress-Control_L>", self.on_ctrl_press)
        self.root.bind("<KeyRelease-Control_L>", self.on_ctrl_release)
        self.root.bind("<KeyPress-Control_R>", self.on_ctrl_press)
        self.root.bind("<KeyRelease-Control_R>", self.on_ctrl_release)
        
        # Label pour la carte de profondeur
        self.depth_label = Label(self.images_frame, text="Carte de profondeur")
        self.depth_label.grid(row=0, column=1, pady=5)
        self.depth_display = Label(self.images_frame)
        self.depth_display.grid(row=1, column=1, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Sélectionnez un dossier d'images pour commencer")
        self.status_bar = Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Export button
        self.export_btn = Button(main_frame, text="Exporter JSON", command=self.export_json)
        self.export_btn.pack(side=tk.BOTTOM, pady=10)
        
        # Désactiver les sliders et boutons au début
        self.min_depth_slider["state"] = "disabled"
        self.max_depth_slider["state"] = "disabled"
        self.colormap_check["state"] = "disabled"
        self.add_vehicle_btn["state"] = "disabled"
        self.orientation_menu["state"] = "disabled"
        self.apply_orientation_btn["state"] = "disabled"
        self.finish_vehicle_btn["state"] = "disabled"
        self.prev_image_btn["state"] = "disabled"
        self.next_image_btn["state"] = "disabled"
    
    def select_folder(self):
        """Sélectionne un dossier d'images à traiter"""
        folder_path = filedialog.askdirectory(title="Sélectionner un dossier d'images")
        if not folder_path:
            return
            
        self.image_folder = folder_path
        
        # Chercher toutes les images dans le dossier
        self.image_files = []
        valid_extensions = ('.jpg', '.jpeg', '.png')
        for file in os.listdir(folder_path):
            if file.lower().endswith(valid_extensions):
                self.image_files.append(os.path.join(folder_path, file))
        
        # Vérifier s'il y a des images dans le dossier
        if not self.image_files:
            self.status_var.set(f"Aucune image trouvée dans le dossier {folder_path}")
            return
            
        # Initialiser l'index et charger la première image
        self.current_image_index = -1
        self.all_json_data = {}  # Réinitialiser les données JSON
        
        # Activer les boutons de navigation
        self.next_image_btn["state"] = "normal"
        self.prev_image_btn["state"] = "disabled"  # Pas d'image précédente au début
        
        # Charger la première image
        self.next_image()
        
        self.status_var.set(f"{len(self.image_files)} images trouvées dans le dossier {folder_path}")
    
    def previous_image(self):
        """Passe à l'image précédente dans le dossier"""
        # Sauvegarder les données de l'image actuelle
        if self.image_path and self.json_image_info:
            self.all_json_data[self.image_path] = self.json_image_info.copy()
        
        # Vérifier si on peut reculer
        if self.current_image_index <= 0:
            self.status_var.set("Vous êtes déjà à la première image.")
            self.prev_image_btn["state"] = "disabled"
            return
        
        # Passer à l'image précédente
        self.current_image_index -= 1
        self.image_path = self.image_files[self.current_image_index]
        
        # Mettre à jour l'information sur l'image
        self.image_info_var.set(f"Image {self.current_image_index + 1}/{len(self.image_files)}: {os.path.basename(self.image_path)}")
        
        # Réinitialiser les variables
        self.rectangles = []
        self.vehicle_count = 0
        self.current_vehicle_id = None
        
        # Charger l'image et configurer le JSON
        path_tmp = "./data/train/images/" + os.path.basename(self.image_path)
        
        # Vérifier si cette image a déjà des données JSON
        if self.image_path in self.all_json_data:
            self.json_image_info = self.all_json_data[self.image_path].copy()
            self.load_saved_vehicles()
        else:
            self.json_image_info = {"path": path_tmp, "vehicles": {}}
        
        self.original_image = cv2.imread(self.image_path)
        cv2.imwrite(path_tmp, self.original_image)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Afficher sur le canvas
        self.display_image_on_canvas(self.original_image)
        
        # Calculer la carte de profondeur
        self.compute_depth()
        
        # Activer/désactiver les boutons de navigation
        self.next_image_btn["state"] = "normal"
        if self.current_image_index <= 0:
            self.prev_image_btn["state"] = "disabled"
        
        # Activer les boutons
        self.add_vehicle_btn["state"] = "normal"
        
        self.status_var.set(f"Image chargée: {self.image_path}")
    
    def next_image(self):
        """Passe à l'image suivante dans le dossier"""
        # Sauvegarder les données de l'image actuelle dans all_json_data si une image est chargée
        if self.image_path and self.json_image_info:
            self.all_json_data[self.image_path] = self.json_image_info.copy()
        
        # Vérifier si on a atteint la fin de la liste
        if self.current_image_index >= len(self.image_files) - 1:
            self.status_var.set("Toutes les images ont été traitées.")
            return
        
        # Passer à l'image suivante
        self.current_image_index += 1
        self.image_path = self.image_files[self.current_image_index]
        
        # Mettre à jour l'information sur l'image
        self.image_info_var.set(f"Image {self.current_image_index + 1}/{len(self.image_files)}: {os.path.basename(self.image_path)}")
        
        # Réinitialiser les variables
        self.rectangles = []
        self.vehicle_count = 0
        self.current_vehicle_id = None
        
        # Charger l'image et configurer le JSON
        path_tmp = "./data/train/images/" + os.path.basename(self.image_path)
        
        # Vérifier si cette image a déjà des données JSON
        if self.image_path in self.all_json_data:
            self.json_image_info = self.all_json_data[self.image_path].copy()
            self.load_saved_vehicles()
        else:
            self.json_image_info = {"path": path_tmp, "vehicles": {}}
        
        self.original_image = cv2.imread(self.image_path)
        cv2.imwrite(path_tmp, self.original_image)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Afficher sur le canvas
        self.display_image_on_canvas(self.original_image)
        
        # Calculer la carte de profondeur
        self.compute_depth()
        
        # Activer/désactiver les boutons de navigation
        self.prev_image_btn["state"] = "normal"
        if self.current_image_index >= len(self.image_files) - 1:
            self.next_image_btn["state"] = "disabled"
        else:
            self.next_image_btn["state"] = "normal"
        
        # Activer les boutons
        self.add_vehicle_btn["state"] = "normal"
        
        self.status_var.set(f"Image chargée: {self.image_path}")
    
    def load_json(self):
        """Charge les données à partir d'un fichier JSON existant"""
        json_path = filedialog.askopenfilename(
            title="Sélectionner un fichier JSON",
            filetypes=[("Fichiers JSON", "*.json")]
        )
        
        if not json_path:
            return
        
        try:
            with open(json_path, 'r') as f:
                loaded_data = json.load(f)
            
            # Vérifier si le JSON a le bon format
            if "images" not in loaded_data:
                messagebox.showerror("Erreur", "Format de fichier JSON invalide: clé 'images' manquante")
                return
            
            # Charger les données
            self.all_json_data = loaded_data["images"]
            
            # Extraire le dossier d'images si disponible
            if "image_folder" in loaded_data:
                self.image_folder = loaded_data["image_folder"]
            
            # Construire la liste des images
            self.image_files = list(self.all_json_data.keys())
            
            # Afficher un résumé
            self.show_json_summary(loaded_data)
            
            # Initialiser l'index et charger la première image si des images sont disponibles
            if self.image_files:
                self.current_image_index = -1
                
                # Activer le bouton d'image suivante
                self.next_image_btn["state"] = "normal"
                self.prev_image_btn["state"] = "disabled"
                
                # Charger la première image
                self.next_image()
                
                self.status_var.set(f"{len(self.image_files)} images chargées depuis {json_path}")
            else:
                self.status_var.set(f"Aucune image trouvée dans le fichier JSON {json_path}")
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement du fichier JSON: {str(e)}")
            self.status_var.set(f"Erreur lors du chargement du fichier JSON")
    
    def show_json_summary(self, json_data):
        """Affiche un résumé des données JSON chargées"""
        # Créer une fenêtre pop-up pour le résumé
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Résumé des données JSON")
        summary_window.geometry("500x400")
        
        # Créer un widget Text pour afficher le résumé
        summary_text = tk.Text(summary_window, wrap=tk.WORD, padx=10, pady=10)
        summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Calculer et afficher les statistiques
        total_images = json_data.get("total_images", len(json_data.get("images", {})))
        total_vehicles = json_data.get("total_vehicles", 0)
        
        summary = f"Résumé des données chargées:\n\n"
        summary += f"- Nombre total d'images: {total_images}\n"
        summary += f"- Nombre total de véhicules annotés: {total_vehicles}\n"
        
        # Dossier d'images
        if "image_folder" in json_data:
            summary += f"- Dossier d'images: {json_data['image_folder']}\n"
        
        # Répartition des véhicules par image
        summary += "\nRépartition des véhicules par image:\n"
        
        images_data = json_data.get("images", {})
        for idx, (image_path, image_info) in enumerate(images_data.items(), 1):
            vehicle_count = len(image_info.get("vehicles", {}))
            image_name = os.path.basename(image_path)
            summary += f"{idx}. {image_name}: {vehicle_count} véhicule(s)\n"
            
            # Limiter l'affichage si trop d'images
            if idx >= 20 and len(images_data) > 20:
                summary += f"\n... et {len(images_data) - 20} autres images ...\n"
                break
        
        # Insérer le résumé dans le widget Text
        summary_text.insert(tk.END, summary)
        summary_text.config(state=tk.DISABLED)  # Rendre en lecture seule
        
        # Bouton pour fermer la fenêtre
        close_btn = tk.Button(summary_window, text="Fermer", command=summary_window.destroy)
        close_btn.pack(pady=10)
    
    def load_saved_vehicles(self):
        """Charge les véhicules sauvegardés à partir des données JSON"""
        # Récupérer les véhicules depuis le JSON
        vehicles_data = self.json_image_info.get("vehicles", {})
        
        # Réinitialiser la liste des rectangles
        self.rectangles = []
        self.vehicle_count = 0
        
        # Recréer les rectangles pour chaque véhicule
        for vehicle_id, vehicle_info in vehicles_data.items():
            vehicle_id = int(vehicle_id)
            if vehicle_id > self.vehicle_count:
                self.vehicle_count = vehicle_id
            
            corners = vehicle_info.get("corners", [])
            orientation = vehicle_info.get("orientation", "avant")
            
            # Créer le rectangle sur le canvas
            rect_id = self.original_canvas.create_polygon(
                corners[0][0], corners[0][1],
                corners[1][0], corners[1][1],
                corners[2][0], corners[2][1],
                corners[3][0], corners[3][1],
                outline="red", width=2, fill="", tags=f"vehicle_{vehicle_id}"
            )
            
            # Créer les poignées aux coins
            handles = []
            for i, (x, y) in enumerate(corners):
                handle_id = self.original_canvas.create_oval(
                    x-5, y-5, x+5, y+5, 
                    fill="red", tags=f"handle_{vehicle_id}_{i}"
                )
                handles.append(handle_id)
            
            # Ajouter le texte d'orientation
            text_x = sum(corner[0] for corner in corners) / 4
            text_y = sum(corner[1] for corner in corners) / 4
            text_id = self.original_canvas.create_text(
                text_x, text_y, text=orientation,
                fill="red", font=("Arial", 12, "bold")
            )
            
            # Ajouter le rectangle à la liste
            self.rectangles.append({
                "id": vehicle_id,
                "canvas_id": rect_id,
                "corners": corners,
                "handles": handles,
                "text_id": text_id,
                "orientation": orientation
            })
        
        # Mettre à jour les sliders si des valeurs sont disponibles
        if "min_depth" in self.json_image_info:
            self.min_depth_slider.set(self.json_image_info["min_depth"])
        if "max_depth" in self.json_image_info:
            self.max_depth_slider.set(self.json_image_info["max_depth"])

    def display_image_on_canvas(self, img, target_size=(800, 500)):
        # Redimensionner l'image pour qu'elle s'insère dans la taille cible tout en gardant le ratio
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        target_w, target_h = target_size
        if aspect_ratio > 1:  # Image plus large que haute
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:  # Image plus haute que large
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Créer une image de fond noire de la taille cible
        canvas_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Centrer l'image redimensionnée sur le fond noir
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        # Convertir en PIL pour tkinter
        img_pil = Image.fromarray(canvas_img)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        
        # Configurer le canvas pour la taille cible
        self.original_canvas.config(width=target_w, height=target_h)
        
        # Afficher l'image sur le canvas
        self.original_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        
        # Redessiner les rectangles existants
        self.redraw_all_rectangles()
    
    def redraw_all_rectangles(self):
        # Supprimer tous les rectangles du canvas
        for rect_id in self.rectangles:
            self.original_canvas.delete(rect_id["canvas_id"])
        
        # Redessiner chaque rectangle
        for rect in self.rectangles:
            color = "green" if rect["id"] == self.current_vehicle_id else "red"
            rect["canvas_id"] = self.original_canvas.create_polygon(
                rect["corners"][0][0], rect["corners"][0][1],
                rect["corners"][1][0], rect["corners"][1][1],
                rect["corners"][2][0], rect["corners"][2][1],
                rect["corners"][3][0], rect["corners"][3][1],
                outline=color, width=2, fill="", tags=f"vehicle_{rect['id']}"
            )
            
            # Ajouter des poignées aux coins
            for i, (x, y) in enumerate(rect["corners"]):
                handle_id = self.original_canvas.create_oval(
                    x-5, y-5, x+5, y+5, 
                    fill=color, tags=f"handle_{rect['id']}_{i}"
                )
                rect["handles"].append(handle_id)
            
            # Afficher l'orientation
            text_x = sum(corner[0] for corner in rect["corners"]) / 4
            text_y = sum(corner[1] for corner in rect["corners"]) / 4
            text_id = self.original_canvas.create_text(
                text_x, text_y, text=rect["orientation"],
                fill=color, font=("Arial", 12, "bold")
            )
            rect["text_id"] = text_id
    
    def compute_depth(self):
        if self.image_path:
            self.status_var.set("Calcul de la carte de profondeur en cours...")
            self.root.update()
            
            t = time.time()
            self.current_depth_map, self.output_path = self.depth_generator.compute_depth_map(
                self.image_path, save=True, apply_colormap=self.use_colormap.get()
            )

            # Enregistrer le chemin de la carte de profondeur dans le JSON
            self.json_image_info["depth_map_path"] = self.output_path
            
            # Afficher la carte de profondeur
            self.display_image(self.current_depth_map, self.depth_display)
            
            execution_time = time.time() - t
            self.status_var.set(f"Carte de profondeur calculée en {execution_time:.2f} secondes")
            
            # Activer les sliders
            self.min_depth_slider["state"] = "normal"
            self.max_depth_slider["state"] = "normal"
            self.colormap_check["state"] = "normal"
    
    def display_image(self, img, label, max_size=(800, 400)):
        # Redimensionner l'image pour l'affichage
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        if h > max_size[1]:
            h = max_size[1]
            w = int(h * aspect_ratio)
        
        if w > max_size[0]:
            w = max_size[0]
            h = int(w / aspect_ratio)
        
        img_resized = cv2.resize(img, (w, h))
        
        # Convertir en PIL pour tkinter
        if len(img_resized.shape) == 2:  # Image en niveaux de gris
            img_pil = Image.fromarray(img_resized)
        else:  # Image couleur
            img_pil = Image.fromarray(img_resized)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        label.configure(image=img_tk)
        label.image = img_tk  # Garder une référence
    
    def update_depth_range(self, event=None):
        if self.depth_generator.raw_depth is not None:
            min_depth = self.min_depth_slider.get()
            max_depth = self.max_depth_slider.get()
            
            # Assurer que min_depth est inférieur à max_depth
            if (min_depth >= max_depth):
                if event and hasattr(event, "widget"):
                    if event.widget == self.min_depth_slider:
                        min_depth = max_depth - 0.01
                        self.min_depth_slider.set(min_depth)
                    else:
                        max_depth = min_depth + 0.01
                        self.max_depth_slider.set(max_depth)
            
            self.status_var.set(f"Mise à jour de la tranche de profondeur: {min_depth:.2f} - {max_depth:.2f}")
            
            # Recalculer la carte de profondeur avec la nouvelle plage
            depth_map, _ = self.depth_generator.compute_depth_map(
                self.image_path, 
                min_depth=min_depth, 
                max_depth=max_depth,
                apply_colormap=self.use_colormap.get(),
                save=False
            )
            
            # Enregistrer les valeurs dans json_image_info
            self.json_image_info["min_depth"] = min_depth
            self.json_image_info["max_depth"] = max_depth
            
            # Si un véhicule est en cours d'édition, mettre à jour sa plage de profondeur
            if self.current_vehicle_id is not None:
                self.json_image_info["vehicles"][str(self.current_vehicle_id)]["min_depth"] = min_depth
                self.json_image_info["vehicles"][str(self.current_vehicle_id)]["max_depth"] = max_depth
            
            # Afficher la nouvelle carte de profondeur
            self.display_image(depth_map, self.depth_display)
    
    def add_vehicle(self):
        # Vérifier qu'une image est chargée
        if self.original_image is None:
            self.status_var.set("Chargez d'abord une image")
            return
        
        # Si un véhicule est déjà en cours d'édition, le terminer d'abord
        if self.current_vehicle_id is not None:
            self.finish_vehicle()
        
        # Incrémenter le compteur de véhicules
        self.vehicle_count += 1
        self.current_vehicle_id = self.vehicle_count
        
        # Activer le mode de dessin
        self.draw_mode = True
        
        # Initialiser le rectangle pour le nouveau véhicule
        default_size = min(self.original_canvas.winfo_width(), self.original_canvas.winfo_height()) // 4
        center_x = self.original_canvas.winfo_width() // 2
        center_y = self.original_canvas.winfo_height() // 2
        
        # Créer un rectangle par défaut au centre
        corners = [
            [center_x - default_size//2, center_y - default_size//2],  # Haut gauche
            [center_x + default_size//2, center_y - default_size//2],  # Haut droite
            [center_x + default_size//2, center_y + default_size//2],  # Bas droite
            [center_x - default_size//2, center_y + default_size//2]   # Bas gauche
        ]
        
        # Créer le rectangle sur le canvas
        rect_id = self.original_canvas.create_polygon(
            corners[0][0], corners[0][1],
            corners[1][0], corners[1][1],
            corners[2][0], corners[2][1],
            corners[3][0], corners[3][1],
            outline="green", width=2, fill="", tags=f"vehicle_{self.current_vehicle_id}"
        )
        
        # Créer les poignées aux coins
        handles = []
        for i, (x, y) in enumerate(corners):
            handle_id = self.original_canvas.create_oval(
                x-5, y-5, x+5, y+5, 
                fill="green", tags=f"handle_{self.current_vehicle_id}_{i}"
            )
            handles.append(handle_id)
        
        # Ajouter le texte d'orientation
        text_x = sum(corner[0] for corner in corners) / 4
        text_y = sum(corner[1] for corner in corners) / 4
        text_id = self.original_canvas.create_text(
            text_x, text_y, text="avant",
            fill="green", font=("Arial", 12, "bold")
        )
        
        # Ajouter le rectangle à la liste
        self.rectangles.append({
            "id": self.current_vehicle_id,
            "canvas_id": rect_id,
            "corners": corners,
            "handles": handles,
            "text_id": text_id,
            "orientation": "avant"
        })
        
        # Ajouter au JSON
        self.json_image_info["vehicles"][str(self.current_vehicle_id)] = {
            "min_depth": self.min_depth_slider.get(),
            "max_depth": self.max_depth_slider.get(),
            "corners": corners,
            "orientation": "avant"
        }
        
        # Activer les contrôles d'orientation
        self.orientation_menu["state"] = "normal"
        self.apply_orientation_btn["state"] = "normal"
        self.finish_vehicle_btn["state"] = "normal"
        
        self.status_var.set(f"Véhicule {self.current_vehicle_id} ajouté. Ajustez le rectangle et définissez l'orientation.")
    
    def on_canvas_press(self, event):
        if not self.draw_mode or self.current_vehicle_id is None:
            return
        
        x, y = event.x, event.y
        
        # Parcourir tous les rectangles pour trouver celui en cours d'édition
        for rect in self.rectangles:
            if rect["id"] == self.current_vehicle_id:
                # Vérifier si un coin est sélectionné
                for i, (corner_x, corner_y) in enumerate(rect["corners"]):
                    if abs(x - corner_x) <= 5 and abs(y - corner_y) <= 5:
                        self.dragging = True
                        self.drag_corner = i
                        self.rect_start_x = x
                        self.rect_start_y = y
                        return
                
                # Vérifier si le clic est à l'intérieur du rectangle
                if self.point_in_polygon(x, y, rect["corners"]):
                    self.dragging = True
                    self.drag_corner = "all"  # Déplacer tout le rectangle
                    self.rect_start_x = x
                    self.rect_start_y = y
                    return
    
    def on_canvas_drag(self, event):
        if not self.dragging or self.current_vehicle_id is None:
            return
        
        # Calculer le déplacement
        dx = event.x - self.rect_start_x
        dy = event.y - self.rect_start_y
        
        # Mettre à jour la position du/des coin(s)
        for rect in self.rectangles:
            if rect["id"] == self.current_vehicle_id:
                if self.drag_corner == "all":
                    # Déplacer tous les coins
                    for i in range(4):
                        rect["corners"][i][0] += dx
                        rect["corners"][i][1] += dy
                elif self.ctrl_pressed:
                    # Redimensionner le carré entier par rapport au centre
                    # Calculer le centre du carré
                    center_x = sum(corner[0] for corner in rect["corners"]) / 4
                    center_y = sum(corner[1] for corner in rect["corners"]) / 4
                    
                    # Calculer le vecteur du centre vers le coin en cours de déplacement
                    old_corner_x = rect["corners"][self.drag_corner][0] - dx
                    old_corner_y = rect["corners"][self.drag_corner][1] - dy
                    
                    # Calculer la distance entre le centre et le coin avant déplacement
                    old_dist = ((old_corner_x - center_x)**2 + (old_corner_y - center_y)**2)**0.5
                    
                    # Calculer la nouvelle position du coin
                    new_corner_x = rect["corners"][self.drag_corner][0] + dx
                    new_corner_y = rect["corners"][self.drag_corner][1] + dy
                    
                    # Calculer la nouvelle distance entre le centre et le coin
                    new_dist = ((new_corner_x - center_x)**2 + (new_corner_y - center_y)**2)**0.5
                    
                    if old_dist == 0:  # Éviter la division par zéro
                        scale_factor = 1
                    else:
                        scale_factor = new_dist / old_dist
                    
                    # Mettre à jour tous les coins en fonction du facteur d'échelle
                    for i in range(4):
                        # Vecteur du centre vers le coin
                        corner_x = rect["corners"][i][0]
                        corner_y = rect["corners"][i][1]
                        
                        # Appliquer le facteur d'échelle
                        rect["corners"][i][0] = center_x + (corner_x - center_x) * scale_factor
                        rect["corners"][i][1] = center_y + (corner_y - center_y) * scale_factor
                else:
                    # Déplacer un seul coin
                    rect["corners"][self.drag_corner][0] += dx
                    rect["corners"][self.drag_corner][1] += dy
                
                # Mettre à jour le rectangle sur le canvas
                self.original_canvas.coords(rect["canvas_id"],
                    rect["corners"][0][0], rect["corners"][0][1],
                    rect["corners"][1][0], rect["corners"][1][1],
                    rect["corners"][2][0], rect["corners"][2][1],
                    rect["corners"][3][0], rect["corners"][3][1]
                )
                
                # Mettre à jour les poignées
                for i, (x, y) in enumerate(rect["corners"]):
                    self.original_canvas.coords(rect["handles"][i], x-5, y-5, x+5, y+5)
                
                # Mettre à jour le texte d'orientation
                text_x = sum(corner[0] for corner in rect["corners"]) / 4
                text_y = sum(corner[1] for corner in rect["corners"]) / 4
                self.original_canvas.coords(rect["text_id"], text_x, text_y)
                
                # Mettre à jour le JSON
                self.json_image_info["vehicles"][str(self.current_vehicle_id)]["corners"] = rect["corners"]
                break
        
        # Mettre à jour la position de départ pour le prochain déplacement
        self.rect_start_x = event.x
        self.rect_start_y = event.y
    
    def on_canvas_release(self, event):
        if not self.dragging:
            return
        
        self.dragging = False
        self.drag_corner = None
    
    def point_in_polygon(self, x, y, polygon):
        """Vérifie si un point est à l'intérieur d'un polygone"""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n+1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def apply_orientation(self):
        """Applique l'orientation sélectionnée au véhicule en cours"""
        if self.current_vehicle_id is None:
            return
        
        orientation = self.orientation_var.get()
        
        # Mettre à jour l'orientation dans la liste des rectangles
        for rect in self.rectangles:
            if rect["id"] == self.current_vehicle_id:
                rect["orientation"] = orientation
                self.original_canvas.itemconfig(rect["text_id"], text=orientation)
                break
        
        # Mettre à jour le JSON
        self.json_image_info["vehicles"][str(self.current_vehicle_id)]["orientation"] = orientation
        
        self.status_var.set(f"Véhicule {self.current_vehicle_id}: orientation définie à '{orientation}'")
    
    def finish_vehicle(self):
        """Termine l'édition du véhicule en cours"""
        if self.current_vehicle_id is None:
            return
        
        # Désactiver le mode de dessin
        self.draw_mode = False
        
        # Changer la couleur du rectangle terminé
        for rect in self.rectangles:
            if rect["id"] == self.current_vehicle_id:
                self.original_canvas.itemconfig(rect["canvas_id"], outline="red")
                for handle_id in rect["handles"]:
                    self.original_canvas.itemconfig(handle_id, fill="red")
                self.original_canvas.itemconfig(rect["text_id"], fill="red")
                break
        
        # Réinitialiser
        self.current_vehicle_id = None
        
        # Désactiver les contrôles d'orientation
        self.orientation_menu["state"] = "disabled"
        self.apply_orientation_btn["state"] = "disabled"
        self.finish_vehicle_btn["state"] = "disabled"
        
        self.status_var.set("Véhicule terminé. Double-cliquez sur un véhicule pour l'éditer ou cliquez sur 'Ajouter une voiture'.")
    
    def export_json(self):
        """Exporte les données de toutes les images traitées au format JSON"""
        # Ajouter les données de l'image actuelle à all_json_data
        if self.image_path and self.json_image_info:
            self.all_json_data[self.image_path] = self.json_image_info.copy()
        
        # Vérifier s'il y a des données à exporter
        if not self.all_json_data:
            self.status_var.set("Aucune donnée à exporter")
            return
        
        # Compter le nombre total de véhicules annotés
        total_vehicles = 0
        for image_data in self.all_json_data.values():
            total_vehicles += len(image_data.get("vehicles", {}))
        
        if total_vehicles == 0:
            self.status_var.set("Ajoutez au moins un véhicule avant d'exporter")
            return
        
        # Préparer les données pour l'export
        export_data = {
            "image_folder": self.image_folder,
            "total_images": len(self.all_json_data),
            "total_vehicles": total_vehicles,
            "images": self.all_json_data
        }
        
        # Demander un fichier de sauvegarde
        save_path = filedialog.asksaveasfilename(defaultextension=".json", 
                                                filetypes=[("JSON files", "*.json")])
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(export_data, f, indent=4)
            self.status_var.set(f"Données de {len(self.all_json_data)} images exportées vers {save_path}")

    def on_canvas_double_click(self, event):
        """Gère le double-clic sur le canvas pour éditer un véhicule existant"""
        # Vérifier qu'une image est chargée
        if self.original_image is None:
            return
        
        # Si un véhicule est déjà en cours d'édition, le terminer d'abord
        if self.current_vehicle_id is not None:
            self.finish_vehicle()
        
        x, y = event.x, event.y
        
        # Parcourir tous les rectangles pour trouver celui qui a été cliqué
        for rect in self.rectangles:
            # Vérifier si le clic est à l'intérieur du rectangle ou sur une poignée
            if self.point_in_polygon(x, y, rect["corners"]):
                # Activer l'édition de ce véhicule
                self.activate_vehicle_editing(rect["id"])
                return
            
            # Vérifier si le clic est sur une poignée
            for i, (corner_x, corner_y) in enumerate(rect["corners"]):
                if abs(x - corner_x) <= 5 and abs(y - corner_y) <= 5:
                    # Activer l'édition de ce véhicule
                    self.activate_vehicle_editing(rect["id"])
                    return
    
    def activate_vehicle_editing(self, vehicle_id):
        """Active l'édition d'un véhicule existant"""
        # Définir l'ID du véhicule en cours d'édition
        self.current_vehicle_id = vehicle_id
        
        # Activer le mode de dessin
        self.draw_mode = True
        
        # Mettre à jour l'apparence des rectangles
        for rect in self.rectangles:
            color = "green" if rect["id"] == vehicle_id else "red"
            self.original_canvas.itemconfig(rect["canvas_id"], outline=color)
            for handle_id in rect["handles"]:
                self.original_canvas.itemconfig(handle_id, fill=color)
            self.original_canvas.itemconfig(rect["text_id"], fill=color)
        
        # Récupérer les données du véhicule
        vehicle_data = self.json_image_info["vehicles"].get(str(vehicle_id), {})
        
        # Mettre à jour les sliders avec les valeurs du véhicule
        if "min_depth" in vehicle_data:
            self.min_depth_slider.set(vehicle_data["min_depth"])
        if "max_depth" in vehicle_data:
            self.max_depth_slider.set(vehicle_data["max_depth"])
        
        # Mettre à jour le menu d'orientation
        if "orientation" in vehicle_data:
            self.orientation_var.set(vehicle_data["orientation"])
        
        # Activer les contrôles d'édition
        self.orientation_menu["state"] = "normal"
        self.apply_orientation_btn["state"] = "normal"
        self.finish_vehicle_btn["state"] = "normal"
        
        self.status_var.set(f"Édition du véhicule {vehicle_id}. Ajustez le rectangle et l'orientation.")

    def on_ctrl_press(self, event):
        """Gère l'appui sur la touche Ctrl"""
        self.ctrl_pressed = True
    
    def on_ctrl_release(self, event):
        """Gère le relâchement de la touche Ctrl"""
        self.ctrl_pressed = False

if __name__ == "__main__":
    root = tk.Tk()
    app = DepthMapApp(root)
    root.mainloop()