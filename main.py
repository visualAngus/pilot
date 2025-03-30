import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import time
import tkinter as tk
from tkinter import filedialog, Scale, Button, Label, Frame
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Ajoutez ces imports au début du fichier
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import patches
import json

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
    
    def compute_depth_map(self, image_path, output_path="depth_map", min_depth=None, max_depth=None, apply_colormap=False,save=False):
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

class Cube3D:
    def __init__(self, x=0, y=0, z=0, width=1, height=1, depth=1, rotation=0):
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth
        self.rotation_z = rotation  # Rotation en degrés autour de l'axe Z (plan de l'image)
        self.rotation_y = 0  # Rotation en degrés autour de l'axe Y (vertical)
        self.rotation_x = 0  # Rotation en degrés autour de l'axe X (horizontal)
        self.color = (1, 0, 0, 0.5)  # Rouge semi-transparent par défaut
        
    def get_vertices(self):
        # Renvoie les 8 sommets du cube avec rotation sur tous les axes
        # en utilisant un système de coordonnées local au cube
        x, y, z = self.x, self.y, self.z
        w, h, d = self.width, self.height, self.depth

        # Points de base sans rotation
        vertices_base = [
            [x, y, z], [x + w, y, z], [x + w, y + h, z], [x, y + h, z],  # Face avant
            [x, y, z + d], [x + w, y, z + d], [x + w, y + h, z + d], [x, y + h, z + d]  # Face arrière
        ]

        # Calculer le point central du cube pour faire tourner autour
        center_x = x + w/2
        center_y = y + h/2
        center_z = z + d/2

        # Convertir les angles de rotation en radians
        rot_x_rad = np.radians(self.rotation_x)
        rot_y_rad = np.radians(self.rotation_y)
        rot_z_rad = np.radians(self.rotation_z)

        # Matrices de rotation
        # Rotation autour des axes locaux, dans l'ordre X, Y, Z
        # L'ordre est important pour obtenir une rotation relative au cube
        sin_x, cos_x = np.sin(rot_x_rad), np.cos(rot_x_rad)
        sin_y, cos_y = np.sin(rot_y_rad), np.cos(rot_y_rad)
        sin_z, cos_z = np.sin(rot_z_rad), np.cos(rot_z_rad)

        vertices = []
        for vx, vy, vz in vertices_base:
            # Translater au centre du cube
            dx = vx - center_x
            dy = vy - center_y
            dz = vz - center_z
            
            # Appliquer les rotations dans cet ordre: Z d'abord, puis Y, puis X
            # Cet ordre garantit que les axes de rotation sont relatifs au cube
            
            # Rotation autour de Z
            dx_temp = dx * cos_z - dy * sin_z
            dy = dx * sin_z + dy * cos_z
            dx = dx_temp
            
            # Rotation autour de Y
            dx_temp = dx * cos_y + dz * sin_y
            dz = -dx * sin_y + dz * cos_y
            dx = dx_temp
            
            # Rotation autour de X
            dy_temp = dy * cos_x - dz * sin_x
            dz = dy * sin_x + dz * cos_x
            dy = dy_temp
            
            # Translater de retour
            vertices.append([dx + center_x, dy + center_y, dz + center_z])

        return vertices
    
    def get_local_axes(self):
        """
        Renvoie les points pour dessiner les axes locaux du cube
        """
        # Centre du cube
        center_x = self.x + self.width/2
        center_y = self.y + self.height/2
        center_z = self.z + self.depth/2
        
        # Longueur des axes
        axis_length = max(self.width, self.height, self.depth) * 0.75
        
        # Points des axes locaux sans rotation
        x_axis = [center_x, center_y, center_z, center_x + axis_length, center_y, center_z]
        y_axis = [center_x, center_y, center_z, center_x, center_y + axis_length, center_z]
        z_axis = [center_x, center_y, center_z, center_x, center_y, center_z + axis_length]
        
        # Convertir les angles de rotation en radians
        rot_x_rad = np.radians(self.rotation_x)
        rot_y_rad = np.radians(self.rotation_y)
        rot_z_rad = np.radians(self.rotation_z)
        
        # Matrices de rotation
        sin_x, cos_x = np.sin(rot_x_rad), np.cos(rot_x_rad)
        sin_y, cos_y = np.sin(rot_y_rad), np.cos(rot_y_rad)
        sin_z, cos_z = np.sin(rot_z_rad), np.cos(rot_z_rad)
        
        # Fonction pour appliquer la rotation à un point
        def rotate_point(px, py, pz):
            # Translater au centre
            dx = px - center_x
            dy = py - center_y
            dz = pz - center_z
            
            # Rotation Z
            dx_temp = dx * cos_z - dy * sin_z
            dy = dx * sin_z + dy * cos_z
            dx = dx_temp
            
            # Rotation Y
            dx_temp = dx * cos_y + dz * sin_y
            dz = -dx * sin_y + dz * cos_y
            dx = dx_temp
            
            # Rotation X
            dy_temp = dy * cos_x - dz * sin_x
            dz = dy * sin_x + dz * cos_x
            dy = dy_temp
            
            # Translater de retour
            return [dx + center_x, dy + center_y, dz + center_z]
        
        # Appliquer les rotations aux points des axes
        x_axis_end = rotate_point(x_axis[3], x_axis[4], x_axis[5])
        y_axis_end = rotate_point(y_axis[3], y_axis[4], y_axis[5])
        z_axis_end = rotate_point(z_axis[3], z_axis[4], z_axis[5])
        
        # Renvoyer les axes sous forme de segments [début, fin]
        return [
            [[center_x, center_y, center_z], x_axis_end],  # axe X (rouge)
            [[center_x, center_y, center_z], y_axis_end],  # axe Y (vert)
            [[center_x, center_y, center_z], z_axis_end]   # axe Z (bleu)
        ]
    
    def get_faces(self):
        # Définit les 6 faces du cube par les indices des sommets
        faces = [
            [0, 1, 2, 3],  # face avant
            [4, 5, 6, 7],  # face arrière
            [0, 1, 5, 4],  # face inférieure
            [2, 3, 7, 6],  # face supérieure
            [0, 3, 7, 4],  # face gauche
            [1, 2, 6, 5]   # face droite
        ]
        return faces
    
    def to_dict(self):
        return {
            "position": [self.x, self.y, self.z],
            "dimensions": [self.width, self.height, self.depth],
            "rotation": [self.rotation_x, self.rotation_y, self.rotation_z],
            "color": self.color
        }

class DepthMapApp:
    def __init__(self, root):
        # Garder le code existant et ajouter:
        self.root = root
        self.root.title("Éditeur de cubes 3D pour véhicules")
        self.root.geometry("1500x1000")
        
        # Ajouter ces attributs à la liste existante
        self.cubes = []  # Liste des cubes 3D
        self.selected_cube = None  # Cube actuellement sélectionné
        self.edit_mode = False  # Mode d'édition actif
        self.click_position = None  # Position du dernier clic
        
        # Ajouter un compteur de clics et un cube temporaire
        self.click_count = 0
        self.temp_cube_data = {}
        
        self.depth_generator = DepthMapGenerator()
        self.image_path = None
        self.current_depth_map = None
        self.original_image = None
        self.vehicule_count = 0
        self.json_image_info = {}
        
        # Frame principale
        main_frame = Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame pour les contrôles
        controls_frame = Frame(main_frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        # Bouton pour sélectionner une image
        self.select_btn = Button(controls_frame, text="Sélectionner une image", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
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
        
        # Frame pour les contrôles de profondeur du cube - DÉPLACÉ AVANT disable_cube_controls()
        self.cube_depth_frame = Frame(main_frame)
        self.cube_depth_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Label pour la profondeur du cube
        self.cube_depth_label = Label(self.cube_depth_frame, text="Profondeur du cube (m):")
        self.cube_depth_label.pack(side=tk.LEFT, padx=5)
        
        # Slider pour la profondeur du cube (en mètres)
        self.cube_depth_slider = Scale(self.cube_depth_frame, from_=0.1, to=10.0, orient=tk.HORIZONTAL,
                                       resolution=0.1, length=300, command=self.update_cube_depth)
        self.cube_depth_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.cube_depth_slider.set(1.0)  # Valeur par défaut: 1 mètre
        
        # Frame pour les contrôles de rotation du cube
        self.rotation_frame = Frame(main_frame)
        self.rotation_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Sliders pour la rotation X, Y, Z
        # Rotation X
        self.rotation_x_label = Label(self.rotation_frame, text="Rotation X:")
        self.rotation_x_label.pack(side=tk.LEFT, padx=5)
        self.rotation_x_slider = Scale(self.rotation_frame, from_=-360, to=360, orient=tk.HORIZONTAL,
                           resolution=1, length=150, command=self.update_rotation)
        self.rotation_x_slider.set(0)  # Default value of 0
        self.rotation_x_slider.pack(side=tk.LEFT, padx=5)
        
        # Rotation Y
        self.rotation_y_label = Label(self.rotation_frame, text="Rotation Y:")
        self.rotation_y_label.pack(side=tk.LEFT, padx=5)
        self.rotation_y_slider = Scale(self.rotation_frame, from_=-360, to=360, orient=tk.HORIZONTAL,
                                       resolution=1, length=150, command=self.update_rotation)
        self.rotation_y_slider.set(0)  # Default value of 0
        self.rotation_y_slider.pack(side=tk.LEFT, padx=5)
        
        # Rotation Z
        self.rotation_z_label = Label(self.rotation_frame, text="Rotation Z:")
        self.rotation_z_label.pack(side=tk.LEFT, padx=5)
        self.rotation_z_slider = Scale(self.rotation_frame, from_=-360, to=360, orient=tk.HORIZONTAL,
                                       resolution=1, length=150, command=self.update_rotation)
        self.rotation_z_slider.set(0)  # Default value of 0
        self.rotation_z_slider.pack(side=tk.LEFT, padx=5)
        
        # Frame pour les images
        self.images_frame = Frame(main_frame)
        self.images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Label pour l'image originale
        self.original_label = Label(self.images_frame, text="Image originale")
        self.original_label.grid(row=0, column=0, pady=5)
        self.original_display = Label(self.images_frame)
        self.original_display.grid(row=1, column=0, padx=10, pady=10)
        
        # Label pour la carte de profondeur
        self.depth_label = Label(self.images_frame, text="Carte de profondeur")
        self.depth_label.grid(row=0, column=1, pady=5)
        self.depth_display = Label(self.images_frame)
        self.depth_display.grid(row=1, column=1, padx=10, pady=10)
        
        # Ajouter un Label pour l'affichage de l'overlay des cubes
        self.cube_overlay_label = Label(self.images_frame, text="Overlay des cubes")
        self.cube_overlay_label.grid(row=0, column=2, pady=5)
        self.cube_overlay_display = Label(self.images_frame)
        self.cube_overlay_display.grid(row=1, column=2, padx=10, pady=10)

        # Bouton sous les images
        self.save_btn = Button(self.images_frame, text="ajouter un véhicule", command=self.add_new_vehicule)
        self.save_btn.grid(row=2, column=0, columnspan=2, pady=10)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Sélectionnez une image pour commencer")
        self.status_bar = Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Ajouter un frame pour la visualisation 3D
        self.view_frame = Frame(main_frame)
        self.view_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=10)
        
        # Créer la figure matplotlib pour la visualisation 3D
        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z (profondeur)')
        self.ax.set_title('Vue 3D des véhicules')
        
        # Optionnel: Définir des graduations spécifiques sur l'axe Z
        self.ax.set_zticks([0, 50, 100, 150, 200])
        self.ax.set_zticklabels(['0m', '1m', '2m', '3m', '4m'])  # Conversion en mètres (échelle 50:1)
        
        # Intégrer la figure dans tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.view_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Frame pour les contrôles du cube
        self.cube_controls = Frame(self.view_frame)
        self.cube_controls.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Boutons pour le cube
        self.add_cube_btn = Button(self.cube_controls, text="Ajouter un cube", command=self.add_cube)
        self.add_cube_btn.pack(side=tk.LEFT, padx=5)
        
        self.edit_cube_btn = Button(self.cube_controls, text="Éditer le cube", command=self.toggle_edit_mode)
        self.edit_cube_btn.pack(side=tk.LEFT, padx=5)
        
        self.delete_cube_btn = Button(self.cube_controls, text="Supprimer le cube", command=self.delete_selected_cube)
        self.delete_cube_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_btn = Button(self.cube_controls, text="Exporter les cubes", command=self.export_cubes)
        self.export_btn.pack(side=tk.RIGHT, padx=5)
        
        # Événements souris sur l'image pour placer des cubes
        self.original_display.bind("<Button-1>", self.on_image_click)
        self.depth_display.bind("<Button-1>", self.on_depth_click)
        
        # Configurer les interactions pour le cube
        self.setup_cube_interaction()
        
        # Désactiver les sliders au début
        self.min_depth_slider["state"] = "disabled"
        self.max_depth_slider["state"] = "disabled"
        self.colormap_check["state"] = "disabled"
        self.cube_depth_slider["state"] = "disabled"  # Désactiver le slider de profondeur
        self.rotation_x_slider["state"] = "disabled"  # Désactiver les sliders de rotation
        self.rotation_y_slider["state"] = "disabled"
        self.rotation_z_slider["state"] = "disabled"
        
        # Ajouter ces attributs pour la définition de la perspective
        self.perspective_mode = False
        self.horizon_points = []  # Points définissant la ligne d'horizon
        self.vanishing_points = []  # Points définissant la ligne de fuite (perspective)
        self.vertical_points = []  # Points définissant la ligne verticale
        self.current_line_type = None  # 'horizon', 'vanishing', 'vertical'
        self.perspective_defined = False  # Indique si la perspective a été définie
        
        # Frame pour les contrôles de perspective
        self.perspective_frame = Frame(main_frame)
        self.perspective_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Boutons pour définir les lignes de perspective
        self.define_perspective_btn = Button(self.perspective_frame, text="Définir perspective", command=self.toggle_perspective_mode)
        self.define_perspective_btn.pack(side=tk.LEFT, padx=5)
        
        self.define_horizon_btn = Button(self.perspective_frame, text="Définir horizon", command=lambda: self.set_line_type('horizon'))
        self.define_horizon_btn.pack(side=tk.LEFT, padx=5)
        self.define_horizon_btn["state"] = "disabled"
        
        self.define_vanishing_btn = Button(self.perspective_frame, text="Définir fuite", command=lambda: self.set_line_type('vanishing'))
        self.define_vanishing_btn.pack(side=tk.LEFT, padx=5)
        self.define_vanishing_btn["state"] = "disabled"
        
        self.define_vertical_btn = Button(self.perspective_frame, text="Définir verticale", command=lambda: self.set_line_type('vertical'))
        self.define_vertical_btn.pack(side=tk.LEFT, padx=5)
        self.define_vertical_btn["state"] = "disabled"
        
        self.reset_perspective_btn = Button(self.perspective_frame, text="Réinitialiser", command=self.reset_perspective)
        self.reset_perspective_btn.pack(side=tk.LEFT, padx=5)
        self.reset_perspective_btn["state"] = "disabled"
        
        # Désactiver les contrôles de cube au début
        # self.disable_cube_controls()
        
    def select_image(self):
        # Garder le code existant et ajouter:
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            self.status_var.set(f"Image sélectionnée: {self.image_path}")
            
            # Réinitialiser les cubes
            self.cubes = []
            self.selected_cube = None
            
            # Afficher l'image originale
            self.json_image_info = {"path": self.image_path}
            self.original_image = cv2.imread(self.image_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.original_image, self.original_display)
            self.compute_depth()
        
        if self.original_image is not None:
            self.update_cube_overlay()
    
    def compute_depth(self):
        # Garder le code existant et ajouter:
        if self.image_path:
            self.status_var.set("Calcul de la carte de profondeur en cours...")
            self.root.update()
            
            t = time.time()
            self.current_depth_map, self.output_path = self.depth_generator.compute_depth_map(self.image_path, save=True, apply_colormap=self.use_colormap.get())
            
            # Afficher la carte de profondeur
            self.display_image(self.current_depth_map, self.depth_display)
            
            execution_time = time.time() - t
            self.status_var.set(f"Carte de profondeur calculée en {execution_time:.2f} secondes")
            
            # Activer les sliders et contrôles de cube
            self.min_depth_slider["state"] = "normal"
            self.max_depth_slider["state"] = "normal"
            self.colormap_check["state"] = "normal"
            self.enable_cube_controls()
            
            # Initialiser la vue 3D
            self.update_3d_view()
    
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
                apply_colormap=self.use_colormap.get()
            )

            if self.vehicule_count != 0:
                self.json_image_info["vehicule"+str(self.vehicule_count)]["min_depth"] = min_depth
                self.json_image_info["vehicule"+str(self.vehicule_count)]["max_depth"] = max_depth

                print(self.json_image_info)
            
            # Afficher la nouvelle carte de profondeur
            self.display_image(depth_map, self.depth_display)
    
    def add_new_vehicule(self):
        # Ajouter un nouveau véhicule avec une approche simplifiée
        self.vehicule_count += 1
        self.json_image_info[f"vehicule{self.vehicule_count}"] = {
            "min_depth": self.min_depth_slider.get(),
            "max_depth": self.max_depth_slider.get()
        }
        self.status_var.set(f"Véhicule {self.vehicule_count} ajouté. Cliquez sur l'image pour placer le cube.")
        
        # Activer le mode d'édition pour ajouter un cube
        self.edit_mode = True
        self.selected_cube = None  # On va créer un nouveau cube avec un clic
        
        # Activer les contrôles
        self.enable_cube_controls()
        self.cube_depth_slider["state"] = "normal"

    def save_depth_map(self):
        if self.current_depth_map is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                     filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if save_path:
                cv2.imwrite(save_path, self.current_depth_map)
                self.status_var.set(f"Carte de profondeur sauvegardée: {save_path}")
        else:
            self.status_var.set("Aucune carte de profondeur à sauvegarder")


    def display_image(self, img, label, max_size=(400, 400)):
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

    # Ajouter ces nouvelles méthodes
    
    def on_image_click(self, event):
        # Vérifier si nous sommes en mode définition de perspective
        if self.perspective_mode and self.current_line_type:
            # Convertir les coordonnées du clic en coordonnées d'image
            x, y = self.convert_display_to_image_coords(event.x, event.y, self.original_display)
            
            if self.current_line_type == 'horizon':
                # Ajouter le point à la ligne d'horizon
                self.horizon_points.append((x, y))
                if len(self.horizon_points) == 1:
                    self.status_var.set("Premier point de l'horizon défini. Cliquez sur un second point.")
                elif len(self.horizon_points) == 2:
                    self.status_var.set("Horizon défini. Définissez maintenant la ligne de fuite.")
                    self.set_line_type('vanishing')
            
            elif self.current_line_type == 'vanishing':
                # Ajouter le point à la ligne de fuite
                self.vanishing_points.append((x, y))
                if len(self.vanishing_points) == 1:
                    self.status_var.set("Premier point de la ligne de fuite défini. Cliquez sur un second point.")
                elif len(self.vanishing_points) == 2:
                    self.status_var.set("Ligne de fuite définie. Définissez maintenant une ligne verticale.")
                    self.set_line_type('vertical')
            
            elif self.current_line_type == 'vertical':
                # Ajouter le point à la ligne verticale
                self.vertical_points.append((x, y))
                if len(self.vertical_points) == 1:
                    self.status_var.set("Premier point de la verticale défini. Cliquez sur un second point.")
                elif len(self.vertical_points) == 2:
                    self.status_var.set("Ligne verticale définie.")
                    
                    # Vérifier si la perspective est complète
                    if self.check_perspective_completion():
                        # Calculer la matrice de perspective
                        perspective_data = self.calculate_perspective_matrix()
                        if perspective_data:
                            self.status_var.set(f"Perspective définie! Angles - Horizon: {perspective_data['horizon_angle']:.1f}°, Verticale: {perspective_data['vertical_angle']:.1f}°")
            
            # Mettre à jour l'affichage
            self.update_cube_overlay()
            
            # Vérifier si nous avons terminé de définir un type de ligne
            if ((self.current_line_type == 'horizon' and len(self.horizon_points) == 2) or
                (self.current_line_type == 'vanishing' and len(self.vanishing_points) == 2) or
                (self.current_line_type == 'vertical' and len(self.vertical_points) == 2)):
                
                # Si toutes les lignes sont définies, désactiver le mode perspective
                if self.check_perspective_completion():
                    self.current_line_type = None
            
            return
            
        # Si on n'est pas en mode perspective, le comportement normal
        if not self.edit_mode or self.original_image is None:
            return
            
        # Vérifier si la perspective a été définie avant de permettre l'ajout de cubes
        if not self.perspective_defined:
            self.status_var.set("Vous devez d'abord définir la perspective avant d'ajouter des cubes.")
            return
            
        # Convertir les coordonnées du clic en coordonnées d'image
        x, y = self.convert_display_to_image_coords(event.x, event.y, self.original_display)
        self.process_coordinate_click(x, y)
        
    def on_depth_click(self, event):
        if not self.edit_mode or self.selected_cube is None:
            return
            
        # Convertir les coordonnées du clic en coordonnées d'image
        x, y = self.convert_display_to_image_coords(event.x, event.y, self.depth_display)
        
        # Mettre à jour la profondeur du cube sélectionné
        depth = self.estimate_depth_at_point(x, y)
        self.selected_cube.depth = abs(depth - self.selected_cube.z)
        
        # Mettre à jour les informations du véhicule
        if self.vehicule_count > 0:
            vehicle_key = f"vehicule{self.vehicule_count}"
            if vehicle_key in self.json_image_info:
                self.json_image_info[vehicle_key]["cube"] = self.selected_cube.to_dict()
        
        self.status_var.set(f"Profondeur du cube mise à jour: {self.selected_cube.depth:.2f}")
        
        # Mettre à jour l'affichage 3D
        self.update_3d_view()
    
    def convert_display_to_image_coords(self, display_x, display_y, label):
        # Convertit les coordonnées d'affichage en coordonnées d'image originale
        if hasattr(label, 'image') and label.image:
            # Obtenir les dimensions actuelles de l'affichage
            display_width = label.winfo_width()
            display_height = label.winfo_height()
            
            # Obtenir les dimensions de l'image originale
            if self.original_image is not None:
                orig_height, orig_width = self.original_image.shape[:2]
                
                # Calculer le facteur d'échelle
                scale_x = orig_width / display_width
                scale_y = orig_height / display_height
                
                # Convertir les coordonnées
                image_x = int(display_x * scale_x)
                image_y = int(display_y * scale_y)
                
                return image_x, image_y
        
        return 0, 0
    
    def estimate_depth_at_point(self, x, y):
        # Estime la profondeur à partir de la carte de profondeur
        if self.depth_generator.raw_depth is not None:
            # Assurez-vous que les coordonnées sont dans les limites
            h, w = self.depth_generator.raw_depth.shape
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            
            # Récupérer la valeur de profondeur normalisée (entre 0 et 1)
            normalized_depth = self.depth_generator.raw_depth[y, x]
            
            # Convertir en valeur plus appropriée pour la visualisation 3D
            # (on multiplie par un facteur pour que la profondeur soit visible)
            return normalized_depth * 200  # Ajustez le facteur selon vos besoins
        
        return 100  # Valeur par défaut
    
    def update_3d_view(self):
        # Mettre à jour la visualisation 3D
        self.ax.clear()
        
        # Définir les limites de la vue et orientation des axes
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(0, h)
            self.ax.set_zlim(0, 200)  # Limite de profondeur par défaut
            
            # Ajuster l'orientation des axes pour un système de coordonnées plus standard
            # X: horizontal (de gauche à droite)
            # Y: vertical (de bas en haut - donc on inverse)
            # Z: profondeur (de l'écran vers l'extérieur)
            
            # Définir les limites de la profondeur (Z)
            # if self.depth_generator.raw_depth is not None:
            #     z_min = np.min(self.depth_generator.raw_depth) * 200
            #     z_max = np.max(self.depth_generator.raw_depth) * 200
            #     self.ax.set_zlim(z_min, z_max)
            
            # Ajuster les étiquettes des axes pour clarifier l'orientation
            self.ax.set_xlabel('X (horizontal)')
            self.ax.set_ylabel('Y (vertical)')
            self.ax.set_zlabel('Z (profondeur)')
        
        # Dessiner chaque cube
        for i, cube in enumerate(self.cubes):
            vertices = cube.get_vertices()
            faces = cube.get_faces()
            
            # Dessiner les faces du cube
            for face in faces:
                face_vertices = [vertices[i] for i in face]
                face_vertices.append(face_vertices[0])  # Fermer la face
                xs, ys, zs = zip(*face_vertices)
                self.ax.plot(xs, ys, zs, color='r')
            
            # Si c'est le cube sélectionné, ajouter une marque et ses axes locaux
            if cube == self.selected_cube:
                x, y, z = cube.x, cube.y, cube.z
                self.ax.scatter([x], [y], [z], color='b', s=100, marker='o')
                
                # Obtenir et dessiner les axes locaux du cube
                local_axes = cube.get_local_axes()
                
                # Dessiner l'axe X en rouge
                x_axis = local_axes[0]
                self.ax.plot([x_axis[0][0], x_axis[1][0]], 
                             [x_axis[0][1], x_axis[1][1]], 
                             [x_axis[0][2], x_axis[1][2]], 
                             color='r', linewidth=2)
                
                # Dessiner l'axe Y en vert
                y_axis = local_axes[1]
                self.ax.plot([y_axis[0][0], y_axis[1][0]], 
                             [y_axis[0][1], y_axis[1][1]], 
                             [y_axis[0][2], y_axis[1][2]], 
                             color='g', linewidth=2)
                
                # Dessiner l'axe Z en bleu
                z_axis = local_axes[2]
                self.ax.plot([z_axis[0][0], z_axis[1][0]], 
                             [z_axis[0][1], z_axis[1][1]], 
                             [z_axis[0][2], z_axis[1][2]], 
                             color='b', linewidth=2)
                
                # Ajouter les labels X, Y, Z au bout des axes
                self.ax.text(x_axis[1][0], x_axis[1][1], x_axis[1][2], "X", color='r')
                self.ax.text(y_axis[1][0], y_axis[1][1], y_axis[1][2], "Y", color='g')
                self.ax.text(z_axis[1][0], z_axis[1][1], z_axis[1][2], "Z", color='b')
                
                # Afficher les angles de rotation
                self.ax.text(x, y, z, f"Rot X: {cube.rotation_x:.1f}°\nRot Y: {cube.rotation_y:.1f}°\nRot Z: {cube.rotation_z:.1f}°", color='g')
        
        # Ajouter une légende
        if self.cubes:
            self.ax.set_title(f'{len(self.cubes)} cube(s) placé(s)')
        else:
            self.ax.set_title('Cliquez sur l\'image pour placer un cube')
        
        # Ajuster la perspective pour une meilleure visualisation 3D
        self.ax.view_init(elev=30, azim=45)  # élévation et azimut pour une vue en perspective standard
        
        # Mettre à jour la figure
        self.canvas.draw()
        self.update_cube_overlay()  # Mettre à jour la vue avec les cubes projetés
    
    def add_cube(self):
        # Activer le mode d'édition pour ajouter un cube
        self.edit_mode = True
        self.selected_cube = None
        self.click_count = 0  # Réinitialiser le compteur de clics
        self.status_var.set("Cliquez sur l'image pour placer l'ancre initiale du cube")
        self.edit_cube_btn.config(text="Annuler l'édition")
    
    def toggle_edit_mode(self):
        # Basculer le mode d'édition
        self.edit_mode = not self.edit_mode
        
        if self.edit_mode:
            self.edit_cube_btn.config(text="Annuler l'édition")
            self.status_var.set("Mode édition activé - Cliquez pour éditer")
            
            # Activer les sliders si un cube est sélectionné
            if self.selected_cube is not None:
                self.cube_depth_slider["state"] = "normal"
                self.rotation_x_slider["state"] = "normal"
                self.rotation_y_slider["state"] = "normal"
                self.rotation_z_slider["state"] = "normal"
        else:
            self.edit_cube_btn.config(text="Éditer le cube")
            self.selected_cube = None
            self.click_count = 0  # Réinitialiser le compteur de clics
            self.status_var.set("Mode édition désactivé")
            
            # Désactiver tous les sliders
            self.cube_depth_slider["state"] = "disabled"
            self.rotation_x_slider["state"] = "disabled"
            self.rotation_y_slider["state"] = "disabled"
            self.rotation_z_slider["state"] = "disabled"
    
    def delete_selected_cube(self):
        # Supprimer le cube sélectionné
        if self.selected_cube in self.cubes:
            self.cubes.remove(self.selected_cube)
            
            # Mettre à jour les informations du véhicule
            if self.vehicule_count > 0:
                vehicle_key = f"vehicule{self.vehicule_count}"
                if vehicle_key in self.json_image_info and "cube" in self.json_image_info[vehicle_key]:
                    del self.json_image_info[vehicle_key]["cube"]
            
            self.selected_cube = None
            self.update_3d_view()
            self.status_var.set("Cube supprimé")
            
            # Désactiver tous les sliders
            self.cube_depth_slider["state"] = "disabled"
            self.rotation_x_slider["state"] = "disabled"
            self.rotation_y_slider["state"] = "disabled"
            self.rotation_z_slider["state"] = "disabled"
    
    def export_cubes(self):
        # Exporter les informations des cubes
        if not self.cubes:
            self.status_var.set("Aucun cube à exporter")
            return
        
        export_data = {
            "image_path": self.image_path,
            "vehicles": {}
        }
        
        # Parcourir les véhicules dans json_image_info
        for key, value in self.json_image_info.items():
            if key.startswith("vehicule") and "cube" in value:
                export_data["vehicles"][key] = {
                    "cube": value["cube"],
                    "min_depth": value.get("min_depth", 0),
                    "max_depth": value.get("max_depth", 1)
                }
        
        # Sauvegarder dans un fichier JSON
        save_path = filedialog.asksaveasfilename(defaultextension=".json", 
                                                filetypes=[("JSON files", "*.json")])
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(export_data, f, indent=4)
            self.status_var.set(f"Données exportées vers {save_path}")
    
    def enable_cube_controls(self):
        self.add_cube_btn["state"] = "normal"
        self.edit_cube_btn["state"] = "normal"
        self.delete_cube_btn["state"] = "normal"
        self.export_btn["state"] = "normal"
        
        # Activer les sliders uniquement si un cube est sélectionné
        if self.selected_cube is not None:
            self.cube_depth_slider["state"] = "normal"
            self.rotation_x_slider["state"] = "normal"
            self.rotation_y_slider["state"] = "normal"
            self.rotation_z_slider["state"] = "normal"
            
            # Mettre à jour les sliders avec les valeurs actuelles du cube
            self.rotation_x_slider.set(self.selected_cube.rotation_x)
            self.rotation_y_slider.set(self.selected_cube.rotation_y)
            self.rotation_z_slider.set(self.selected_cube.rotation_z)
        
        # Activer le bouton de définition de perspective
        self.define_perspective_btn["state"] = "normal"
    
    def disable_cube_controls(self):
        self.add_cube_btn["state"] = "disabled"
        self.edit_cube_btn["state"] = "disabled"
        self.delete_cube_btn["state"] = "disabled"
        self.export_btn["state"] = "disabled"
        self.cube_depth_slider["state"] = "disabled"
        self.rotation_x_slider["state"] = "disabled"
        self.rotation_y_slider["state"] = "disabled"
        self.rotation_z_slider["state"] = "disabled"
        
        # Désactiver tous les boutons de perspective
        self.define_perspective_btn["state"] = "disabled"
        self.define_horizon_btn["state"] = "disabled"
        self.define_vanishing_btn["state"] = "disabled"
        self.define_vertical_btn["state"] = "disabled"
        self.reset_perspective_btn["state"] = "disabled"

    def update_cube_overlay(self):
        # Créer une image avec les cubes projetés
        if self.original_image is None:
            return

        # Copier l'image originale
        overlay_img = self.original_image.copy()
        
        # Dessiner les lignes de perspective si elles existent
        if self.perspective_mode or self.perspective_defined:
            # Dessiner la ligne d'horizon (en bleu)
            if len(self.horizon_points) == 2:
                cv2.line(overlay_img, 
                        (int(self.horizon_points[0][0]), int(self.horizon_points[0][1])), 
                        (int(self.horizon_points[1][0]), int(self.horizon_points[1][1])), 
                        (255, 0, 0), 2)  # Rouge
                cv2.putText(overlay_img, "Horizon", 
                           (int(self.horizon_points[0][0]), int(self.horizon_points[0][1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Dessiner la ligne de fuite (en vert)
            if len(self.vanishing_points) == 2:
                cv2.line(overlay_img, 
                        (int(self.vanishing_points[0][0]), int(self.vanishing_points[0][1])), 
                        (int(self.vanishing_points[1][0]), int(self.vanishing_points[1][1])), 
                        (0, 255, 0), 2)  # Vert
                cv2.putText(overlay_img, "Fuite", 
                           (int(self.vanishing_points[0][0]), int(self.vanishing_points[0][1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Dessiner la ligne verticale (en rouge)
            if len(self.vertical_points) == 2:
                cv2.line(overlay_img, 
                        (int(self.vertical_points[0][0]), int(self.vertical_points[0][1])), 
                        (int(self.vertical_points[1][0]), int(self.vertical_points[1][1])), 
                        (0, 0, 255), 2)  # Bleu
                cv2.putText(overlay_img, "Verticale", 
                           (int(self.vertical_points[0][0]), int(self.vertical_points[0][1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Si nous sommes en train de tracer une ligne, dessiner le premier point
            if self.current_line_type == 'horizon' and len(self.horizon_points) == 1:
                cv2.circle(overlay_img, 
                          (int(self.horizon_points[0][0]), int(self.horizon_points[0][1])), 
                          5, (255, 0, 0), -1)
            elif self.current_line_type == 'vanishing' and len(self.vanishing_points) == 1:
                cv2.circle(overlay_img, 
                          (int(self.vanishing_points[0][0]), int(self.vanishing_points[0][1])), 
                          5, (0, 255, 0), -1)
            elif self.current_line_type == 'vertical' and len(self.vertical_points) == 1:
                cv2.circle(overlay_img, 
                          (int(self.vertical_points[0][0]), int(self.vertical_points[0][1])), 
                          5, (0, 0, 255), -1)

        # Dessiner chaque cube
        for i, cube in enumerate(self.cubes):
            # Obtenir les sommets du cube
            vertices = cube.get_vertices()

            # Projeter les sommets 3D en 2D avec perspective
            projected_vertices = self.project_vertices(vertices)

            # Extraire les points de la face avant et arrière
            front_face = projected_vertices[:4]  # Les 4 premiers sommets sont la face avant
            back_face = projected_vertices[4:]   # Les 4 derniers sommets sont la face arrière

            # Déterminer les faces visibles selon la rotation
            # Ici on simplifie en montrant toutes les faces, mais une approche plus sophistiquée
            # consisterait à calculer quelles faces sont visibles selon l'angle de rotation

            # Convertir en points pour OpenCV
            front_points = np.array([(int(v[0]), int(v[1])) for v in front_face], np.int32)
            back_points = np.array([(int(v[0]), int(v[1])) for v in back_face], np.int32)

            # Couleur selon si le cube est sélectionné
            color = (0, 255, 0) if cube == self.selected_cube else (255, 0, 0)

            # Dessiner toutes les faces avec un peu de transparence
            faces = [
                front_points,  # face avant
                back_points,   # face arrière
                np.array([(int(front_face[0][0]), int(front_face[0][1])), 
                          (int(front_face[1][0]), int(front_face[1][1])),
                          (int(back_face[1][0]), int(back_face[1][1])),
                          (int(back_face[0][0]), int(back_face[0][1]))], np.int32),  # face inférieure
                np.array([(int(front_face[2][0]), int(front_face[2][1])), 
                          (int(front_face[3][0]), int(front_face[3][1])),
                          (int(back_face[3][0]), int(back_face[3][1])),
                          (int(back_face[2][0]), int(back_face[2][1]))], np.int32),  # face supérieure
                np.array([(int(front_face[0][0]), int(front_face[0][1])), 
                          (int(front_face[3][0]), int(front_face[3][1])),
                          (int(back_face[3][0]), int(back_face[3][1])),
                          (int(back_face[0][0]), int(back_face[0][1]))], np.int32),  # face gauche
                np.array([(int(front_face[1][0]), int(front_face[1][1])),
                          (int(front_face[2][0]), int(front_face[2][1])),
                          (int(back_face[2][0]), int(back_face[2][1])),
                          (int(back_face[1][0]), int(back_face[1][1]))], np.int32),  # face droite
            ]

            # Dessiner les faces avec couleur semi-transparente
            face_overlay = overlay_img.copy()
            for face_points in faces:
                cv2.fillPoly(face_overlay, [face_points], color)
            
            # Appliquer la transparence
            alpha = 0.3  # Taux de transparence
            cv2.addWeighted(face_overlay, alpha, overlay_img, 1 - alpha, 0, overlay_img)
            
            # Dessiner les contours des faces en lignes plus visibles
            for face_points in faces:
                cv2.polylines(overlay_img, [face_points], True, color, 2)

            # Dessiner des poignées de contrôle si le cube est sélectionné
            if cube == self.selected_cube:
                # Centre du cube pour les contrôles de rotation
                center_x = sum(p[0] for p in front_face) / 4
                center_y = sum(p[1] for p in front_face) / 4
                
                # Dessiner un indicateur de centre
                cv2.circle(overlay_img, (int(center_x), int(center_y)), 4, (255, 255, 255), -1)
                
                # Nous avons supprimé les poignées de rotation puisque nous utilisons maintenant des sliders
                
                # Afficher les valeurs de rotation
                info_text = f"X:{cube.rotation_x:.1f}° Y:{cube.rotation_y:.1f}° Z:{cube.rotation_z:.1f}°"
                cv2.putText(overlay_img, info_text, (int(front_points[0][0]), int(front_points[0][1]) - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Dimensions
                dim_text = f"W:{cube.width:.1f} H:{cube.height:.1f} D:{cube.depth:.1f}"
                cv2.putText(overlay_img, dim_text, (int(front_points[0][0]), int(front_points[0][1]) - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Dessiner les sommets du cube pour faciliter la sélection
                for vertex_idx, vertex in enumerate(projected_vertices):
                    vertex_color = (0, 255, 255) if vertex_idx < 4 else (100, 255, 255)  # Front/Back faces
                    cv2.circle(overlay_img, (int(vertex[0]), int(vertex[1])), 5, vertex_color, -1)
            
                # Afficher les axes locaux comme guides
                local_axes = cube.get_local_axes()
                
                # Calculer les projections 2D des axes
                center = local_axes[0][0]  # Le point central est le même pour tous les axes
                center_2d = self.project_point(center[0], center[1], center[2])
                
                # Axe X (rouge)
                x_end_2d = self.project_point(local_axes[0][1][0], local_axes[0][1][1], local_axes[0][1][2])
                cv2.line(overlay_img, 
                        (int(center_2d[0]), int(center_2d[1])), 
                        (int(x_end_2d[0]), int(x_end_2d[1])), 
                        (0, 0, 255), 2)  # Rouge en BGR
                cv2.putText(overlay_img, "X", (int(x_end_2d[0]), int(x_end_2d[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Axe Y (vert)
                y_end_2d = self.project_point(local_axes[1][1][0], local_axes[1][1][1], local_axes[1][1][2])
                cv2.line(overlay_img, 
                        (int(center_2d[0]), int(center_2d[1])), 
                        (int(y_end_2d[0]), int(y_end_2d[1])), 
                        (0, 255, 0), 2)  # Vert en BGR
                cv2.putText(overlay_img, "Y", (int(y_end_2d[0]), int(y_end_2d[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Axe Z (bleu)
                z_end_2d = self.project_point(local_axes[2][1][0], local_axes[2][1][1], local_axes[2][1][2])
                cv2.line(overlay_img, 
                        (int(center_2d[0]), int(center_2d[1])), 
                        (int(z_end_2d[0]), int(z_end_2d[1])), 
                        (255, 0, 0), 2)  # Bleu en BGR
                cv2.putText(overlay_img, "Z", (int(z_end_2d[0]), int(z_end_2d[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Afficher les valeurs de rotation
                info_text = f"X:{cube.rotation_x:.1f}° Y:{cube.rotation_y:.1f}° Z:{cube.rotation_z:.1f}°"
                cv2.putText(overlay_img, info_text, (int(front_points[0][0]), int(front_points[0][1]) - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            # Ajouter un texte avec l'ID du véhicule
            text_x, text_y = int(front_face[0][0]), int(front_face[0][1]) - 10
            cv2.putText(overlay_img, f"V{i+1}", (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Afficher l'image avec les cubes
        self.display_image(overlay_img, self.cube_overlay_display)
        
        # Stocker l'image pour les interactions souris
        self.overlay_image = overlay_img

    def on_cube_overlay_press(self, event):
        if not self.edit_mode:
            return
            
        # Convertir les coordonnées du clic en coordonnées d'image
        x, y = self.convert_display_to_image_coords(event.x, event.y, self.cube_overlay_display)
        self.drag_start_x, self.drag_start_y = x, y
        
        # Si aucun cube n'est sélectionné, vérifier si le clic est sur un cube
        if not self.selected_cube:
            for cube in self.cubes:
                vertices = cube.get_vertices()
                projected_vertices = self.project_vertices(vertices)
                front_face = projected_vertices[:4]
                
                # Vérifier si le point est à l'intérieur du polygone front_face
                # Construire un polygone valide pour OpenCV
                try:
                    front_points = np.array([(int(v[0]), int(v[1])) for v in front_face], dtype=np.int32)
                    # Vérifier si le polygone a au moins 3 points et est fermé
                    if len(front_points) >= 3:
                        # Test si le point est dans le polygone
                        if cv2.pointPolygonTest(front_points, (float(x), float(y)), False) >= 0:
                            self.selected_cube = cube
                            # Activer et mettre à jour tous les sliders avec les valeurs du cube
                            self.cube_depth_slider["state"] = "normal"
                            self.cube_depth_slider.set(cube.depth / 50)  # Mettre à jour le slider
                            
                            # Activer et mettre à jour les sliders de rotation
                            self.rotation_x_slider["state"] = "normal"
                            self.rotation_y_slider["state"] = "normal"
                            self.rotation_z_slider["state"] = "normal"
                            self.rotation_x_slider.set(cube.rotation_x)
                            self.rotation_y_slider.set(cube.rotation_y)
                            self.rotation_z_slider.set(cube.rotation_z)
                            
                            self.status_var.set("Cube sélectionné. Utilisez les sliders pour modifier la rotation.")
                            self.update_3d_view()  # Mettre à jour l'affichage
                            break
                except cv2.error:
                    # Si une erreur se produit (polygone invalide), essayer une approche alternative
                    # Par exemple, vérifier si le point est proche d'un des coins du cube
                    for vertex in front_face:
                        dist = np.sqrt((x - vertex[0])**2 + (y - vertex[1])**2)
                        if dist < 15:  # Distance de sélection en pixels
                            self.selected_cube = cube
                            # Activer et mettre à jour tous les sliders
                            self.cube_depth_slider["state"] = "normal"
                            self.cube_depth_slider.set(cube.depth / 50)
                            
                            # Activer et mettre à jour les sliders de rotation
                            self.rotation_x_slider["state"] = "normal"
                            self.rotation_y_slider["state"] = "normal"
                            self.rotation_z_slider["state"] = "normal"
                            self.rotation_x_slider.set(cube.rotation_x)
                            self.rotation_y_slider.set(cube.rotation_y)
                            self.rotation_z_slider.set(cube.rotation_z)
                            
                            self.status_var.set("Cube sélectionné via la proximité des sommets.")
                            self.update_3d_view()
                            break
            
            # Si on n'a toujours pas de cube sélectionné, on s'arrête ici
            if not self.selected_cube:
                return
        
        cube = self.selected_cube
        vertices = cube.get_vertices()
        projected_vertices = self.project_vertices(vertices)
        
        # Obtenir les coordonnées des faces
        front_face = projected_vertices[:4]
        
        # Vérifier si le clic est sur un sommet (les 8 sommets)
        for i, vertex in enumerate(projected_vertices):
            distance = np.sqrt((x - vertex[0])**2 + (y - vertex[1])**2)
            if distance < 12:  # Rayon de sélection
                self.drag_mode = 'vertex'
                self.drag_vertex_index = i
                self.status_var.set(f"Modification du sommet {i}...")
                return
        
        # Vérifier si le clic est dans le cube (sur une face)
        # On fait un test simplifié pour éviter les problèmes de pointPolygonTest
        is_in_cube = False
        try:
            # Essayer d'abord avec pointPolygonTest
            front_points = np.array([(float(v[0]), float(v[1])) for v in front_face], dtype=np.float32).reshape((-1, 1, 2))
            if cv2.pointPolygonTest(front_points, (float(x), float(y)), False) >= 0:
                is_in_cube = True
        except cv2.error:
            # En cas d'erreur, vérifier si le point est dans le rectangle englobant
            min_x = min(p[0] for p in front_face)
            max_x = max(p[0] for p in front_face)
            min_y = min(p[1] for p in front_face)
            max_y = max(p[1] for p in front_face)
            
            if (min_x <= x <= max_x) and (min_y <= y <= max_y):
                is_in_cube = True
        
        if is_in_cube:
            self.drag_mode = 'move'
            self.status_var.set("Déplacement du cube...")
            return
        
        # Si le clic n'est sur aucun élément interactif, désélectionner le cube
        self.selected_cube = None
        self.cube_depth_slider["state"] = "disabled"
        self.status_var.set("Aucun cube sélectionné.")
        self.update_3d_view()

    def on_cube_overlay_drag(self, event):
        if not self.drag_mode or not self.selected_cube:
            return
            
        # Convertir les coordonnées actuelles en coordonnées d'image
        current_x, current_y = self.convert_display_to_image_coords(event.x, event.y, self.cube_overlay_display)
        
        # Calculer le déplacement
        dx = current_x - self.drag_start_x
        dy = current_y - self.drag_start_y
        
        cube = self.selected_cube
        
        if self.drag_mode == 'move':
            # Déplacer tout le cube - plus robuste et permet des déplacements plus larges
            cube.x += dx
            cube.y += dy
            self.status_var.set(f"Déplacement: dx={dx:.1f}, dy={dy:.1f}")
            
        elif self.drag_mode == 'vertex':
            # Selon le sommet déplacé, ajuster les dimensions du cube
            i = self.drag_vertex_index
            
            if i < 4:  # Face avant
                if i == 0:  # Coin supérieur gauche
                    # Calculer les nouvelles dimensions tout en tenant compte de la rotation
                    rot_rad = np.radians(cube.rotation_z)
                    cos_theta = np.cos(rot_rad)
                    sin_theta = np.sin(rot_rad)
                    
                    # Calculer le déplacement dans le repère du cube (tourné)
                    dx_rot = dx * cos_theta + dy * sin_theta
                    dy_rot = -dx * sin_theta + dy * cos_theta
                    
                    # Mettre à jour les dimensions (en respectant une taille minimale)
                    if cube.width - dx_rot > 20:
                        cube.x += dx_rot * cos_theta
                        cube.y += dx_rot * sin_theta
                        cube.width -= dx_rot
                    
                    if cube.height - dy_rot > 20:
                        cube.x -= dy_rot * sin_theta
                        cube.y += dy_rot * cos_theta
                        cube.height -= dy_rot
                
                # Ajouter des cas similaires pour les autres sommets...
                # Cette partie peut être développée pour gérer tous les sommets
        
        # Mettre à jour l'affichage
        self.update_3d_view()
        
        # Mettre à jour le point de départ pour le prochain déplacement
        self.drag_start_x, self.drag_start_y = current_x, current_y

    def process_coordinate_click(self, x, y):
        # Nouvelle approche: un seul clic pour créer le cube initial
        # La position du clic définit le centre de la face avant
        
        # Vérifier si la perspective a été définie
        if not self.perspective_defined:
            self.status_var.set("Vous devez d'abord définir la perspective avant d'ajouter des cubes.")
            return
        
        # Définir une taille par défaut pour le cube
        default_width = 100
        default_height = 100
        
        # Calculer les coordonnées du coin supérieur gauche
        top_left_x = x - default_width / 2
        top_left_y = y - default_height / 2
        
        # Récupérer les données de perspective pour orienter le cube correctement
        perspective_data = self.calculate_perspective_matrix()
        
        # Créer un nouveau cube centré sur le point cliqué avec rotation initiale basée sur la perspective
        new_cube = Cube3D(
            x=top_left_x,
            y=top_left_y,
            z=0,
            width=default_width,
            height=default_height,
            depth=float(self.cube_depth_slider.get()) * 50,  # Utiliser la valeur du slider
            rotation=perspective_data['horizon_angle'] if perspective_data else 0  # Orientation basée sur l'horizon
        )
        
        # Appliquer la rotation verticale basée sur la perspective
        if perspective_data:
            new_cube.rotation_x = perspective_data['vertical_angle']
        
        # Ajouter le cube à la liste et le sélectionner
        self.cubes.append(new_cube)
        self.selected_cube = new_cube
        
        # Mettre à jour les informations du véhicule
        if self.vehicule_count > 0:
            vehicle_key = f"vehicule{self.vehicule_count}"
            if vehicle_key in self.json_image_info:
                self.json_image_info[vehicle_key]["cube"] = new_cube.to_dict()
                
                # Stocker les coordonnées du point de clic pour référence
                self.json_image_info[vehicle_key]["center_point"] = [x, y]
                
                # Utiliser la profondeur du slider
                depth_meters = float(self.cube_depth_slider.get())
                self.json_image_info[vehicle_key]["depth_meters"] = depth_meters
        
        self.status_var.set(f"Cube créé à ({x}, {y}). Utilisez les sliders pour ajuster la rotation.")
        
        # Activer tous les sliders
        self.cube_depth_slider["state"] = "normal"
        self.rotation_x_slider["state"] = "normal"
        self.rotation_y_slider["state"] = "normal"
        self.rotation_z_slider["state"] = "normal"
        
        # Réinitialiser les sliders de rotation à 0
        self.rotation_x_slider.set(0)
        self.rotation_y_slider.set(0)
        self.rotation_z_slider.set(0)
        
        # Mettre à jour l'affichage
        self.update_3d_view()
        self.update_cube_overlay()

    def update_cube_depth(self, event=None):
        # Mettre à jour uniquement la profondeur du cube tout en préservant les points cliqués
        if self.selected_cube is not None:
            # Convertir la valeur du slider (en mètres) en unités de l'image
            depth_meters = float(self.cube_depth_slider.get())
            conversion_factor = 50  # 1 mètre = 50 unités
            new_depth = depth_meters * conversion_factor
            
            # Récupérer l'angle de rotation actuel
            current_rotation = self.selected_cube.rotation_z  # Utiliser rotation_z au lieu de rotation
            
            # Mettre à jour la profondeur sans changer les autres dimensions ni positions
            self.selected_cube.depth = new_depth
            
            # Mettre à jour les informations du véhicule
            if self.vehicule_count > 0:
                vehicle_key = f"vehicule{self.vehicule_count}"
                if vehicle_key in self.json_image_info:
                    # Mettre à jour le cube
                    self.json_image_info[vehicle_key]["cube"] = self.selected_cube.to_dict()
                    self.json_image_info[vehicle_key]["depth_meters"] = depth_meters
                    
                    # Si nous avons les points originaux, recalculer le point de la face avant
                    if "points" in self.json_image_info[vehicle_key]:
                        points = self.json_image_info[vehicle_key]["points"]
                        
                        # Récupérer le coin inférieur gauche arrière
                        back_top_left = points["back_top_left"]
                        back_bottom_right = points["back_bottom_right"]
                        
                        # Déterminer le coin inférieur gauche de la face arrière
                        width = abs(back_bottom_right[0] - back_top_left[0])
                        height = abs(back_bottom_right[1] - back_top_left[1])
                        
                        # Déterminer le coin inférieur gauche de la face arrière
                        back_bottom_left_x = min(back_top_left[0], back_bottom_right[0])
                        back_bottom_left_y = max(back_top_left[1], back_bottom_right[1])
                        
                        # Calculer le nouveau point de la face avant basé sur la nouvelle profondeur
                        rotation_rad = np.radians(current_rotation)
                        new_translation_x = new_depth * np.cos(rotation_rad)
                        new_translation_y = new_depth * np.sin(rotation_rad)
                        
                        new_front_bottom_left = [
                            back_bottom_left_x + new_translation_x,
                            back_bottom_left_y + new_translation_y
                        ]
                        
                        # Mettre à jour le point de la face avant
                        points["front_bottom_left"] = new_front_bottom_left
            
            self.status_var.set(f"Profondeur du cube mise à jour: {depth_meters:.1f} m")
            
            # Mettre à jour l'affichage
            self.update_3d_view()
            self.update_cube_overlay()

    def project_vertices(self, vertices):
        """
        Projette les sommets 3D en 2D avec perspective.
        """
        projected = []
        for vertex in vertices:
            x, y, z = vertex
            # Facteur de perspective (plus le z est grand, plus le point semble petit/lointain)
            # Valeur focal_length contrôle la force de l'effet de perspective
            focal_length = 500
            # Plus z est grand, plus le point est loin dans la profondeur
            scale = focal_length / (focal_length + z)
            
            projected_x = x * scale
            projected_y = y * scale
            projected.append((projected_x, projected_y))
        return projected

    def project_point(self, x, y, z):
        """
        Projette un point 3D en 2D avec perspective
        """
        # Facteur de perspective (plus le z est grand, plus le point semble petit/lointain)
        focal_length = 500
        scale = focal_length / (focal_length + z)
        
        projected_x = x * scale
        projected_y = y * scale
        return (projected_x, projected_y)

    # Ajouter ces nouvelles méthodes pour la manipulation du cube
    def setup_cube_interaction(self):
        """
        Configure les événements souris pour l'interaction avec les cubes dans la vue overlay
        """
        # Configurer les événements souris pour l'interaction avec les cubes
        self.cube_overlay_display.bind("<ButtonPress-1>", self.on_cube_overlay_press)
        self.cube_overlay_display.bind("<B1-Motion>", self.on_cube_overlay_drag)
        self.cube_overlay_display.bind("<ButtonRelease-1>", self.on_cube_overlay_release)
        
        # Variables pour suivre l'interaction
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_mode = None  # 'move', 'rotate', 'vertex' ou None
        self.drag_vertex_index = None  # Index du sommet en cours de déplacement
        self.rotation_center = None  # Centre de rotation (x, y)

    def on_cube_overlay_release(self, event):
        """
        Gère le relâchement du bouton de la souris dans la vue overlay
        """
        if self.drag_mode and self.selected_cube:
            # Mettre à jour les informations du cube dans json_image_info
            if self.vehicule_count > 0:
                vehicle_key = f"vehicule{self.vehicule_count}"
                if vehicle_key in self.json_image_info:
                    self.json_image_info[vehicle_key]["cube"] = self.selected_cube.to_dict()
            
            # Réinitialiser les variables de drag
            self.drag_mode = None
            self.drag_vertex_index = None
            self.rotation_center = None
            self.status_var.set("Cube modifié.")

    def update_rotation(self, event=None):
        """
        Met à jour les rotations du cube sélectionné à partir des valeurs des sliders
        """
        if self.selected_cube is not None:
            try:
                # Récupérer les valeurs des sliders
                x_rotation = float(self.rotation_x_slider.get())
                y_rotation = float(self.rotation_y_slider.get())
                z_rotation = float(self.rotation_z_slider.get())
                
                # Mettre à jour les rotations du cube
                self.selected_cube.rotation_x = x_rotation
                self.selected_cube.rotation_y = y_rotation
                self.selected_cube.rotation_z = z_rotation
                
                # Debug information
                print(f"Rotation updated: X={x_rotation}°, Y={y_rotation}°, Z={z_rotation}°")
                
                # Mettre à jour l'affichage
                self.status_var.set(f"Rotation mise à jour: X={x_rotation}° (autour de l'axe local X), " +
                                   f"Y={y_rotation}° (autour de l'axe local Y), " +
                                   f"Z={z_rotation}° (autour de l'axe local Z)")
                self.update_3d_view()
                
                # Mettre à jour les informations dans json_image_info
                if self.vehicule_count > 0:
                    vehicle_key = f"vehicule{self.vehicule_count}"
                    if vehicle_key in self.json_image_info and self.selected_cube:
                        self.json_image_info[vehicle_key]["cube"] = self.selected_cube.to_dict()
            except Exception as e:
                # Afficher l'erreur pour le débogage
                print(f"Erreur lors de la mise à jour de la rotation: {e}")
                self.status_var.set(f"Erreur: {e}")

    def toggle_perspective_mode(self):
        """
        Active ou désactive le mode de définition de la perspective
        """
        self.perspective_mode = not self.perspective_mode
        
        if self.perspective_mode:
            # Désactiver les autres modes
            self.edit_mode = False
            self.edit_cube_btn.config(text="Éditer le cube")
            
            # Activer les boutons de perspective
            self.define_horizon_btn["state"] = "normal"
            self.define_vanishing_btn["state"] = "normal"
            self.define_vertical_btn["state"] = "normal"
            self.reset_perspective_btn["state"] = "normal"
            self.define_perspective_btn.config(text="Quitter mode perspective")
            
            # Définir l'horizon par défaut comme première ligne à tracer
            self.set_line_type('horizon')
            
            self.status_var.set("Mode définition perspective activé. Définissez d'abord la ligne d'horizon.")
        else:
            # Désactiver les boutons de perspective
            self.define_horizon_btn["state"] = "disabled"
            self.define_vanishing_btn["state"] = "disabled"
            self.define_vertical_btn["state"] = "disabled"
            self.reset_perspective_btn["state"] = "disabled"
            self.define_perspective_btn.config(text="Définir perspective")
            
            # Réinitialiser le type de ligne en cours
            self.current_line_type = None
            
            self.status_var.set("Mode définition perspective désactivé.")
        
        # Mettre à jour l'affichage
        self.update_cube_overlay()
    
    def set_line_type(self, line_type):
        """
        Définit le type de ligne à tracer
        """
        self.current_line_type = line_type
        
        if line_type == 'horizon':
            self.horizon_points = []
            self.status_var.set("Cliquez sur deux points pour définir l'horizon (ligne horizontale)")
        elif line_type == 'vanishing':
            self.vanishing_points = []
            self.status_var.set("Cliquez sur deux points pour définir la ligne de fuite (perspective)")
        elif line_type == 'vertical':
            self.vertical_points = []
            self.status_var.set("Cliquez sur deux points pour définir une ligne verticale (normale au sol)")
    
    def reset_perspective(self):
        """
        Réinitialise toutes les lignes de perspective
        """
        self.horizon_points = []
        self.vanishing_points = []
        self.vertical_points = []
        self.perspective_defined = False
        self.status_var.set("Perspective réinitialisée. Définissez à nouveau les lignes.")
        self.update_cube_overlay()
    
    def check_perspective_completion(self):
        """
        Vérifie si toutes les lignes de perspective ont été définies
        """
        if len(self.horizon_points) == 2 and len(self.vanishing_points) == 2 and len(self.vertical_points) == 2:
            self.perspective_defined = True
            self.status_var.set("Perspective définie! Vous pouvez maintenant placer des cubes.")
            return True
        return False
    
    def calculate_perspective_matrix(self):
        """
        Calcule la matrice de perspective basée sur les lignes définies
        """
        if not self.perspective_defined:
            return None
        
        # Calculer les vecteurs directeurs des lignes
        horizon_vector = [
            self.horizon_points[1][0] - self.horizon_points[0][0],
            self.horizon_points[1][1] - self.horizon_points[0][1]
        ]
        horizon_length = np.sqrt(horizon_vector[0]**2 + horizon_vector[1]**2)
        horizon_vector = [horizon_vector[0]/horizon_length, horizon_vector[1]/horizon_length]
        
        vanishing_vector = [
            self.vanishing_points[1][0] - self.vanishing_points[0][0],
            self.vanishing_points[1][1] - self.vanishing_points[0][1]
        ]
        vanishing_length = np.sqrt(vanishing_vector[0]**2 + vanishing_vector[1]**2)
        vanishing_vector = [vanishing_vector[0]/vanishing_length, vanishing_vector[1]/vanishing_length]
        
        vertical_vector = [
            self.vertical_points[1][0] - self.vertical_points[0][0],
            self.vertical_points[1][1] - self.vertical_points[0][1]
        ]
        vertical_length = np.sqrt(vertical_vector[0]**2 + vertical_vector[1]**2)
        vertical_vector = [vertical_vector[0]/vertical_length, vertical_vector[1]/vertical_length]
        
        # Calculer l'angle entre l'horizon et la ligne de fuite
        dot_product = horizon_vector[0]*vanishing_vector[0] + horizon_vector[1]*vanishing_vector[1]
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Déterminer si l'angle est dans le sens horaire ou anti-horaire
        cross_product = horizon_vector[0]*vanishing_vector[1] - horizon_vector[1]*vanishing_vector[0]
        if cross_product < 0:
            angle = -angle
        
        # Calculer l'angle entre la verticale et l'axe Y standard
        vertical_standard = [0, -1]  # Verticale standard (vers le haut)
        dot_product_vertical = vertical_vector[0]*vertical_standard[0] + vertical_vector[1]*vertical_standard[1]
        angle_vertical = np.arccos(np.clip(dot_product_vertical, -1.0, 1.0))
        
        # Déterminer si l'angle est dans le sens horaire ou anti-horaire
        cross_product_vertical = vertical_vector[0]*vertical_standard[1] - vertical_vector[1]*vertical_standard[0]
        if cross_product_vertical < 0:
            angle_vertical = -angle_vertical
        
        return {
            'horizon_angle': np.degrees(angle),
            'vertical_angle': np.degrees(angle_vertical)
        }

if __name__ == "__main__":
    root = tk.Tk()
    app = DepthMapApp(root)
    root.mainloop()