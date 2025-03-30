import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk

class TrajectoryAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotateur de Trajectoire")
        self.root.geometry("1200x800")
        
        # Variables
        self.image_folder = None
        self.image_files = []
        self.current_index = 0
        self.trajectory = []
        self.dataset = {}
        self.img = None
        self.photo = None
        self.tags = []  # Liste des tags disponibles
        self.selected_tags = []  # Tags sélectionnés pour l'image actuelle
        
        # Charger les tags préexistants
        self.load_tags()
        
        # Charger un dataset existant si disponible
        self.load_existing_dataset()
        
        # Sélection du dossier
        self.select_folder()
        
        if not self.image_folder:
            messagebox.showerror("Erreur", "Aucun dossier sélectionné. Le programme va se fermer.")
            self.root.destroy()
            return
        
        # Interface
        self.create_ui()
        
        # Charger la première image
        self.root.update()  # Mettre à jour la fenêtre pour obtenir les dimensions correctes
        self.load_image()
        
        # Activer le focus pour permettre la saisie de texte
        self.root.focus_force()
    
    def load_tags(self):
        """Charger les tags existants depuis un fichier"""
        try:
            if os.path.exists("tags.json"):
                with open("tags.json", "r") as f:
                    self.tags = json.load(f)
                print(f"Tags chargés: {len(self.tags)}")
            else:
                # Tags par défaut
                self.tags = ["ligne_droite", "intersection", "obstacle", "zone_dangereuse","stop","feu","passage_pieton","priorite","depassement","voiture_proche","pieton_proche","velo_proche"]
                self.save_tags()
        except Exception as e:
            messagebox.showwarning("Attention", f"Impossible de charger les tags: {str(e)}")
            self.tags = ["ligne_droite", "intersection", "obstacle", "zone_dangereuse","stop","feu","passage_pieton","priorite","depassement","voiture_proche","pieton_proche","velo_proche"]
    
    def save_tags(self):
        """Sauvegarder les tags dans un fichier"""
        try:
            with open("tags.json", "w") as f:
                json.dump(self.tags, f, indent=4)
        except Exception as e:
            messagebox.showwarning("Attention", f"Impossible de sauvegarder les tags: {str(e)}")
    
    def add_new_tag(self):
        """Ajouter un nouveau tag à la liste"""
        new_tag = simpledialog.askstring("Nouveau tag", "Entrez le nom du nouveau tag:", parent=self.root)
        if new_tag and new_tag.strip():
            new_tag = new_tag.strip().lower().replace(" ", "_")
            if new_tag not in self.tags:
                self.tags.append(new_tag)
                self.save_tags()
                self.update_tag_display()
                messagebox.showinfo("Succès", f"Tag '{new_tag}' ajouté avec succès.")
            else:
                messagebox.showwarning("Attention", "Ce tag existe déjà.")
    
    def update_tag_display(self):
        """Mettre à jour l'affichage des tags"""
        # Effacer toutes les cases à cocher
        for widget in self.tag_frame.winfo_children():
            widget.destroy()
        
        # Recréer les cases à cocher pour tous les tags
        self.tag_vars = {}
        for i, tag in enumerate(self.tags):
            var = tk.BooleanVar(value=tag in self.selected_tags)
            cb = tk.Checkbutton(self.tag_frame, text=tag, variable=var)
            cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=5, pady=2)
            self.tag_vars[tag] = var
    
    def load_existing_dataset(self):
        try:
            if os.path.exists("dataset.json"):
                with open("dataset.json", "r") as f:
                    self.dataset = json.load(f)
                messagebox.showinfo("Information", f"{len(self.dataset)} annotations chargées depuis dataset.json")
        except Exception as e:
            messagebox.showwarning("Attention", f"Impossible de charger le dataset existant: {str(e)}")
    
    def select_folder(self):
        folder = filedialog.askdirectory(title="Sélectionner le dossier d'images")
        if folder:
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
            images = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
            
            if not images:
                messagebox.showwarning("Attention", "Aucune image trouvée dans ce dossier.")
                self.select_folder()
                return
            
            self.image_folder = folder
            self.image_files = sorted(images)
    
    def create_ui(self):
        # Frame principale
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas pour l'image (avec scrollbars)
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="lightgray", width=800, height=600, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Notebook pour les données et tags
        notebook = ttk.Notebook(self.main_frame)
        notebook.pack(fill=tk.X, pady=5)
        
        # Tab pour les informations et données
        info_frame = tk.Frame(notebook)
        notebook.add(info_frame, text="Données")
        
        # Créer les champs pour les données
        self.data_entries = {}
        data_frame = tk.Frame(info_frame)
        data_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)
        
        labels = ["Vitesse (km/h)", "Angle (°)", "Accélération (m/s²)"]
        for i, label in enumerate(labels):
            lbl = tk.Label(data_frame, text=label)
            lbl.grid(row=0, column=i, padx=5)
            
            # Utiliser StringVar pour plus de contrôle sur le contenu des Entry
            var = tk.StringVar()
            entry = tk.Entry(data_frame, width=15, textvariable=var)
            entry.grid(row=1, column=i, padx=5, pady=5)
            
            # S'assurer que l'Entry est activé et accessible
            entry.config(state="normal")
            
            # Ajouter un gestionnaire d'événements de focus
            entry.bind("<FocusIn>", lambda e, w=entry: self.on_entry_focus(w))
            
            self.data_entries[label] = (entry, var)
        
        # Tab pour les tags
        tag_tab = tk.Frame(notebook)
        notebook.add(tag_tab, text="Tags")
        
        # Cadre pour les tags avec scrollbar
        tag_outer_frame = tk.Frame(tag_tab)
        tag_outer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bouton pour ajouter un nouveau tag
        tk.Button(tag_outer_frame, text="Ajouter un nouveau tag", 
                 command=self.add_new_tag).pack(fill=tk.X, pady=5)
                 
        # Frame pour les tags eux-mêmes
        self.tag_frame = tk.Frame(tag_outer_frame)
        self.tag_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialiser l'affichage des tags
        self.update_tag_display()
        
        # Boutons de navigation et actions
        button_frame = tk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(button_frame, text="Effacer la trajectoire", command=self.clear_trajectory,
                 width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Image précédente", command=self.prev_image,
                 width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Image suivante", command=self.save_and_next,
                 width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Sauvegarder", command=self.save_dataset,
                 width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Quitter", command=self.save_and_exit,
                 width=15).pack(side=tk.LEFT, padx=5)
        
        # Barre d'état
        self.status_var = tk.StringVar()
        status_bar = tk.Label(self.main_frame, textvariable=self.status_var, 
                            bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def on_entry_focus(self, widget):
        """Gère le focus sur un widget d'entrée"""
        # S'assurer que le widget a le focus et est éditable
        widget.config(state="normal")
        
        # Mettre à jour le statut
        self.status_var.set(f"Entrez une valeur pour {widget.grid_info()['row']}:{widget.grid_info()['column']}")
    
    def load_image(self):
        if not self.image_files or self.current_index >= len(self.image_files):
            self.status_var.set("Fin des images atteinte")
            return False
        
        # Réinitialiser
        self.canvas.delete("all")
        self.trajectory = []
        self.selected_tags = []
        for entry, var in self.data_entries.values():
            var.set("")  # Utiliser la StringVar pour effacer le contenu
        
        try:
            # Charger l'image
            image_name = self.image_files[self.current_index]
            image_path = os.path.join(self.image_folder, image_name)
            
            if not os.path.exists(image_path):
                self.status_var.set(f"Fichier introuvable: {image_path}")
                return False
            
            # Ouvrir avec PIL
            self.img = Image.open(image_path)
            
            # Obtenir les dimensions du canvas
            canvas_width = self.canvas.winfo_width()
            if canvas_width <= 1:  # Si le canvas n'est pas encore dimensionné correctement
                canvas_width = 800
            canvas_height = self.canvas.winfo_height()
            if canvas_height <= 1:
                canvas_height = 600
            
            # Redimensionner l'image en préservant le ratio
            img_width, img_height = self.img.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            
            new_width = int(img_width * ratio * 0.9)  # 90% de la taille disponible
            new_height = int(img_height * ratio * 0.9)
            
            if new_width > 0 and new_height > 0:
                self.img = self.img.resize((new_width, new_height), Image.LANCZOS)
                self.photo = ImageTk.PhotoImage(self.img)
                
                # Afficher l'image au centre
                self.canvas.create_image(canvas_width/2, canvas_height/2, 
                                        image=self.photo, anchor=tk.CENTER)
                
                # Mettre à jour le titre et le statut
                self.root.title(f"Annotation - {image_name}")
                self.status_var.set(f"Image {self.current_index+1}/{len(self.image_files)}: {image_name}")
                
                # Charger les données existantes
                if image_name in self.dataset:
                    data = self.dataset[image_name]
                    self.trajectory = data["trajectory"]
                    self.redraw_trajectory()
                    
                    for key, value in data["data"].items():
                        if key in self.data_entries:
                            entry, var = self.data_entries[key]
                            var.set(str(value))  # Utiliser la StringVar pour définir le contenu
                    
                    # Charger les tags sélectionnés
                    if "tags" in data:
                        self.selected_tags = data["tags"]
                
                # Mettre à jour l'affichage des tags
                self.update_tag_display()
                
                return True
            else:
                self.status_var.set(f"Image trop grande ou canvas trop petit")
                return False
                
        except Exception as e:
            self.status_var.set(f"Erreur: {str(e)}")
            return False
    
    def on_canvas_click(self, event):
        if not self.photo:
            return
            
        x, y = event.x, event.y
        self.trajectory.append((x, y))
        
        # Dessiner un point
        self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="red", outline="black")
        
        # Dessiner une ligne
        if len(self.trajectory) > 1:
            prev_x, prev_y = self.trajectory[-2]
            self.canvas.create_line(prev_x, prev_y, x, y, fill="blue", width=2)
    
    def redraw_trajectory(self):
        if not self.trajectory:
            return
            
        for i, (x, y) in enumerate(self.trajectory):
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="red", outline="black")
            
            if i > 0:
                prev_x, prev_y = self.trajectory[i-1]
                self.canvas.create_line(prev_x, prev_y, x, y, fill="blue", width=2)
    
    def clear_trajectory(self):
        self.trajectory = []
        if self.photo:
            # Redessiner juste l'image
            self.canvas.delete("all")
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.canvas.create_image(canvas_width/2, canvas_height/2, 
                                    image=self.photo, anchor=tk.CENTER)
    
    def save_current_annotation(self):
        if not self.image_files or self.current_index >= len(self.image_files):
            return False
        
        image_name = self.image_files[self.current_index]
        
        # Récupérer les valeurs des champs
        data = {}
        for key, (entry, var) in self.data_entries.items():
            value = var.get().strip()  # Obtenir la valeur de la StringVar
            # Convertir en nombre si possible
            try:
                if "." in value:
                    data[key] = float(value)
                else:
                    data[key] = int(value)
            except ValueError:
                data[key] = value
        
        # Récupérer les tags sélectionnés
        selected_tags = []
        for tag, var in self.tag_vars.items():
            if var.get():
                selected_tags.append(tag)
        
        # Sauvegarder
        self.dataset[image_name] = {
            "trajectory": self.trajectory,
            "data": data,
            "tags": selected_tags,
            "tag_ids": [self.tags.index(tag) for tag in selected_tags if tag in self.tags]  # Ajouter les IDs des tags
        }
        
        # Mettre à jour la liste des tags sélectionnés
        self.selected_tags = selected_tags
        
        return True
    
    def save_dataset(self):
        try:
            with open("dataset.json", "w") as f:
                json.dump(self.dataset, f, indent=4)
            messagebox.showinfo("Information", f"Dataset sauvegardé avec {len(self.dataset)} annotations")
            return True
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")
            return False
    
    def save_and_next(self):
        self.save_current_annotation()
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image()
        else:
            self.status_var.set("Fin des images atteinte")
            self.save_dataset()
    
    def prev_image(self):
        self.save_current_annotation()
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()
    
    def save_and_exit(self):
        self.save_current_annotation()
        if self.save_dataset():
            self.root.destroy()

# Programme principal
if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryAnnotator(root)
    root.mainloop()