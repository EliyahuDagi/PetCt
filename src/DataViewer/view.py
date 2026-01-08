import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

class MainView(tk.Tk):
    def __init__(self, presenter=None):
        super().__init__()
        self.title("Pet-CT Data Viewer")
        self.geometry("1200x800")
        
        self.presenter = presenter
        
        self._create_toolbar()
        self._create_plot_area()
        self._create_controls()

    def set_presenter(self, presenter):
        self.presenter = presenter

    def _create_toolbar(self):
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        load_btn = ttk.Button(toolbar_frame, text="Load Dataset Folder", command=self._on_load_click)
        load_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.patient_lbl = ttk.Label(toolbar_frame, text="No Patient Loaded")
        self.patient_lbl.pack(side=tk.LEFT, padx=20)

    def _create_plot_area(self):
        # Using Matplotlib Figure
        self.fig = Figure(figsize=(12, 6), dpi=100)
        
        # 3 Subplots: CT, PET, Fusion
        self.ax_ct = self.fig.add_subplot(131)
        self.ax_ct.set_title("CT")
        
        self.ax_pet = self.fig.add_subplot(132)
        self.ax_pet.set_title("PET")
        
        self.ax_fusion = self.fig.add_subplot(133)
        self.ax_fusion.set_title("Fusion")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Optional: Add Matplotlib Toolbar
        # toolbar = NavigationToolbar2Tk(self.canvas, self)
        # toolbar.update()
        # self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _create_controls(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Patient Navigation
        prev_pat_btn = ttk.Button(control_frame, text="<< Prev Patient", command=self._on_prev_patient)
        prev_pat_btn.pack(side=tk.LEFT, padx=20)
        
        next_pat_btn = ttk.Button(control_frame, text="Next Patient >>", command=self._on_next_patient)
        next_pat_btn.pack(side=tk.LEFT, padx=5)

        # Slice Navigation
        self.slice_scale = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Slice", command=self._on_slice_change)
        self.slice_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        
    def _on_load_click(self):
        path = filedialog.askdirectory()
        if path and self.presenter:
            self.presenter.load_dataset(path)

    def _on_prev_patient(self):
        if self.presenter:
            self.presenter.prev_patient()

    def _on_next_patient(self):
        if self.presenter:
            self.presenter.next_patient()

    def _on_slice_change(self, value):
        if self.presenter:
            self.presenter.set_slice(int(value))

    def update_images(self, ct_img, pet_img, slice_idx):
        self.ax_ct.clear()
        self.ax_pet.clear()
        self.ax_fusion.clear()
        
        self.ax_ct.set_title(f"CT (Slice {slice_idx})")
        self.ax_pet.set_title("PET")
        self.ax_fusion.set_title("Fusion")
        self.ax_ct.axis('off')
        self.ax_pet.axis('off')
        self.ax_fusion.axis('off')
        
        if ct_img is not None:
             self.ax_ct.imshow(ct_img, cmap='gray')
        
        if pet_img is not None:
             self.ax_pet.imshow(pet_img, cmap='hot')
             
        # Fusion
        if ct_img is not None and pet_img is not None:
            # Simple scaling for display
            # Resize PET to match CT if shapes differ (simple zoom)
            import scipy.ndimage
            
            ct_shape = ct_img.shape
            pet_shape = pet_img.shape
            
            pet_resized = pet_img
            if ct_shape != pet_shape:
                zoom_factor = (ct_shape[0]/pet_shape[0], ct_shape[1]/pet_shape[1])
                pet_resized = scipy.ndimage.zoom(pet_img, zoom_factor, order=1)

            self.ax_fusion.imshow(ct_img, cmap='gray')
            self.ax_fusion.imshow(pet_resized, cmap='hot', alpha=0.5)
        
        self.canvas.draw()

    def set_max_slice(self, max_slice):
        self.slice_scale.config(to=max_slice - 1)
        
    def set_current_patient_info(self, info_text):
        self.patient_lbl.config(text=info_text)
