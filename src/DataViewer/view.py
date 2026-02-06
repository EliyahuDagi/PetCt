import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
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
        
        gdrive_btn = ttk.Button(toolbar_frame, text="Load from GDrive", command=self._on_gdrive_click)
        gdrive_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.patient_lbl = ttk.Label(toolbar_frame, text="No Patient Loaded")
        self.patient_lbl.pack(side=tk.LEFT, padx=20)

    def _create_plot_area(self):
        # Using Matplotlib Figure
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.fig.patch.set_facecolor('black') # Dark theme background
        
        # 3 Subplots: CT, PET, Fusion
        # Turn off axis for cleaner "RadiAnt-like" look
        self.ax_ct = self.fig.add_subplot(131)
        self.ax_ct.set_axis_off()
        self.ax_ct.set_title("CT", color='white')
        
        self.ax_pet = self.fig.add_subplot(132)
        self.ax_pet.set_axis_off()
        self.ax_pet.set_title("PET", color='white')
        
        self.ax_fusion = self.fig.add_subplot(133)
        self.ax_fusion.set_axis_off()
        self.ax_fusion.set_title("Fusion", color='white')
        
        # Placeholder Images
        blank_data = np.zeros((512, 512))
        self.img_ct = self.ax_ct.imshow(blank_data, cmap='gray', vmin=-1000, vmax=1000)
        self.img_pet = self.ax_pet.imshow(blank_data, cmap='hot')
        self.img_fusion = self.ax_fusion.imshow(blank_data, cmap='gray') # Base layer
        # For fusion, we might need complex alpha blending, but for now let's keep it simple
        
        # HUD Overlays (Text Artists)
        self.overlays = {}
        self._setup_overlays(self.ax_ct, "ct")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Event Bindings
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        
        self.dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

    def _setup_overlays(self, ax, prefix):
        # Top Left: Patient Info
        self.overlays[f'{prefix}_tl'] = ax.text(0.02, 0.98, "Patient Name\nID: 12345", 
                                                transform=ax.transAxes, color='yellow', verticalalignment='top', fontsize=9)
        # Top Right: Hospital / Study Info
        self.overlays[f'{prefix}_tr'] = ax.text(0.98, 0.98, "Hospital Name\nStudy Desc", 
                                                transform=ax.transAxes, color='yellow', verticalalignment='top', horizontalalignment='right', fontsize=9)
        # Bottom Left: Tech Info (WL/WW, Slice Location)
        self.overlays[f'{prefix}_bl'] = ax.text(0.02, 0.02, "WL: 40 WW: 400\nSlice: 0", 
                                                transform=ax.transAxes, color='yellow', verticalalignment='bottom', fontsize=9)
        # Bottom Right: Orientation / Geometry
        self.overlays[f'{prefix}_br'] = ax.text(0.98, 0.02, "Thickness: 2.5mm", 
                                                transform=ax.transAxes, color='yellow', verticalalignment='bottom', horizontalalignment='right', fontsize=9)

    def _on_scroll(self, event):
        if event.inaxes and self.presenter:
            if event.button == 'up':
                self.presenter.change_slice(1)
            elif event.button == 'down':
                self.presenter.change_slice(-1)

    def _on_mouse_press(self, event):
        if event.button == 3: # Right Click for Window/Level
            self.dragging = True
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y

    def _on_mouse_move(self, event):
        if self.dragging and self.presenter:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            self.presenter.change_window_level(dx, dy)
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y

    def _on_mouse_release(self, event):
        self.dragging = False

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

    def _on_gdrive_click(self):
        folder_id = simpledialog.askstring("Google Drive", "Enter Folder ID:")
        if folder_id and self.presenter:
            self.presenter.load_from_gdrive(folder_id)

    def _on_prev_patient(self):
        if self.presenter:
            self.presenter.prev_patient()

    def _on_next_patient(self):
        if self.presenter:
            self.presenter.next_patient()

    def _on_slice_change(self, value):
        if self.presenter:
            self.presenter.set_slice(int(value))

    def update_images(self, ct_img, pet_img, slice_idx, wl=50, ww=400):
        # Efficiently update image data without clearing axes
        if ct_img is not None:
            self.img_ct.set_data(ct_img)
            # Apply Window/Level
            vmin = wl - (ww / 2)
            vmax = wl + (ww / 2)
            self.img_ct.set_clim(vmin, vmax)
            self.ax_ct.set_title(f"CT (Slice {slice_idx})", color='white')
            
            # Simple Fusion Logic (Alpha blending)
            # Normalize CT to 0-1 range for background
            ct_norm = np.clip((ct_img - vmin) / ww, 0, 1)
            # Just show CT for now in fusion or blending
            # For simplicity in this step, let's keep Fusion as CT only or handle overlay later properly
            # Or assume pet_img overlay on CT
            self.img_fusion.set_data(ct_img) # Placeholder for fusion logic
            self.img_fusion.set_clim(vmin, vmax)

        if pet_img is not None:
            self.img_pet.set_data(pet_img)
            self.img_pet.set_clim(0, np.max(pet_img) if np.max(pet_img) > 0 else 1)
        
        self.canvas.draw_idle()

    def update_overlays(self, metadata_dict):
        # Update text artists based on dictionary
        # metadata_dict: { 'name': 'Patient X', 'id': '123', 'wl': 50, 'ww': 400, 'slice': 10, 'pos': -123.5 }
        if 'name' in metadata_dict and 'id' in metadata_dict:
             txt = f"{metadata_dict['name']}\nID: {metadata_dict['id']}"
             self.overlays['ct_tl'].set_text(txt)
             
        if 'wl' in metadata_dict and 'ww' in metadata_dict:
            txt = f"WL: {int(metadata_dict['wl'])} WW: {int(metadata_dict['ww'])}\nSlice: {metadata_dict.get('slice', 0)}"
            if 'pos' in metadata_dict:
                txt += f"\nZ: {metadata_dict['pos']:.1f}mm"
            self.overlays['ct_bl'].set_text(txt)
            
        if 'thickness' in metadata_dict:
             self.overlays['ct_br'].set_text(f"Thickness: {metadata_dict['thickness']}mm")

        # Force redraw is called in update_images mostly, but if only text changes:
        # self.canvas.draw_idle()

    def set_max_slice(self, max_slice):
        self.slice_scale.config(to=max_slice - 1)
        
    def set_current_patient_info(self, info_text):
        self.patient_lbl.config(text=info_text)
