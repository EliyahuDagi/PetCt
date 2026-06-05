import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import colors as mcolors
import numpy as np

class MainView(tk.Tk):
    def __init__(self, presenter=None):
        super().__init__()
        self.title("Pet-CT Data Viewer")
        self.geometry("1200x800")
        
        self.presenter = presenter
        self.segmentation_value_map = {"All Classes": 'all_classes'}
        self.segmentation_color_lut = {}
        self.segmentation_label_map = {}
        self.current_segmentation_label = 'all_classes'
        self.segmentation_sources = ["TotalSegmentor"]
        self.current_segmentation_source = "TotalSegmentor"
        
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
        
        # MPR Buttons
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Button(toolbar_frame, text="Axial", command=lambda: self.presenter.set_orientation('AXIAL')).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Coronal", command=lambda: self.presenter.set_orientation('CORONAL')).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Sagittal", command=lambda: self.presenter.set_orientation('SAGITTAL')).pack(side=tk.LEFT, padx=2)

        self.patient_lbl = ttk.Label(toolbar_frame, text="No Patient Loaded")
        self.patient_lbl.pack(side=tk.LEFT, padx=20)
        
        # ZOI / Segmentation Button
        zoi_btn = ttk.Button(toolbar_frame, text="ZOI", command=lambda: self.presenter.toggle_segmentation_zoi() if self.presenter else None)
        zoi_btn.pack(side=tk.LEFT, padx=5)

    def _create_plot_area(self):
        # Using Matplotlib Figure
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.fig.patch.set_facecolor('black') # Dark theme background
        
        self.current_seg_img = None  # Init
        
        # 3 Subplots: CT, PET AC, PET NAC
        # Turn off axis for cleaner "RadiAnt-like" look
        self.ax_ct = self.fig.add_subplot(131)
        self.ax_ct.set_axis_off()
        self.ax_ct.set_title("CT", color='white')
        
        self.ax_pet = self.fig.add_subplot(132)
        self.ax_pet.set_axis_off()
        self.ax_pet.set_title("PET AC", color='white')
        
        self.ax_fusion = self.fig.add_subplot(133)
        self.ax_fusion.set_axis_off()
        self.ax_fusion.set_title("PET NAC", color='white')
        
        # Placeholder Images
        blank_data = np.zeros((512, 512))
        self.img_ct = self.ax_ct.imshow(blank_data, cmap='gray', vmin=-1000, vmax=1000)
        
        # Segmentation Overlay on CT using RGBA image
        self.img_seg_overlay = self.ax_ct.imshow(np.zeros((512, 512, 4), dtype=float), alpha=0.0)

        self.img_pet = self.ax_pet.imshow(blank_data, cmap='hot')
        self.img_pet_nac = self.ax_fusion.imshow(blank_data, cmap='hot')
        
        # ZOI Rectangle patches
        from matplotlib.patches import Rectangle
        self.rect_ct = Rectangle((0,0), 1, 1, linewidth=2, edgecolor='yellow', facecolor='none', visible=False)
        self.ax_ct.add_patch(self.rect_ct)
        
        self.rect_pet = Rectangle((0,0), 1, 1, linewidth=2, edgecolor='yellow', facecolor='none', visible=False)
        self.ax_pet.add_patch(self.rect_pet)
        
        self.rect_fusion = Rectangle((0,0), 1, 1, linewidth=2, edgecolor='yellow', facecolor='none', visible=False)
        self.ax_fusion.add_patch(self.rect_fusion)
        
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
        # Overlays moved to side info panel
        pass

    def _on_scroll(self, event):
        if event.inaxes and self.presenter:
            # Handle Zoom (Ctrl + Scroll)
            if event.key == 'control':
                base_scale = 1.2
                # Scroll up (val>0) -> Zoom In (Limits shrink) -> Factor < 1
                scale_factor = 1/base_scale if event.button == 'up' else base_scale
                
                ax = event.inaxes
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()
                
                xdata = event.xdata
                ydata = event.ydata
                if xdata is None or ydata is None: return

                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
                
                relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
                rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
                
                ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
                ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
                self.canvas.draw_idle()

            elif event.button == 'up':
                self.presenter.change_slice(1)
            elif event.button == 'down':
                self.presenter.change_slice(-1)

    def _on_mouse_press(self, event):
        if event.button == 3: # Right Click
            self.dragging = True
            self.mode = 'WL'
        elif event.button == 2: # Middle Click
            self.dragging = True
            self.mode = 'PAN'
        
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def _on_mouse_move(self, event):
        # Handle Dragging
        if self.dragging and self.presenter:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            
            if hasattr(self, 'mode') and self.mode == 'PAN':
                 if event.inaxes:
                     ax = event.inaxes
                     xlim = ax.get_xlim()
                     ylim = ax.get_ylim()
                     
                     # Pixel to data scale estimate
                     bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                     width_px = bbox.width * self.fig.dpi
                     height_px = bbox.height * self.fig.dpi
                     
                     if width_px > 0 and height_px > 0:
                         scale_x = (xlim[1] - xlim[0]) / width_px
                         scale_y = (ylim[1] - ylim[0]) / height_px
                         
                         ax.set_xlim(xlim[0] - dx*scale_x, xlim[1] - dx*scale_x)
                         ax.set_ylim(ylim[0] + dy*scale_y, ylim[1] + dy*scale_y)
                         self.canvas.draw_idle()
                     
            elif getattr(self, 'mode', 'WL') == 'WL':
                self.presenter.change_window_level(dx, dy)
            
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            
        # Handle Hover (Pixel Probe)
        if event.inaxes:
            try:
                x, y = int(event.xdata), int(event.ydata)
                
                # Determine which image source to probe
                val = None
                img_source = "?"
                val_unit = ""
                
                if event.inaxes == self.ax_ct:
                    img_source = "CT"
                    val_unit = "HU"
                    val = self.img_ct.get_cursor_data(event)
                    
                elif event.inaxes == self.ax_pet:
                    img_source = "PET"
                    val = self.img_pet.get_cursor_data(event)
                    
                    # Convert to SUV
                    suv_factor = self.presenter.model.get_suv_factor()
                    if val is not None:
                         if hasattr(val, 'item'): val = val.item()
                         val = val * suv_factor
                    val_unit = "SUV bw (g/ml)"

                elif event.inaxes == self.ax_fusion:
                    img_source = "PET NAC"
                    val = self.img_pet_nac.get_cursor_data(event)
                    suv_factor = self.presenter.model.get_suv_factor()
                    if val is not None:
                        if hasattr(val, 'item'):
                            val = val.item()
                        val = val * suv_factor
                    val_unit = "SUV bw (g/ml)"
                
                if val is not None:
                    if hasattr(val, 'item'): val = val.item()
                    if not (np.ma.is_masked(val) or np.isnan(val)):
                        val_text = f"[{img_source}] X: {x} Y: {y}  Value: {val:.2f} {val_unit}"
                        self.status_bar_var.set(val_text)
                    else:
                        self.status_bar_var.set(f"[{img_source}] Background")
                else:
                    self.status_bar_var.set(f"[{img_source}] Out of bounds")
            except Exception:
                 # self.status_bar_var.set("Error")
                 pass

    def _create_controls(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Status Bar for Hover Info
        self.status_bar_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_bar_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Patient Info Panel (Floating Top Right or Integrated)
        # Using a Frame packed to the right of controls might get hidden.
        # Let's put it in a separate frame above controls or use place()
        self.info_label_var = tk.StringVar(value="No Data")
        
        # We can pack it into the existing toolbar or create a new frame
        # Let's try to find the toolbar? No, let's just make a floating label on the canvas?
        # Or just a LabelFrame in the Control area?
        info_frame = ttk.LabelFrame(control_frame, text="Patient Info")
        info_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)
        ttk.Label(info_frame, textvariable=self.info_label_var, justify=tk.LEFT).pack(padx=5, pady=5)

        # Segmentation Selection
        seg_frame = ttk.LabelFrame(control_frame, text="Segmentation")
        seg_frame.pack(side=tk.LEFT, padx=10, pady=5)

        ttk.Label(seg_frame, text="Source").pack(padx=5, pady=(2, 0), anchor=tk.W)
        self.seg_source_var = tk.StringVar(value=self.current_segmentation_source)
        self.seg_source_label = ttk.Label(seg_frame, textvariable=self.seg_source_var)
        self.seg_source_label.pack(padx=5, pady=2, anchor=tk.W)

        ttk.Label(seg_frame, text="Class").pack(padx=5, pady=(4, 0), anchor=tk.W)
        self.segmentation_var = tk.StringVar(value="All Classes")
        self.segmentation_combo = ttk.Combobox(seg_frame, textvariable=self.segmentation_var, state="disabled", width=28, values=["All Classes"])
        self.segmentation_combo.pack(padx=5, pady=2)
        self.segmentation_combo.bind("<<ComboboxSelected>>", self._on_segmentation_class_change)
        
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

    def _on_segmentation_class_change(self, event=None):
        selection = self.segmentation_var.get()
        label_value = self.segmentation_value_map.get(selection, 'all_classes')
        self.current_segmentation_label = label_value
        if self.presenter:
            self.presenter.set_segmentation_class(label_value)

    def _on_segmentation_source_change(self, event=None):
        # Segmentation source is fixed in the viewer.
        return

    def update_images(self, ct_img, pet_img, pet_nac_img, seg_img, slice_idx, wl=50, ww=400, aspect=1.0, 
                      ct_extent=None, pet_extent=None, pet_nac_extent=None, segmentation_label='all_classes', zoi_box=None):
        # Update Images with Physical Extents
        self.current_segmentation_label = segmentation_label or 'all_classes'
        self.current_seg_img = seg_img  # Store raw seg for probing
        
        # Update ZOI box visibility and position
        if zoi_box:
            # zoi_box = [x, y, w, h] in physical coords.
            # Matplotlib Rectangle takes (x,y), w, h.
            x, y, w, h = zoi_box
            self.rect_ct.set_xy((x, y))
            self.rect_ct.set_width(w)
            self.rect_ct.set_height(h)
            self.rect_ct.set_visible(True)
            
            self.rect_pet.set_xy((x, y))
            self.rect_pet.set_width(w)
            self.rect_pet.set_height(h)
            self.rect_pet.set_visible(True)

            self.rect_fusion.set_xy((x, y))
            self.rect_fusion.set_width(w)
            self.rect_fusion.set_height(h)
            self.rect_fusion.set_visible(True)
        else:
            self.rect_ct.set_visible(False)
            self.rect_pet.set_visible(False)
            self.rect_fusion.set_visible(False)

        # 1. CT
        if ct_img is not None:
            self.img_ct.set_data(ct_img)
            if ct_extent:
                # extent=[left, right, bottom, top]
                self.img_ct.set_extent(ct_extent)
                self.ax_ct.set_xlim(ct_extent[0], ct_extent[1])
                self.ax_ct.set_ylim(ct_extent[2], ct_extent[3])
                # Force equal aspect ratio
                self.ax_ct.set_aspect('equal')
            else:
                self.img_ct.set_extent([0, ct_img.shape[1], ct_img.shape[0], 0])
                self.ax_ct.set_aspect(aspect)

            # Apply Window/Level
            vmin = wl - (ww / 2)
            vmax = wl + (ww / 2)
            self.img_ct.set_clim(vmin, vmax)
            self.ax_ct.set_title(f"CT (Slice {slice_idx})", color='white')

            overlay = self._build_segmentation_overlay(seg_img, self.current_segmentation_label)
            if overlay is not None:
                self.img_seg_overlay.set_data(overlay)
                self.img_seg_overlay.set_alpha(1.0)
                if ct_extent:
                    self.img_seg_overlay.set_extent(ct_extent)
                else:
                    self.img_seg_overlay.set_extent([0, seg_img.shape[1], seg_img.shape[0], 0])
            else:
                # Clear overlay
                blank = np.zeros((ct_img.shape[0], ct_img.shape[1], 4), dtype=float)
                self.img_seg_overlay.set_data(blank)
                self.img_seg_overlay.set_alpha(0.0)

        # 2. PET AC
        if pet_img is not None:
            self.img_pet.set_data(pet_img)
            
            if pet_extent:
                self.img_pet.set_extent(pet_extent)
                self.ax_pet.set_xlim(pet_extent[0], pet_extent[1])
                self.ax_pet.set_ylim(pet_extent[2], pet_extent[3])
                self.ax_pet.set_aspect('equal')
            elif ct_extent:
                # Fallback to CT extent if PET extent is missing (assumes alignment)
                self.img_pet.set_extent(ct_extent)
                self.ax_pet.set_xlim(ct_extent[0], ct_extent[1])
                self.ax_pet.set_ylim(ct_extent[2], ct_extent[3])
                self.ax_pet.set_aspect('equal')
            else:
                self.img_pet.set_extent([0, pet_img.shape[1], pet_img.shape[0], 0])
                self.ax_pet.set_aspect(aspect)
                
            self.img_pet.set_clim(0, np.max(pet_img) if np.max(pet_img) > 0 else 1)
            self.ax_pet.set_title(f"PET AC (Slice {slice_idx})", color='white')
        else:
            if ct_img is not None:
                blank = np.zeros_like(ct_img)
            else:
                blank = np.zeros((512, 512))
            self.img_pet.set_data(blank)
            self.img_pet.set_clim(0, 1)
            self.ax_pet.set_title("PET AC", color='white')

        # 3. PET NAC
        if pet_nac_img is not None:
            self.img_pet_nac.set_data(pet_nac_img)

            if pet_nac_extent:
                self.img_pet_nac.set_extent(pet_nac_extent)
                self.ax_fusion.set_xlim(pet_nac_extent[0], pet_nac_extent[1])
                self.ax_fusion.set_ylim(pet_nac_extent[2], pet_nac_extent[3])
                self.ax_fusion.set_aspect('equal')
            elif ct_extent:
                self.img_pet_nac.set_extent(ct_extent)
                self.ax_fusion.set_xlim(ct_extent[0], ct_extent[1])
                self.ax_fusion.set_ylim(ct_extent[2], ct_extent[3])
                self.ax_fusion.set_aspect('equal')
            else:
                self.img_pet_nac.set_extent([0, pet_nac_img.shape[1], pet_nac_img.shape[0], 0])
                self.ax_fusion.set_aspect(aspect)

            self.img_pet_nac.set_clim(0, np.max(pet_nac_img) if np.max(pet_nac_img) > 0 else 1)
            self.ax_fusion.set_title(f"PET NAC (Slice {slice_idx})", color='white')
        else:
            if ct_img is not None:
                blank = np.zeros_like(ct_img)
            else:
                blank = np.zeros((512, 512))
            self.img_pet_nac.set_data(blank)
            self.img_pet_nac.set_clim(0, 1)
            self.ax_fusion.set_title("PET NAC", color='white')
        
        self.canvas.draw_idle()

    def update_overlays(self, metadata_dict):
        # Update Side Panel Info
        info = []
        if 'name' in metadata_dict: info.append(f"Name: {metadata_dict['name']}")
        if 'id' in metadata_dict: info.append(f"ID: {metadata_dict['id']}")
        if 'wl' in metadata_dict: info.append(f"WL/WW: {int(metadata_dict['wl'])}/{int(metadata_dict['ww'])}")
        if 'slice' in metadata_dict: info.append(f"Slice: {metadata_dict['slice']}")
        if 'thickness' in metadata_dict: info.append(f"Thick: {metadata_dict['thickness']}mm")
        if 'pos' in metadata_dict: info.append(f"Z: {metadata_dict.get('pos', 0):.1f}mm")
        
        if hasattr(self, 'info_label_var'):
            self.info_label_var.set(" | ".join(info))
        elif hasattr(self, 'patient_lbl'):
             # Fallback
             pass

    def _get_pixel_value_at_location(self, artist, x, y, data_override=None):
        """ Returns the interpolated or nearest value from an artist at physical coordinates x, y. """
        if artist is None: return None
        
        try:
             # Check if coords are inside extent
             extent = artist.get_extent() # left, right, bottom, top
             # Handle inverted Y axis if top < bottom or standard if bottom < top. 
             # Standard image extent in mpl with origin='upper': top is usually 0 (or low), bottom is high? No.
             # In set_extent([L, R, B, T]): Bottom is max Y in image coords for 'upper'?
             # Let's simple check min/max.
             
             x_min, x_max = min(extent[0], extent[1]), max(extent[0], extent[1])
             y_min, y_max = min(extent[2], extent[3]), max(extent[2], extent[3])
             
             if not (x_min <= x <= x_max and y_min <= y <= y_max):
                 return None
             
             # Map physical to index
             data = data_override if data_override is not None else artist.get_array()
             if data is None: return None
             
             h, w = data.shape[:2]  # Handle RGBA shapes too
             
             # Calculate ratios
             # For X: (x - left) / (right - left)
             u = (x - extent[0]) / (extent[1] - extent[0])
             
             # For Y: 
             # If origin is upper (standard for medical): Top (extent[3]) corresponds to row 0.
             # Bottom (extent[2]) corresponds to row H.
             # So v = (y - top) / (bottom - top) 
             # Wait, usually extent[2] is bottom, extent[3] is top.
             # y-axis points upward in plots usually, so top > bottom.
             # But if update_images uses [0, W, H, 0], then Bottom=H, Top=0.
             # So Top < Bottom.
             v = (y - extent[3]) / (extent[2] - extent[3])
             
             ix = int(u * w)
             iy = int(v * h)
             
             # Clamp just in case
             ix = max(0, min(w-1, ix))
             iy = max(0, min(h-1, iy))
             
             val = data[iy, ix]
             if np.ma.is_masked(val) or np.isnan(val):
                 return None
             return val
        except Exception:
            return None

    def _sync_zoom_pan(self, xlim, ylim):
        """ Applies the same limits to all compatible axes """
        for ax in [self.ax_ct, self.ax_pet, self.ax_fusion]:
            if ax:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes and self.presenter:
            # Handle Zoom (Ctrl + Scroll)
            if event.key == 'control':
                base_scale = 1.2
                # Scroll up (val>0) -> Zoom In (Limits shrink) -> Factor < 1
                scale_factor = 1/base_scale if event.button == 'up' else base_scale
                
                ax = event.inaxes
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()
                
                xdata = event.xdata
                ydata = event.ydata
                if xdata is None or ydata is None: return

                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
                
                relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
                rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
                
                new_xlim = [xdata - new_width * (1-relx), xdata + new_width * (relx)]
                new_ylim = [ydata - new_height * (1-rely), ydata + new_height * (rely)]
                
                # Apply to ALL axes
                self._sync_zoom_pan(new_xlim, new_ylim)

            elif event.button == 'up':
                self.presenter.change_slice(1)
            elif event.button == 'down':
                self.presenter.change_slice(-1)

    def _on_mouse_press(self, event):
        if event.button == 3: # Right Click
            self.dragging = True
            self.mode = 'WL'
        elif event.button == 2: # Middle Click
            self.dragging = True
            self.mode = 'PAN'
        
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def _on_mouse_move(self, event):
        # Handle Dragging
        if self.dragging and self.presenter:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            
            if hasattr(self, 'mode') and self.mode == 'PAN':
                 if event.inaxes:
                     ax = event.inaxes
                     xlim = ax.get_xlim()
                     ylim = ax.get_ylim()
                     
                     # Pixel to data scale estimate
                     bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                     width_px = bbox.width * self.fig.dpi
                     height_px = bbox.height * self.fig.dpi
                     
                     if width_px > 0 and height_px > 0:
                         scale_x = (xlim[1] - xlim[0]) / width_px
                         scale_y = (ylim[1] - ylim[0]) / height_px
                         
                         new_xlim = [xlim[0] - dx*scale_x, xlim[1] - dx*scale_x]
                         new_ylim = [ylim[0] + dy*scale_y, ylim[1] + dy*scale_y]
                         
                         self._sync_zoom_pan(new_xlim, new_ylim)
                     
            elif getattr(self, 'mode', 'WL') == 'WL':
                self.presenter.change_window_level(dx, dy)
            
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            
        # Handle Hover (Pixel Probe)
        if event.inaxes:
            try:
                # Use physical coordinates from event
                x, y = event.xdata, event.ydata
                
                # Get Values from both sources
                val_ct = self._get_pixel_value_at_location(self.img_ct, x, y)
                val_pet = self._get_pixel_value_at_location(self.img_pet, x, y)
                val_pet_nac = self._get_pixel_value_at_location(self.img_pet_nac, x, y)
                
                # Get segmentation class using the overlay artist's geometry
                val_seg = self._get_pixel_value_at_location(self.img_seg_overlay, x, y, data_override=self.current_seg_img)

                status_parts = []
                status_parts.append(f"Pos: ({x:.1f}, {y:.1f})")
                
                if val_ct is not None:
                     status_parts.append(f"CT: {val_ct:.1f} HU")
                
                if val_pet is not None:
                     suv_factor = self.presenter.model.get_suv_factor()
                     val_suv = val_pet * suv_factor
                     status_parts.append(f"PET AC: {val_suv:.2f} SUV")

                if val_pet_nac is not None:
                    suv_factor = self.presenter.model.get_suv_factor()
                    val_suv = val_pet_nac * suv_factor
                    status_parts.append(f"PET NAC: {val_suv:.2f} SUV")
                
                if val_seg is not None and val_seg > 0:
                     label_id = int(val_seg)
                     label_name = self.segmentation_label_map.get(label_id, str(label_id))
                     status_parts.append(f"Class: {label_name}")

                if not val_ct and not val_pet and not val_pet_nac:
                    status_parts.append("Background")

                self.status_bar_var.set(" | ".join(status_parts))
                
            except Exception as e:
                 # print(e)
                 pass


    def _on_mouse_release(self, event):
        self.dragging = False

        # Force redraw is called in update_images mostly, but if only text changes:
        # self.canvas.draw_idle()

    def set_max_slice(self, max_slice):
        self.slice_scale.config(to=max_slice - 1)
        
    def set_current_patient_info(self, info_text):
        self.patient_lbl.config(text=info_text)

    def set_segmentation_sources(self, sources, current=None):
        self.segmentation_sources = sources or []
        target = current if current in (self.segmentation_sources or []) else self.current_segmentation_source
        self.seg_source_var.set(target)
        self.current_segmentation_source = target

    def set_segmentation_source_selection(self, source_name):
        if not source_name:
            return
        if source_name in (self.segmentation_sources or []):
            self.seg_source_var.set(source_name)
            self.current_segmentation_source = source_name

    def set_segmentation_classes(self, classes, label_map=None, default='all_classes'):
        classes = classes or []
        self.segmentation_label_map = label_map or {}
        self.segmentation_value_map = {"All Classes": 'all_classes'}
        display_values = ["All Classes"]
        for item in classes:
            display_name = f"[{item['id']}] {item['name']}"
            display_values.append(display_name)
            self.segmentation_value_map[display_name] = item['id']

        self.segmentation_combo['values'] = display_values
        if len(display_values) > 1:
            self.segmentation_combo.config(state='readonly')
        else:
            self.segmentation_combo.config(state='disabled')

        self._generate_segmentation_colors(classes)
        self.set_current_segmentation_selection(default)

    def set_current_segmentation_selection(self, label_value):
        target_value = label_value if label_value in self.segmentation_value_map.values() else 'all_classes'
        for display, value in self.segmentation_value_map.items():
            if value == target_value:
                self.segmentation_var.set(display)
                break
        else:
            self.segmentation_var.set("All Classes")
            target_value = 'all_classes'
        self.current_segmentation_label = target_value

    def _generate_segmentation_colors(self, classes):
        self.segmentation_color_lut = {}
        for item in classes:
            label_id = item.get('id')
            if label_id is None:
                continue
            self.segmentation_color_lut[label_id] = self._compute_color_from_label(label_id)

    def _compute_color_from_label(self, label_id):
        hue = ((int(label_id) * 37) % 360) / 360.0
        rgb = mcolors.hsv_to_rgb((hue, 0.65, 0.95))
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.65)

    def _get_color_for_label(self, label_id):
        label_id = int(label_id)
        if label_id not in self.segmentation_color_lut:
            self.segmentation_color_lut[label_id] = self._compute_color_from_label(label_id)
        return self.segmentation_color_lut[label_id]

    def _build_segmentation_overlay(self, seg_img, segmentation_label):
        if seg_img is None:
            return None
        if not np.any(seg_img):
            return None

        label_value = segmentation_label or 'all_classes'
        overlay = np.zeros(seg_img.shape + (4,), dtype=float)

        if label_value == 'all_classes':
            unique_labels = [int(v) for v in np.unique(seg_img) if v != 0]
            for label_id in unique_labels:
                color = self._get_color_for_label(label_id)
                overlay[seg_img == label_id] = color
        else:
            try:
                label_id = int(label_value)
            except (ValueError, TypeError):
                return None
            mask = seg_img == label_id
            if not np.any(mask):
                return None
            color = self._get_color_for_label(label_id)
            overlay[mask] = color

        if np.max(overlay[..., 3]) == 0:
            return None
        return overlay

    def zoom_to_bounds(self, bounds):
        """
        bounds: [xmin, xmax, ymax, ymin] or similar.
        Note: model.get_segmentation_bounds returns [x1, x2, bottom, top] for Axial where Bottom>Top (value-wise) if inverted Y?
        Matplotlib axis set_ylim(bottom, top).
        DataViewer uses standard image coordinates where Y increases downwards?
        Typically in _on_scroll, ylim is [min, max].
        Let's interpret bounds as [x_min, x_max, y_min, y_max] to be safe, 
        or strictly trust the presenter. The present passes result from get_segmentation_bounds directly.
        """
        if not bounds: return
        # model returns [x1, x2, y2, y1] for AXIAL where y2 > y1 usually?
        # Let's apply directly to limits.
        xlim = [bounds[0], bounds[1]]
        ylim = [bounds[2], bounds[3]]
        self._sync_zoom_pan(xlim, ylim)
