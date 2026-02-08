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
        
        # MPR Buttons
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Button(toolbar_frame, text="Axial", command=lambda: self.presenter.set_orientation('AXIAL')).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Coronal", command=lambda: self.presenter.set_orientation('CORONAL')).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Sagittal", command=lambda: self.presenter.set_orientation('SAGITTAL')).pack(side=tk.LEFT, padx=2)

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
                     # Probe CT on fusion for now
                     img_source = "Fusion(CT)"
                     val_unit = "HU"
                     val = self.img_fusion.get_cursor_data(event)
                
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
            
            # Store base text for hover append
            self.last_wl_text = txt
            
            self.overlays['ct_bl'].set_text(txt)
            
        if 'thickness' in metadata_dict:
             self.overlays['ct_br'].set_text(f"Thickness: {metadata_dict['thickness']}mm")

    def _create_controls(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Status Bar for Hover Info
        self.status_bar_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_bar_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
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

    def update_images(self, ct_img, pet_img, slice_idx, wl=50, ww=400, aspect=1.0, ct_extent=None, pet_extent=None):
        # Update Images with Physical Extents
        
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
             
             # 3. Fusion (Base Layer CT)
             self.img_fusion.set_data(ct_img) 
             self.img_fusion.set_clim(vmin, vmax)
             if ct_extent:
                 self.img_fusion.set_extent(ct_extent)
                 self.ax_fusion.set_xlim(ct_extent[0], ct_extent[1])
                 self.ax_fusion.set_ylim(ct_extent[2], ct_extent[3])
                 self.ax_fusion.set_aspect('equal')
             else:
                 self.img_fusion.set_extent([0, ct_img.shape[1], ct_img.shape[0], 0])
                 self.ax_fusion.set_aspect(aspect)

        # 2. PET
        if pet_img is not None:
             self.img_pet.set_data(pet_img)
             
             if pet_extent:
                 self.img_pet.set_extent(pet_extent)
                 self.ax_pet.set_xlim(pet_extent[0], pet_extent[1])
                 self.ax_pet.set_ylim(pet_extent[2], pet_extent[3])
                 self.ax_pet.set_aspect('equal')
             else:
                 self.img_pet.set_extent([0, pet_img.shape[1], pet_img.shape[0], 0])
                 self.ax_pet.set_aspect(aspect)
                 
             self.img_pet.set_clim(0, np.max(pet_img) if np.max(pet_img) > 0 else 1)
             
             # Fusion Overlay (PET on CT)
             # Check if we already have the overlay text artist or image
             if not hasattr(self, 'img_fusion_overlay'):
                  blank = np.zeros_like(pet_img)
                  self.img_fusion_overlay = self.ax_fusion.imshow(blank, cmap='hot', alpha=0.4)
             
             self.img_fusion_overlay.set_data(pet_img)
             if pet_extent:
                 self.img_fusion_overlay.set_extent(pet_extent)
             else:
                 self.img_fusion_overlay.set_extent([0, pet_img.shape[1], pet_img.shape[0], 0])

             self.img_fusion_overlay.set_clim(0, np.max(pet_img) if np.max(pet_img) > 0 else 1)
             
        else:
             pass
        
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
            
            # Store base text for hover append
            self.last_wl_text = txt
            
            self.overlays['ct_bl'].set_text(txt)
            
        if 'thickness' in metadata_dict:
             self.overlays['ct_br'].set_text(f"Thickness: {metadata_dict['thickness']}mm")

    def _get_pixel_value_at_location(self, artist, x, y):
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
             data = artist.get_array()
             if data is None: return None
             
             h, w = data.shape
             
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
                
                status_parts = []
                status_parts.append(f"Pos: ({x:.1f}, {y:.1f})")
                
                if val_ct is not None:
                     status_parts.append(f"CT: {val_ct:.1f} HU")
                
                if val_pet is not None:
                     suv_factor = self.presenter.model.get_suv_factor()
                     val_suv = val_pet * suv_factor
                     status_parts.append(f"PET: {val_suv:.2f} SUV")
                
                if not val_ct and not val_pet:
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
