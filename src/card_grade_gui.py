import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
from predict_card_grade import predict_grade
from crop_single_image import crop_pokemon_card

def highlight_card_defects(image_path, predictions):
    """
    Analyze the card image and highlight actual physical defects such as
    corner wear, scratches, and edge damage
    Returns the original image with highlighted defects
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Create a copy for highlighting
    highlighted_image = image.copy()
    
    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use edge detection to find the overall card first (not just yellow border)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    
    # Find contours of the entire card
    card_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not card_contours:
        # Fall back to yellow border detection if card detection fails
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return highlighted_image
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
    else:
        # Use the largest contour as the card
        largest_card_contour = max(card_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_card_contour)
    
    # Find the yellow border for defect analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 1. DETECT CORNER WEAR - IMPROVED METHOD
    # Calculate corner size based on card dimensions
    corner_size = min(w, h) // 7  # Slightly larger to ensure we get the entire corner
    
    # Find the actual corners of the card using corner detection
    corners_harris = cv2.cornerHarris(gray, 5, 3, 0.04)
    corners_norm = cv2.normalize(corners_harris, None, 0, 255, cv2.NORM_MINMAX)
    corners_norm = np.uint8(corners_norm)
    
    # Threshold for corner detection
    _, corners_thresh = cv2.threshold(corners_norm, 100, 255, cv2.THRESH_BINARY)
    
    # Find coordinates of corner points
    corner_coords = []
    corner_regions = []
    
    # Define the corner regions first
    tl_region = gray[y:y+corner_size, x:x+corner_size]
    tr_region = gray[y:y+corner_size, x+w-corner_size:x+w]
    bl_region = gray[y+h-corner_size:y+h, x:x+corner_size]
    br_region = gray[y+h-corner_size:y+h, x+w-corner_size:x+w]
    
    corner_regions = [
        {"region": tl_region, "pos": (x, y), "name": "Top-left"},
        {"region": tr_region, "pos": (x+w-corner_size, y), "name": "Top-right"},
        {"region": bl_region, "pos": (x, y+h-corner_size), "name": "Bottom-left"},
        {"region": br_region, "pos": (x+w-corner_size, y+h-corner_size), "name": "Bottom-right"}
    ]
    
    for corner in corner_regions:
        if corner["region"].size == 0:
            continue
            
        # Apply Canny edge detection to find edges in the corner
        cx, cy = corner["pos"]
        cw, ch = corner_size, corner_size
        
        # Extract corner region for yellow mask check
        corner_mask = yellow_mask[cy:cy+ch, cx:cx+cw]
        corner_edges = cv2.Canny(corner["region"], 80, 200)
        
        # Count edge pixels - more edges can indicate wear
        edge_pixel_count = np.count_nonzero(corner_edges)
        
        # Calculate edge density
        if corner["region"].size > 0:
            edge_density = edge_pixel_count / corner["region"].size
        else:
            edge_density = 0
        
        # Check corner yellow mask for completeness
        if corner_mask.size > 0:
            yellow_ratio = np.count_nonzero(corner_mask) / corner_mask.size
            
            # If corner has high edge density or missing yellow border, it may be worn
            if edge_density > 0.05 or yellow_ratio < 0.9:
                cv2.rectangle(highlighted_image, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
                cv2.putText(highlighted_image, "Corner wear", (cx, cy-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 2. DETECT EDGE DAMAGE
    edge_thickness = 5
    edges = [
        (x, y, w, edge_thickness),  # Top edge
        (x, y, edge_thickness, h),  # Left edge
        (x, y + h - edge_thickness, w, edge_thickness),  # Bottom edge
        (x + w - edge_thickness, y, edge_thickness, h)  # Right edge
    ]
    
    edge_names = ["Top", "Left", "Bottom", "Right"]
    
    for i, (ex, ey, ew, eh) in enumerate(edges):
        # Extract edge region
        edge_region = gray[ey:ey+eh, ex:ex+ew]
        if edge_region.size == 0:
            continue
            
        # Apply edge detection
        edge_edges = cv2.Canny(edge_region, 100, 200)
        
        # Check edge completeness in yellow mask
        edge_mask = yellow_mask[ey:ey+eh, ex:ex+ew]
        if edge_mask.size > 0:
            yellow_ratio = np.count_nonzero(edge_mask) / edge_mask.size
            
            # If edge has incomplete yellow border, highlight it
            if yellow_ratio < 0.92:
                cv2.rectangle(highlighted_image, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
                # Position the text based on which edge
                if i == 0:  # Top
                    text_pos = (ex + ew//2 - 40, ey - 5)
                elif i == 1:  # Left
                    text_pos = (ex - 5, ey + eh//2)
                elif i == 2:  # Bottom
                    text_pos = (ex + ew//2 - 40, ey + eh + 15)
                else:  # Right
                    text_pos = (ex + ew + 5, ey + eh//2)
                
                cv2.putText(highlighted_image, f"{edge_names[i]} edge wear", text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 3. DETECT SCRATCHES ON CARD SURFACE
    # Extract the card surface (avoid the borders)
    inner_margin = 20
    card_surface = gray[y+inner_margin:y+h-inner_margin, x+inner_margin:x+w-inner_margin]
    
    if card_surface.size > 0:
        # Use Canny edge detection with lower threshold to find subtle scratches
        surface_edges = cv2.Canny(card_surface, 50, 150)
        
        # Apply morphological operations to connect broken lines (potential scratches)
        kernel = np.ones((3, 3), np.uint8)
        surface_edges = cv2.dilate(surface_edges, kernel, iterations=1)
        
        # Find contours of potential scratches
        scratch_contours, _ = cv2.findContours(surface_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in scratch_contours:
            # Filter by size and shape to identify likely scratches
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate aspect ratio of the bounding rect
            x_s, y_s, w_s, h_s = cv2.boundingRect(contour)
            aspect_ratio = max(w_s, h_s) / (min(w_s, h_s) + 0.01)  # Avoid division by zero
            
            # Scratches typically have small area but large aspect ratio
            # and small area-to-perimeter ratio
            if area > 5 and aspect_ratio > 3 and area / (perimeter + 0.01) < 1.0:
                # Adjust coordinates to the full image
                contour[:, :, 0] += x + inner_margin
                contour[:, :, 1] += y + inner_margin
                
                # Draw the scratch contour
                cv2.drawContours(highlighted_image, [contour], 0, (0, 0, 255), 2)
                
                # Add label
                rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                cv2.putText(highlighted_image, "Scratch", (rect_x, rect_y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 4. CHECK CENTERING ISSUES
    left_border = x
    right_border = image.shape[1] - (x + w)
    top_border = y
    bottom_border = image.shape[0] - (y + h)
    
    # Calculate horizontal and vertical border differences
    h_diff = abs(left_border - right_border) / max(left_border, right_border, 1)
    v_diff = abs(top_border - bottom_border) / max(top_border, bottom_border, 1)
    
    # If centering is off by more than 15%, highlight it
    if h_diff > 0.15:
        if left_border < right_border:
            cv2.rectangle(highlighted_image, (0, y), (x, y+h), (0, 165, 255), 2)
            cv2.putText(highlighted_image, "Left centering", (5, y+h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        else:
            cv2.rectangle(highlighted_image, (x+w, y), (image.shape[1], y+h), (0, 165, 255), 2)
            cv2.putText(highlighted_image, "Right centering", (x+w+5, y+h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    
    if v_diff > 0.15:
        if top_border < bottom_border:
            cv2.rectangle(highlighted_image, (x, 0), (x+w, y), (0, 165, 255), 2)
            cv2.putText(highlighted_image, "Top centering", (x+w//2-40, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        else:
            cv2.rectangle(highlighted_image, (x, y+h), (x+w, image.shape[0]), (0, 165, 255), 2)
            cv2.putText(highlighted_image, "Bottom centering", (x+w//2-40, image.shape[0]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    
    return highlighted_image

class PokemonCardGraderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokemon Card Grader")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")
        
        # Set app icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(
            header_frame, 
            text="Pokemon Card Grader", 
            font=("Helvetica", 24, "bold")
        ).pack()
        
        ttk.Label(
            header_frame,
            text="Upload a Pokemon card image to predict its grade",
            font=("Helvetica", 12)
        ).pack(pady=5)
        
        # Create content frame (split into left and right)
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left side (image display)
        self.image_frame = ttk.LabelFrame(content_frame, text="Card Image", padding="10")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Default image display
        self.display_default_image()
        
        # Right side (controls and predictions)
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Upload button
        upload_frame = ttk.Frame(right_frame)
        upload_frame.pack(fill=tk.X, pady=10)
        
        self.upload_button = ttk.Button(
            upload_frame,
            text="Upload Card Image",
            command=self.upload_image
        )
        self.upload_button.pack(pady=10)
        
        # Add highlight defects checkbox
        self.highlight_var = tk.BooleanVar(value=False)
        self.highlight_check = ttk.Checkbutton(
            upload_frame,
            text="Highlight Card Defects",
            variable=self.highlight_var,
            command=self.toggle_highlight
        )
        self.highlight_check.pack(pady=5)
        
        # Status indicator
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(upload_frame, textvariable=self.status_var)
        self.status_label.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(upload_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=5)
        
        # Prediction results
        self.results_frame = ttk.LabelFrame(right_frame, text="Grade Predictions", padding="10")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Default message
        self.default_result_label = ttk.Label(
            self.results_frame,
            text="Upload a card to see predictions",
            font=("Helvetica", 10, "italic")
        )
        self.default_result_label.pack(pady=20)
        
        # Results treeview
        self.result_tree = ttk.Treeview(self.results_frame, columns=("grade", "model", "confidence"), show="headings")
        self.result_tree.heading("grade", text="Grade")
        self.result_tree.heading("model", text="Model")
        self.result_tree.heading("confidence", text="Confidence")
        
        self.result_tree.column("grade", width=80, anchor=tk.CENTER)
        self.result_tree.column("model", width=150)
        self.result_tree.column("confidence", width=120, anchor=tk.CENTER)
        
        # Footer
        footer_frame = ttk.Frame(self.main_frame)
        footer_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(
            footer_frame,
            text="© 2023 Pokemon Card Grader",
            font=("Helvetica", 8)
        ).pack(side=tk.RIGHT)
        
        # Store the current image path and predictions
        self.current_image_path = None
        self.current_predictions = None
        self.original_image = None
        self.highlighted_image = None
        
    def display_default_image(self):
        """Show a default placeholder image"""
        # Create a gray placeholder image
        placeholder = np.ones((300, 300, 3), dtype=np.uint8) * 230
        
        # Add text to the placeholder
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "No Image"
        textsize = cv2.getTextSize(text, font, 0.7, 1)[0]
        
        # Center the text
        x = (placeholder.shape[1] - textsize[0]) // 2
        y = (placeholder.shape[0] + textsize[1]) // 2
        
        cv2.putText(placeholder, text, (x, y), font, 0.7, (120, 120, 120), 1, cv2.LINE_AA)
        
        # Convert to tkinter image
        img = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        
        # Update the image label
        self.image_label.configure(image=img)
        self.image_label.image = img
    
    def upload_image(self):
        """Open file dialog to upload an image"""
        file_path = filedialog.askopenfilename(
            title="Select Card Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            # Read the original image and store it
            self.original_image = cv2.imread(file_path)
            cropped_image = crop_pokemon_card(self.original_image)
            if cropped_image is not None:
                self.display_image(cropped_image)
                cv2.imwrite("src/cropped_image.jpg", cropped_image)
                self.current_image_path = "src/cropped_image.jpg"
                print("Cropped image saved to cropped_image.jpg")
                self.predict_card_grade("src/cropped_image.jpg")
            else:
                print("No cropped image to save")
                self.display_image(self.current_image_path)
                self.predict_card_grade(self.current_image_path)

    
    def display_image(self, img):
        """Display the given image"""
        try:
            if isinstance(img, str):
                # If img is a path
                img = cv2.imread(img)
                if img is None:
                    raise ValueError(f"Failed to load image from {img}")
            
            # Convert from BGR to RGB for PIL
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get frame size
            frame_width = self.image_frame.winfo_width() - 20
            frame_height = self.image_frame.winfo_height() - 20
            
            if frame_width <= 1:  # Frame not yet sized, use default
                frame_width = 350
                frame_height = 400
            
            # Calculate aspect ratio
            h, w = img.shape[:2]
            aspect = w / h
            
            # Resize to fit the frame
            if h > frame_height or w > frame_width:
                if frame_width / aspect <= frame_height:
                    new_w = frame_width
                    new_h = int(frame_width / aspect)
                else:
                    new_h = frame_height
                    new_w = int(frame_height * aspect)
                
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to tkinter image
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            
            # Update the image label
            self.image_label.configure(image=img)
            self.image_label.image = img
            
        except Exception as e:
            self.status_var.set(f"Error displaying image: {e}")
    
    def toggle_highlight(self):
        """Toggle between original and highlighted image"""
        if not self.current_image_path or not self.current_predictions:
            return
            
        if self.highlight_var.get():
            # Show highlighted image
            if self.highlighted_image is None:
                # Generate highlighted image if not already done
                self.highlighted_image = highlight_card_defects(self.current_image_path, self.current_predictions)
            
            if self.highlighted_image is not None:
                self.display_image(self.highlighted_image)
                self.status_var.set("Showing defects")
        else:
            # Show cropped image instead of original image
            self.display_image(self.current_image_path)
            self.status_var.set("Showing cropped image")
    
    def predict_card_grade(self, image_path):
        """Run the prediction in a separate thread"""
        # Show progress and update status
        self.status_var.set("Processing...")
        self.progress.start(10)
        
        # Clear previous results
        self.default_result_label.pack_forget()
        for i in self.result_tree.get_children():
            self.result_tree.delete(i)
            
        # Reset images
        self.highlighted_image = None
        
        # Run prediction in a separate thread
        threading.Thread(target=self._run_prediction, args=(image_path,), daemon=True).start()
    
    def _run_prediction(self, image_path):
        """Background thread to run the prediction"""
        try:
            # Get predictions
            predictions = predict_grade(image_path)
            self.current_predictions = predictions
            
            # Update UI in the main thread
            self.root.after(0, lambda: self._update_results(predictions))
            
        except Exception as e:
            # Update UI with error
            self.root.after(0, lambda: self._show_error(str(e)))
    
    def _update_results(self, predictions):
        """Update UI with prediction results"""
        self.progress.stop()
        
        if predictions:
            # Show the treeview with results
            self.result_tree.pack(fill=tk.BOTH, expand=True)
            
            # Add results to the treeview
            for i, (model_name, grade, confidence) in enumerate(predictions):
                self.result_tree.insert(
                    "", "end", 
                    values=(grade, model_name, f"{confidence:.1f}%"),
                    tags=('top' if i == 0 else '')
                )
            
            # Highlight the top prediction
            self.result_tree.tag_configure('top', background='#e6f3ff')
            
            self.status_var.set("Prediction complete")
            
            # Enable the highlight checkbox
            self.highlight_check.state(['!disabled'])
            
            # If highlight option is selected, generate and display highlighted image
            if self.highlight_var.get():
                self.highlighted_image = highlight_card_defects(self.current_image_path, predictions)
                if self.highlighted_image is not None:
                    self.display_image(self.highlighted_image)
                    self.status_var.set("Showing defects")
        else:
            # Show error message
            self._show_error("Could not predict card grade")
            # Disable the highlight checkbox
            self.highlight_check.state(['disabled'])
    
    def _show_error(self, message):
        """Display error message"""
        self.progress.stop()
        self.status_var.set(f"Error: {message}")
        self.default_result_label.configure(text=f"Error: {message}")
        self.default_result_label.pack(pady=20)
        self.result_tree.pack_forget()
        # Disable the highlight checkbox
        self.highlight_check.state(['disabled'])

def main():
    root = tk.Tk()
    app = PokemonCardGraderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 