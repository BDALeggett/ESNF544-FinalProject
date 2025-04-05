import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
from predict_card_grade import predict_grade

def highlight_card_defects(image_path, predictions):
    """
    Analyze the card image and highlight potential defects with red boxes
    Returns the original image with highlighted defects
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Create a copy for highlighting
    highlighted_image = image.copy()
    
    # Get predicted grade (use the top prediction)
    if not predictions or len(predictions) < 1:
        return highlighted_image
    
    top_model, top_grade, confidence = predictions[0]
    
    # Process the image to find the yellow border
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find contours of the yellow border
    contours, _ = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return highlighted_image
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Different highlighting strategies based on grade
    if top_grade == "≤7":
        # For poor grades, highlight multiple issues
        
        # Check corners
        corner_size = min(w, h) // 8
        corners = [
            (x, y, corner_size, corner_size),  # Top-left
            (x + w - corner_size, y, corner_size, corner_size),  # Top-right
            (x, y + h - corner_size, corner_size, corner_size),  # Bottom-left
            (x + w - corner_size, y + h - corner_size, corner_size, corner_size)  # Bottom-right
        ]
        
        # Highlight corners with issues
        for cx, cy, cw, ch in corners:
            corner_img = yellow_mask[cy:cy+ch, cx:cx+cw]
            if corner_img.size > 0:
                white_ratio = np.count_nonzero(corner_img) / corner_img.size
                if white_ratio < 0.85:  # Less than 85% yellow in corner indicates damage
                    cv2.rectangle(highlighted_image, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
        
        # Highlight centering issues
        left_border = x
        right_border = image.shape[1] - (x + w)
        top_border = y
        bottom_border = image.shape[0] - (y + h)
        
        # If borders differ by more than 20%, highlight the smaller border
        h_diff = abs(left_border - right_border) / max(left_border, right_border)
        v_diff = abs(top_border - bottom_border) / max(top_border, bottom_border)
        
        if h_diff > 0.2:
            if left_border < right_border:
                cv2.rectangle(highlighted_image, (0, y), (x, y+h), (0, 0, 255), 2)
            else:
                cv2.rectangle(highlighted_image, (x+w, y), (image.shape[1], y+h), (0, 0, 255), 2)
        
        if v_diff > 0.2:
            if top_border < bottom_border:
                cv2.rectangle(highlighted_image, (x, 0), (x+w, y), (0, 0, 255), 2)
            else:
                cv2.rectangle(highlighted_image, (x, y+h), (x+w, image.shape[0]), (0, 0, 255), 2)
                
    elif top_grade == "8":
        # For grade 8, highlight moderate issues
        
        # Check corners with less strict criteria
        corner_size = min(w, h) // 10
        corners = [
            (x, y, corner_size, corner_size),  # Top-left
            (x + w - corner_size, y, corner_size, corner_size),  # Top-right
            (x, y + h - corner_size, corner_size, corner_size),  # Bottom-left
            (x + w - corner_size, y + h - corner_size, corner_size, corner_size)  # Bottom-right
        ]
        
        for cx, cy, cw, ch in corners:
            corner_img = yellow_mask[cy:cy+ch, cx:cx+cw]
            if corner_img.size > 0:
                white_ratio = np.count_nonzero(corner_img) / corner_img.size
                if white_ratio < 0.9:  # Slightly less strict threshold
                    cv2.rectangle(highlighted_image, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
        
        # Check for edge wear
        edge_thickness = 5
        edges = [
            (x, y, w, edge_thickness),  # Top edge
            (x, y, edge_thickness, h),  # Left edge
            (x, y + h - edge_thickness, w, edge_thickness),  # Bottom edge
            (x + w - edge_thickness, y, edge_thickness, h)  # Right edge
        ]
        
        for ex, ey, ew, eh in edges:
            edge_img = yellow_mask[ey:ey+eh, ex:ex+ew]
            if edge_img.size > 0:
                white_ratio = np.count_nonzero(edge_img) / edge_img.size
                if white_ratio < 0.92:  # Less than 92% yellow on edge indicates wear
                    cv2.rectangle(highlighted_image, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
                    
    elif top_grade == "9":
        # For grade 9, highlight minor issues
        
        # Check only specific areas for very minor issues
        # Mainly just look at corners with stricter criteria
        corner_size = min(w, h) // 12
        corners = [
            (x, y, corner_size, corner_size),  # Top-left
            (x + w - corner_size, y, corner_size, corner_size),  # Top-right
            (x, y + h - corner_size, corner_size, corner_size),  # Bottom-left
            (x + w - corner_size, y + h - corner_size, corner_size, corner_size)  # Bottom-right
        ]
        
        for cx, cy, cw, ch in corners:
            corner_img = yellow_mask[cy:cy+ch, cx:cx+cw]
            if corner_img.size > 0:
                white_ratio = np.count_nonzero(corner_img) / corner_img.size
                if white_ratio < 0.95:  # Very strict threshold
                    cv2.rectangle(highlighted_image, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
                    
        # Check for subtle centering issues
        left_border = x
        right_border = image.shape[1] - (x + w)
        top_border = y
        bottom_border = image.shape[0] - (y + h)
        
        # Much stricter centering criteria for grade 9
        h_diff = abs(left_border - right_border) / max(left_border, right_border)
        v_diff = abs(top_border - bottom_border) / max(top_border, bottom_border)
        
        if h_diff > 0.1:  # Only 10% difference allowed
            if left_border < right_border:
                cv2.rectangle(highlighted_image, (0, y), (x, y+h), (0, 0, 255), 2)
            else:
                cv2.rectangle(highlighted_image, (x+w, y), (image.shape[1], y+h), (0, 0, 255), 2)
        
        if v_diff > 0.1:
            if top_border < bottom_border:
                cv2.rectangle(highlighted_image, (x, 0), (x+w, y), (0, 0, 255), 2)
            else:
                cv2.rectangle(highlighted_image, (x, y+h), (x+w, image.shape[0]), (0, 0, 255), 2)
    
    # For grade 10, we don't highlight anything as it's near perfect
    
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
            self.display_image(self.original_image)
            self.predict_card_grade(file_path)
    
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
            # Show original image
            if self.original_image is not None:
                self.display_image(self.original_image)
                self.status_var.set("Showing original image")
    
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