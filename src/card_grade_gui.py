import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
from predict_card_grade import predict_grade

class PokemonCardGraderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokemon Card Grader")
        self.root.geometry("800x600")
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
        
        # Store the current image path
        self.current_image_path = None
        
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
            self.display_image(file_path)
            self.predict_card_grade(file_path)
    
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            # Read and resize the image for display
            img = cv2.imread(image_path)
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
    
    def predict_card_grade(self, image_path):
        """Run the prediction in a separate thread"""
        # Show progress and update status
        self.status_var.set("Processing...")
        self.progress.start(10)
        
        # Clear previous results
        self.default_result_label.pack_forget()
        for i in self.result_tree.get_children():
            self.result_tree.delete(i)
        
        # Run prediction in a separate thread
        threading.Thread(target=self._run_prediction, args=(image_path,), daemon=True).start()
    
    def _run_prediction(self, image_path):
        """Background thread to run the prediction"""
        try:
            # Get predictions
            predictions = predict_grade(image_path)
            
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
        else:
            # Show error message
            self._show_error("Could not predict card grade")
    
    def _show_error(self, message):
        """Display error message"""
        self.progress.stop()
        self.status_var.set(f"Error: {message}")
        self.default_result_label.configure(text=f"Error: {message}")
        self.default_result_label.pack(pady=20)
        self.result_tree.pack_forget()

def main():
    root = tk.Tk()
    app = PokemonCardGraderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 