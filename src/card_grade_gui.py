# Import necessary libraries for image processing, GUI creation, and threading
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import os
from predict_card_grade import predict_grade
from crop_single_image import crop_pokemon_card
from predict_card_grade import highlight_card_defects

class PokemonCardGraderApp:
    """Main application class for the Pokemon Card Grader GUI."""
    
    def __init__(self, root):
        """Initialize the main application window and UI components."""
        self.root = root
        self.root.title("Pokemon Card Grader")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")
        
        # Create main frame with padding
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create and configure header section
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
        
        # Create main content area
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create image display area
        self.image_frame = ttk.LabelFrame(content_frame, text="Card Image", padding="10")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        self.display_default_image()
        
        # Create right panel for controls and results
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Create upload section
        upload_frame = ttk.Frame(right_frame)
        upload_frame.pack(fill=tk.X, pady=10)
        
        self.upload_button = ttk.Button(
            upload_frame,
            text="Upload Card Image",
            command=self.upload_image
        )
        self.upload_button.pack(pady=10)
        
        # Create highlight defects checkbox
        self.highlight_var = tk.BooleanVar(value=False)
        self.highlight_check = ttk.Checkbutton(
            upload_frame,
            text="Highlight Card Defects",
            variable=self.highlight_var,
            command=self.toggle_highlight
        )
        self.highlight_check.pack(pady=5)
        
        # Create status display
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(upload_frame, textvariable=self.status_var)
        self.status_label.pack(pady=5)
        
        # Create progress bar
        self.progress = ttk.Progressbar(upload_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=5)
        
        # Create results display area
        self.results_frame = ttk.LabelFrame(right_frame, text="Grade Predictions", padding="10")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.default_result_label = ttk.Label(
            self.results_frame,
            text="Upload a card to see predictions",
            font=("Helvetica", 10, "italic")
        )
        self.default_result_label.pack(pady=20)
        
        # Create results table
        self.result_tree = ttk.Treeview(self.results_frame, columns=("grade", "model", "confidence"), show="headings")
        self.result_tree.heading("grade", text="Grade")
        self.result_tree.heading("model", text="Model")
        self.result_tree.heading("confidence", text="Confidence")
        
        self.result_tree.column("grade", width=80, anchor=tk.CENTER)
        self.result_tree.column("model", width=150)
        self.result_tree.column("confidence", width=120, anchor=tk.CENTER)
        
        # Create footer
        footer_frame = ttk.Frame(self.main_frame)
        footer_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(
            footer_frame,
            text="© 2023 Pokemon Card Grader",
            font=("Helvetica", 8)
        ).pack(side=tk.RIGHT)
        
        # Initialize state variables
        self.current_image_path = None
        self.current_predictions = None
        self.original_image = None
        self.highlighted_image = None
        
        # Create temp directory if not exists
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def display_default_image(self):
        #Display a placeholder image when no card is uploaded.
        placeholder = np.ones((300, 300, 3), dtype=np.uint8) * 230
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "No Image"
        textsize = cv2.getTextSize(text, font, 0.7, 1)[0]
        
        x = (placeholder.shape[1] - textsize[0]) // 2
        y = (placeholder.shape[0] + textsize[1]) // 2
        
        cv2.putText(placeholder, text, (x, y), font, 0.7, (120, 120, 120), 1, cv2.LINE_AA)
        
        img = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        
        self.image_label.configure(image=img)
        self.image_label.image = img
    
    def upload_image(self):
        """Handle image upload and processing."""
        file_path = filedialog.askopenfilename(
            title="Select Card Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            # Update title to show bg removal status
            from crop_single_image import REMOVE_BACKGROUND
            bg_status = "BG Removal ON" if REMOVE_BACKGROUND else "BG Removal OFF"
            self.root.title(f"Pokemon Card Grader - {bg_status}")
            
            self.current_image_path = file_path
            self.original_image = cv2.imread(file_path)
            cropped_image = crop_pokemon_card(self.original_image)
            if cropped_image is not None:
                self.display_image(cropped_image)
                cropped_image_path = os.path.join(self.temp_dir, "cropped_image.jpg")
                cv2.imwrite(cropped_image_path, cropped_image)
                self.current_image_path = cropped_image_path
                print(f"Cropped image saved to {cropped_image_path}")
                self.predict_card_grade(cropped_image_path)
            else:
                print("No cropped image to save")
                self.display_image(self.current_image_path)
                self.predict_card_grade(self.current_image_path)

    def display_image(self, img):
        #Display the given image in the GUI with proper scaling.
        try:
            if isinstance(img, str):
                img = cv2.imread(img)
                if img is None:
                    raise ValueError(f"Failed to load image from {img}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            frame_width = self.image_frame.winfo_width() - 20
            frame_height = self.image_frame.winfo_height() - 20
            
            if frame_width <= 1:
                frame_width = 350
                frame_height = 400
            
            h, w = img.shape[:2]
            aspect = w / h
            
            if h > frame_height or w > frame_width:
                if frame_width / aspect <= frame_height:
                    new_w = frame_width
                    new_h = int(frame_width / aspect)
                else:
                    new_h = frame_height
                    new_w = int(frame_height * aspect)
                
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=img)
            self.image_label.image = img
            
        except Exception as e:
            self.status_var.set(f"Error displaying image: {e}")
    
    def toggle_highlight(self):
        #Toggle the display of card defects highlighting.
        if not self.current_image_path or not self.current_predictions:
            return
            
        if self.highlight_var.get():
            if self.highlighted_image is None:
                self.highlighted_image = highlight_card_defects(self.current_image_path, self.current_predictions)
            
            if self.highlighted_image is not None:
                self.display_image(self.highlighted_image)
                self.status_var.set("Showing defects")
        else:
            self.display_image(self.current_image_path)
            self.status_var.set("Showing cropped image")
    
    def predict_card_grade(self, image_path):
        #Start the card grade prediction process.
        self.status_var.set("Processing...")
        self.progress.start(10)
        
        self.default_result_label.pack_forget()
        for i in self.result_tree.get_children():
            self.result_tree.delete(i)
            
        self.highlighted_image = None
        
        threading.Thread(target=self._run_prediction, args=(image_path,), daemon=True).start()
    
    def _run_prediction(self, image_path):
        #Run the card grade prediction in a separate thread.
        try:
            predictions = predict_grade(image_path)
            self.current_predictions = predictions
            
            self.root.after(0, lambda: self._update_results(predictions))
            
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
    
    def _update_results(self, predictions):
        #Update the GUI with prediction results.
        self.progress.stop()
        
        if predictions:
            self.result_tree.pack(fill=tk.BOTH, expand=True)
            
            for i, (model_name, grade, confidence) in enumerate(predictions):
                self.result_tree.insert(
                    "", "end", 
                    values=(grade, model_name, f"{confidence:.1f}%"),
                    tags=('top' if i == 0 else '')
                )
            
            self.result_tree.tag_configure('top', background='#e6f3ff')
            
            self.status_var.set("Prediction complete")
            
            self.highlight_check.state(['!disabled'])
            
            if self.highlight_var.get():
                self.highlighted_image = highlight_card_defects(self.current_image_path, predictions)
                if self.highlighted_image is not None:
                    self.display_image(self.highlighted_image)
                    self.status_var.set("Showing defects")
        else:
            self._show_error("Could not predict card grade")
            self.highlight_check.state(['disabled'])
    
    def _show_error(self, message):
        #Display an error message in the GUI.
        self.progress.stop()
        self.status_var.set(f"Error: {message}")
        self.default_result_label.configure(text=f"Error: {message}")
        self.default_result_label.pack(pady=20)
        self.result_tree.pack_forget()
        self.highlight_check.state(['disabled'])

def main():
    root = tk.Tk()
    app = PokemonCardGraderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 