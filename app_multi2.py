import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
import gradio as gr
import cv2

# Load your trained model
model = load_model("models/best_model.h5")

# Your class mapping
class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
}

def process_digit_crop(digit_crop):
    """
    Processes a single cropped digit to match the EMNIST 28x28 format.
    """
    # Convert numpy array from OpenCV to PIL Image
    image = Image.fromarray(digit_crop)
    
    # Find bounding box
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
        
    # Resize and pad to 28x28
    new_size = 28
    new_im = Image.new("L", (new_size, new_size), 0) # Black background
    
    padding = 4 # 4 pixels on each side
    target_dim = new_size - (padding * 2)
    
    # Calculate resize ratio
    ratio = min(target_dim / image.width, target_dim / image.height)
    new_w = int(image.width * ratio)
    new_h = int(image.height * ratio)
    
    # Resize with high-quality filter
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Paste the digit into the center of the 28x28 canvas
    paste_x = (new_size - new_w) // 2
    paste_y = (new_size - new_h) // 2
    new_im.paste(image, (paste_x, paste_y))
    
    return new_im


def predict_multi_digit(image):
    """
    Takes a full image, segments digits, and returns the predicted string.
    """
    if image is None:
        return "No image provided"

    # Handle transparency (RGBA) - Common in Sketchpad
    if image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste the image using alpha channel as mask
        background.paste(image, mask=image.split()[3])
        image = background

    # 1. Convert PIL Image to OpenCV format (Numpy array) and Grayscale
    cv_image = np.array(image.convert('L'))

    # 2. Preprocess: Thresholding
    # Sketchpad (Black on White) -> needs inversion to become White on Black
    thresh = cv2.adaptiveThreshold(
        cv_image, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        19, 
        9
    )
    
    # Dilate to connect broken strokes
    kernel = np.ones((3,3), np.uint8)
    dilated_thresh = cv2.dilate(thresh, kernel, iterations=2)

    # 3. Find Contours
    contours, _ = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Filter and Sort Contours
    digit_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out small noise
        if w * h > 50: 
            digit_contours.append((x, y, w, h))

    # Sort contours from left to right
    digit_contours.sort(key=lambda b: b[0])

    # 5. Loop, Process, and Predict
    result_string = ""
    if not digit_contours:
        return "No digits detected."

    for (x, y, w, h) in digit_contours:
        # Crop from thresholded image
        digit_crop = thresh[y:y+h, x:x+w]
        
        # Process to 28x28
        processed_digit_img = process_digit_crop(digit_crop)
        
        # Prepare for model
        img_array = np.array(processed_digit_img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0) # (1, 28, 28)
        img_array = np.expand_dims(img_array, axis=-1) # (1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array, verbose=0)[0]
        digit_label = np.argmax(prediction)
        result_string += class_mapping.get(digit_label, '?')

    return result_string

def handle_sketch(sketch_input):
    """
    Helper to unwrap the dictionary format from Gradio Sketchpad
    """
    if sketch_input is None:
        return "Please draw on the canvas."
    
    # Check if input is a dictionary (Gradio 4.x+)
    final_image = sketch_input
    if isinstance(sketch_input, dict):
        # 'composite' contains the drawing merged with the background
        final_image = sketch_input.get("composite")
        
    return predict_multi_digit(final_image)

# --- Gradio Blocks Interface ---
with gr.Blocks(title="Multi-Digit Recognition") as iface:
    gr.Markdown("# Multi-Digit Recognition System")
    gr.Markdown("Use the tabs below to choose your input method.")

    # Shared Output Display
    output_text = gr.Textbox(label="Result", show_copy_button=True, scale=2)

    with gr.Tabs():
        # --- TAB 1: Upload Image ---
        with gr.TabItem("Upload Image"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Upload Image File")
            with gr.Row():
                upload_btn = gr.Button("Recognize Uploaded Image", variant="primary")
                
            # Bind Upload Button
            upload_btn.click(
                fn=predict_multi_digit,
                inputs=img_input,
                outputs=output_text
            )

        # --- TAB 2: Sketchpad ---
        with gr.TabItem("Write Digits"):
            with gr.Row():
                # Brush radius set slightly larger for better detection
                sketch_input = gr.Sketchpad(type="pil", label="Draw Here", brush=gr.Brush(colors=["#000000"], color_mode="fixed"))
            with gr.Row():
                sketch_btn = gr.Button("Recognize Drawing", variant="primary")
            
            # Bind Sketch Button
            sketch_btn.click(
                fn=handle_sketch,
                inputs=sketch_input,
                outputs=output_text
            )

if __name__ == "__main__":
    iface.launch()