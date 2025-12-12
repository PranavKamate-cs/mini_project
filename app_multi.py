import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
import gradio as gr
import cv2 # Import OpenCV

# Load your trained model
model = load_model("models/best_model.h5")

# Your class mapping (though for 0-9, the index is the label)
class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
}

def process_digit_crop(digit_crop):
    """
    Processes a single cropped digit to match the EMNIST 28x28 format.
    This is adapted from your app1.py's logic.
    """
    # Convert numpy array from OpenCV to PIL Image
    image = Image.fromarray(digit_crop)
    
    # We don't need to invert if we use THRESH_BINARY_INV, 
    # but if the crop is black-on-white, we would invert.
    # image = ImageOps.invert(image) 

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
    Takes a full image with multiple digits, segments them,
    and returns the predicted string.
    """
    if image is None:
        return ""

    # 1. Convert PIL Image (from Gradio) to OpenCV format (Numpy array)
    # Convert to grayscale at the same time
    cv_image = np.array(image.convert('L'))

    # 2. Preprocess: Thresholding
    # This turns the image into pure black and white.
    # THRESH_BINARY_INV makes the digits white (255) and background black (0).
    # This is what findContours expects.
    _, thresh = cv2.threshold(cv_image, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    dilated_thresh = cv2.dilate(thresh, kernel, iterations=2)
    # 3. Find Contours
    # RETR_EXTERNAL gets only the outer-most contours (ignores holes)
    # CHAIN_APPROX_SIMPLE saves memory by only storing end points of lines
    contours, _ = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Filter and Sort Contours
    digit_contours = []
    for c in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(c)
        
        # Filter out small noise
        if w * h > 50: # You can adjust this "area" threshold
            digit_contours.append((x, y, w, h))

    # **CRITICAL STEP: Sort contours from left to right**
    # We sort by the 'x' coordinate (the first element of our tuple)
    digit_contours.sort(key=lambda b: b[0])

    # 5. Loop, Process, and Predict
    result_string = ""
    for (x, y, w, h) in digit_contours:
        # "Cut" the digit from the thresholded image
        digit_crop = thresh[y:y+h, x:x+w]
        
        # Process the crop to be 28x28
        processed_digit_img = process_digit_crop(digit_crop)
        
        # Convert to numpy array for the model
        img_array = np.array(processed_digit_img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0) # (1, 28, 28)
        img_array = np.expand_dims(img_array, axis=-1) # (1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array, verbose=0)[0]
        
        # Get the digit with the highest probability
        digit_label = np.argmax(prediction)
        result_string += class_mapping.get(digit_label, '?')

    return result_string

# --- Set up the new Gradio Interface ---
iface = gr.Interface(
    fn=predict_multi_digit,
    # Use gr.Image for uploading, not gr.Sketchpad
    inputs=gr.Image(type="pil", label="Upload an Image with Digits"), 
    # Use gr.Textbox for the final string output
    outputs=gr.Textbox(label="Recognized Digits"),
    live=True,
    title="Multi-Digit Recognition",
    description="Upload an image with several digits (e.g., '173') written on it."
)

if __name__ == "__main__":
    iface.launch()