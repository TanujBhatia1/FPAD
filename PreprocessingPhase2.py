import cv2
import numpy as np
import os

#Preprocessing phase 2 includes: (Normalization , image enhancement , all images are converted to grayscale)

def normalize_fingerprint(image, target_dpi=500, current_dpi=300):
    """
    Normalize the fingerprint image to the target DPI.
    
    Parameters:
    - image: The input fingerprint image
    - target_dpi: The desired DPI (e.g., 500)
    - current_dpi: The current DPI of the image
    
    Returns:
    - normalized_image: The resized image to match the target DPI
    """
    scale_factor = target_dpi / current_dpi
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    normalized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return normalized_image

def enhance_and_check_quality(image):
    """
    Enhance the fingerprint image and check for quality using a Sobel-based sharpness metric.
    
    Parameters:
    - image: The input fingerprint image
    
    Returns:
    - enhanced_image: The enhanced fingerprint image
    - quality_pass: Boolean indicating whether the image passed the quality check
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)
    
    # Apply Sobel filter to measure sharpness
    sobel_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sharpness = np.mean(sobel_magnitude)
    
    # Set a threshold for sharpness
    sharpness_threshold = 100  # Adjust based on your dataset
    quality_pass = sharpness > sharpness_threshold
    
    return enhanced_image, quality_pass

def process_fingerprint(image_path, output_path):
    """
    Normalize, enhance, and check the quality of a fingerprint image.
    
    Parameters:
    - image_path: str, path to the input image
    - output_path: str, path to save the processed image
    
    Returns:
    - None
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image from {image_path}")
        return
    
    # Step 1: Normalize the fingerprint image
    normalized_image = normalize_fingerprint(image)
    
    # Step 2: Enhance the image and check quality
    enhanced_image, quality_pass = enhance_and_check_quality(normalized_image)
    
    # Step 3: Save the processed image if it passes the quality check
    if quality_pass:
        cv2.imwrite(output_path, enhanced_image)
        print(f"Processed image saved to {output_path}")
    else:
        print(f"Image did not pass the quality check: {image_path}")

# Example usage
input_folder = r'C:\Users\tanuj\OneDrive\Desktop\ROI_Fingerprints\ExtractedLiv\Scanner'  #previous optimized output as input to this step
output_folder = r'C:\Users\tanuj\OneDrive\Desktop\ROI_Fingerprints\ExtractLivFinal\Scanner'   #path to save preprocessing phase 2 images

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_fingerprint(input_path, output_path)
