import cv2
import numpy as np
import os

def rotate_to_vertical(image):
    """
    Rotate the image to make the fingerprint vertical.
    
    Parameters:
    - image: np.array, input image
    
    Returns:
    - rotated_image: np.array, rotated image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour which is likely to be the fingerprint
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the minimum area rectangle enclosing the largest contour
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]
        
        # Adjust angle for correct rotation
        if angle < -45:
            angle = 90 + angle
        
        # Rotate the image to make the fingerprint vertical
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        return rotated_image
    else:
        print("Error: No contours found for rotation.")
        return image  # Return the original image if no contours are found

def preprocess_image(image_path, output_path, roi_size=(100, 200)):
    """
    Preprocess an image by extracting the ROI around the center and aligning the image.
    
    Parameters:
    - image_path: str, path to the input image
    - output_path: str, path to save the processed image
    - roi_size: tuple, size of the ROI (width, height)
    
    Returns:
    - None
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image from {image_path}")
        return

    # Rotate the image to make the fingerprint vertical
    rotated_image = rotate_to_vertical(image)

    # Convert to grayscale
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter out small contours
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
        
        if not contours:
            print(f"Error: No significant contours found in the image {image_path}")
            return
        
        # Find the largest contour which is likely to be the fingerprint
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Define the ROI around the detected center
        roi_width, roi_height = roi_size
        x_start = max(center_x - roi_width // 2, 0)
        y_start = max(center_y - roi_height // 2, 0)
        x_end = min(center_x + roi_width // 2, rotated_image.shape[1])
        y_end = min(center_y + roi_height // 2, rotated_image.shape[0])
        
        # Ensure the ROI size is exactly 100x200 pixels
        if (x_end - x_start) != roi_width:
            x_end = x_start + roi_width
        if (y_end - y_start) != roi_height:
            y_end = y_start + roi_height
        
        # Crop the ROI from the rotated image
        roi = rotated_image[y_start:y_end, x_start:x_end]
        
        # Save the processed image
        cv2.imwrite(output_path, roi)
        print(f"Processed image saved to {output_path}")
    else:
        print(f"Error: No contours found in the image {image_path}")

# Example usage
input_folder = r'C:\Users\tanuj\OneDrive\Desktop\ROI_Fingerprints\IIITD Fingerphoto dataset\Live Scan'
output_folder = r'C:\Users\tanuj\OneDrive\Desktop\ROI_Fingerprints\IIITD Extracted FingerTip\LivScan'
roi_size = (100, 200)  # Define the desired ROI size

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        preprocess_image(input_path, output_path, roi_size=roi_size)
