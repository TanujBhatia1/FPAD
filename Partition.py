import os
import random
import shutil
from sklearn.model_selection import train_test_split

def create_partition(input_folder, output_folder, train_size=0.3, valid_size=0.2, test_size=0.5):
    """
    Partitions the dataset into train, validation, and test sets based on PAI species.

    Parameters:
    - input_folder: str, path to the folder containing the dataset images.
    - output_folder: str, path to the folder where the partitions will be saved.
    - train_size: float, proportion of the dataset to include in the train set.
    - valid_size: float, proportion of the dataset to include in the validation set.
    - test_size: float, proportion of the dataset to include in the test set.

    Returns:
    - None
    """
    # Ensure output directories exist
    train_dir = os.path.join(output_folder, 'train')
    valid_dir = os.path.join(output_folder, 'valid')
    test_dir = os.path.join(output_folder, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Define PAI species groups
    pai_groups = {
        'printout': [],
        'transparent': [],
        'default_color': [],
        'colored_silicone': []
    }

    # Categorize images into PAI species groups based on filenames
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Categorize based on the PAI species within the filename
            if any(keyword in filename for keyword in ['printout_']):
                pai_groups['printout'].append(filename)
            elif any(keyword in filename for keyword in ['ds_original', 'ds_brown_transparent','gelafix_', 'gelatin_fx_' , 'schoolglue_']):
                pai_groups['transparent'].append(filename)
            elif any(keyword in filename for keyword in ['ds_', 'ef_']):
                pai_groups['colored_silicone'].append(filename)
            else:
                pai_groups['default_color'].append(filename)

    # Initialize sets for each partition
    train_set, valid_set, test_set = [], [], []

    # Split each PAI group into train, validation, and test sets
    for group, images in pai_groups.items():
        train_images, temp_images = train_test_split(images, test_size=(valid_size + test_size))
        valid_images, test_images = train_test_split(temp_images, test_size=(test_size / (valid_size + test_size)))

        train_set.extend(train_images)
        valid_set.extend(valid_images)
        test_set.extend(test_images)

    # Move files to respective directories
    for filename in train_set:
        shutil.copy(os.path.join(input_folder, filename), os.path.join(train_dir, filename))
    for filename in valid_set:
        shutil.copy(os.path.join(input_folder, filename), os.path.join(valid_dir, filename))
    for filename in test_set:
        shutil.copy(os.path.join(input_folder, filename), os.path.join(test_dir, filename))

    # Output the results
    print(f"Dataset split complete: {len(train_set)} train, {len(valid_set)} valid, {len(test_set)} test.")

if __name__ == "__main__":
    # Example usage
    input_folder = r'C:\Users\tanuj\OneDrive\Desktop\ROI_Fingerprints\OutputOptimized'  # Preprocessed dataset as the input
    output_folder = r'C:\Users\tanuj\OneDrive\Desktop\ROI_Fingerprints\FakePartitionOutput'
    
    # Ensure to update input and output paths
    create_partition(input_folder, output_folder)
