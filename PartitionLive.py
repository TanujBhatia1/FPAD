import os
import shutil
from sklearn.model_selection import train_test_split

def split_and_save_dataset(source_folder, destination_folder, device_type, train_size=0.3, valid_size=0.2, test_size=0.5):
    """
    Split the dataset from source_folder into training, validation, and test sets,
    and save the live data to the appropriate folders in destination_folder.

    :param source_folder: The folder containing the original live fingerprint images.
    :param destination_folder: The base folder to save the split datasets.
    :param device_type: Either 'camera' or 'scanner' to specify the dataset source.
    :param train_size: The proportion of the dataset to include in the train split.
    :param valid_size: The proportion of the dataset to include in the validation split.
    :param test_size: The proportion of the dataset to include in the test split.
    """
    # Ensure the sizes add up to 1
    if not (0 <= train_size <= 1 and 0 <= valid_size <= 1 and 0 <= test_size <= 1):
        raise ValueError("train_size, valid_size, and test_size must be between 0 and 1")

    if train_size + valid_size + test_size != 1.0:
        raise ValueError("train_size, valid_size, and test_size must sum to 1.0")

    # Path to the live fingerprint images in the specific device folder
    live_path = os.path.join(source_folder, device_type)

    live_files = [f for f in os.listdir(live_path) if os.path.isfile(os.path.join(live_path, f))]

    # Split into train and temp (test + valid)
    train_files, temp_files = train_test_split(live_files, test_size=(test_size + valid_size), train_size=train_size, random_state=42)

    # Calculate the adjusted test and validation size proportions
    test_proportion = test_size / (test_size + valid_size)

    # Further split temp into test and valid
    test_files, valid_files = train_test_split(temp_files, test_size=test_proportion, random_state=42)

    # Function to copy files
    def copy_files(file_list, destination):
        for file_name in file_list:
            shutil.copy2(os.path.join(live_path, file_name), os.path.join(destination, file_name))

    # Copy the files to respective directories
    copy_files(train_files, os.path.join(destination_folder, 'train/live'))
    copy_files(test_files, os.path.join(destination_folder, 'test/live'))
    copy_files(valid_files, os.path.join(destination_folder, 'valid/live'))

    # Print the size of each partition
    print(f"--- {device_type.capitalize()} Dataset Partition Sizes ---")
    print(f"Train size: {len(train_files)} images")
    print(f"Test size: {len(test_files)} images")
    print(f"Validation size: {len(valid_files)} images")
    print()

if __name__ == "__main__":
    source_folder = r"C:\Users\tanuj\OneDrive\Desktop\ROI_Fingerprints\ExtractedLiv"  # Replace with the path to your source folder containing 'camera' and 'scanner'
    destination_folder = r"C:\Users\tanuj\OneDrive\Desktop\ROI_Fingerprints\DatasetPartition"  # Ensure this path is correct and the folder structure is already created

    # Split and save the dataset for both 'camera' and 'scanner' subfolders
    for device in ['camera', 'scanner']:
        split_and_save_dataset(source_folder, destination_folder, device, train_size=0.3, valid_size=0.2, test_size=0.5)
    

