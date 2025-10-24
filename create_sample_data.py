import os
import random
import shutil

def create_a_sample_tiny_dataset(input_folder_path, output_folder_path, no_of_class = 50,sample_percentage=25):
    """
    Creates a sample tiny dataset from the input folder path (maintains 
    'train'/'val' structure) and saves it to the output folder path.
    """
    # Ensure the output base directory exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Loop through the data splits ("train", "val")
    train_classes = []  
    val_classes = []
    for d in ["train", "val"]:

        # 1. Define the split's input and output paths
        input_split_path = os.path.join(input_folder_path, d)
        output_split_path = os.path.join(output_folder_path, d)
        
        # 2. Ensure the split's output directory exists (e.g., 'sample_data/train')
        if not os.path.exists(output_split_path):
            os.makedirs(output_split_path)
        
        # Loop through each class folder within the split
        for class_name in os.listdir(input_split_path):
            # 3. Define the class input path
            input_class_path = os.path.join(input_split_path, class_name)
            if d == "train":
                if len(train_classes) >= no_of_class:
                    continue
                train_classes.append(class_name)
                
            if d == "val":
                sample_percentage = 100
                if class_name not in train_classes:
                    continue
                else:
                    val_classes.append(class_name)
            # Skip if it's not a directory (e.g., a hidden file)
            if not os.path.isdir(input_class_path):
                continue
                
            # 4. Define the class output path
            output_class_path = os.path.join(output_split_path, class_name)
            
            # 5. Ensure the class output directory exists (e.g., 'sample_data/train/cat')
            if not os.path.exists(output_class_path):
                os.makedirs(output_class_path)

            # Get the list of images, shuffle, and calculate sample size
            images = os.listdir(input_class_path)
            random.shuffle(images)
            print(f"Split: {d}, Class: {class_name}, Total images: {len(images)}")
            
            sample_size = int(len(images) * sample_percentage / 100)
            
            if sample_size == 0 and len(images) > 0:
                 sample_size = 1 # Ensure at least one image is sampled if the class is not empty
            
            print(f"Sampling {sample_size} images ({sample_percentage}%).")

            # Copy the sampled images
            for image_name in images[:sample_size]:
                src_path = os.path.join(input_class_path, image_name)
                dst_path = os.path.join(output_class_path, image_name)
                
                # print(f"Copying: {src_path} to {dst_path}") # Optional: for verbose logging
                shutil.copy2(src_path, dst_path)
    
    print("Dataset sampling complete.")

# Example of how to call the corrected function:
create_a_sample_tiny_dataset(
    input_folder_path ="/home/deep/Documents/jeba/Classification_R_D/res/data", 
    output_folder_path = "./sample_data",
    no_of_class = 100,
    sample_percentage=50
)