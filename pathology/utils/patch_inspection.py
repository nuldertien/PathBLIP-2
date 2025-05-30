import os 
from typing import List
import json
import torch
import tqdm
import os.path

def inspection_of_patches(folder_paths: List[str] = 
                          ["./hipt_features/"], 
                          file_ends_with: str ='.pth') -> dict:
    """
    Inspect the number of patches per specimen and save as a dictionary, this 
    is needed to determine how padding should be done because of the different 
    number of patches per specimen. 

    Args:
    folder_path (str): Path to the folder containing the features
    file_ends_with (str): The file extension of the files that should be 
    inspected

    Returns:
    patches_variable (dict): A dictionary containing the number of patches per 
    specimen
    """
    # Recursively search using os.walk to find all paths that are also in directories
    file_paths = []
    for folder_path in folder_paths:
        for root, dirs, files in os.walk(folder_path):
            for name in files:
                path = os.path.join(root, name)
                if path.endswith(file_ends_with):
                    file_paths.append(path)

    # For each path, search amount of patches
    patches_variable = {}
    for file_path in tqdm.tqdm(file_paths):
        specimen_name = file_path.split('/')[-1].split('.')[0]

        # Load the features
        features = torch.load(file_path)
        
        # Try to extract the patches 
        try:
            patches = len(features[1].keys())
            # Save the patches
            patches_variable[specimen_name] = patches
        except:
            print(f"Error with {file_path}")

    return patches_variable

def extract_last_layer_hipt(specimen, special_treatment: bool):
    """
    Before:
    {0: 
        {(specimen, image, x, y): 
            {'feature': [feature_vector (256x384)], 
            'position': [(image, x, y)]}
        },
     1: 
        {(specimen, image, x, y): 
            {'feature': [feature_vector (1x192)], 
            'position': [(image, x, y)]}
        }	
    }

    After:
    {
        (specimen, image_0, x, y): torch.tensor(feature_vector (1x192)),
        ...
        (specimen, image_n, x, y): torch.tensor(feature_vector (1x192))
    }
    """
    new_features = {}
    specimen_last_layer = specimen[1]
    for position, feature in specimen_last_layer.items():
        
        if special_treatment:
            position = list(position)
            position[0] = position[0] + 100000
            position = tuple(position)

        new_features[position] = torch.tensor(feature['feature'])
    
    return new_features    

def find_tensor_files(path: str) -> List[str]:
    """
    Find all tensor files in a directory
    """
    tensor_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".pth"):
                tensor_files.append(os.path.join(root, file))
    return tensor_files

def write_away_last_layer_features(folder_path: str, 
                                   new_folder_path: str) -> None:
    """
    Write away the last layer features of the HIPT2 model
    """
    special_treatment = False
    if folder_path == "./hipt_superbatches_remainder/":
        special_treatment = True

    # Check if the folder exists
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    tensor_files = find_tensor_files(folder_path)
    for tensor_file in tqdm.tqdm(tensor_files):
        
        specimen_file = tensor_file.split("/")[-1]
        specimen_number = int(specimen_file.split(".")[0])
        
        if special_treatment:
            specimen_number += 100000
            specimen_file = str(specimen_number) + ".pth"
        
        if os.path.isfile(new_folder_path + specimen_file):
            continue

        hipt_tensor = torch.load(tensor_file)
        
        last_layer_hipt = extract_last_layer_hipt(hipt_tensor, 
                                                  special_treatment)

        specimen_number = list(last_layer_hipt.keys())[0][0]
        torch.save(last_layer_hipt, 
                   os.path.join(new_folder_path, f'{specimen_number}.pth'))

if __name__ == '__main__':
    # patches_variable = inspection_of_patches()
    # print(patches_variable)

    # with open('./patches_variable.json', 'w') as f:
    #     json.dump(patches_variable, f)
    
    paths = [
        "./extracted_features/",
    ]
    for path in paths:
        write_away_last_layer_features(path, "./hipt_last_layer/")
    
    # paths = [
    #     "./hipt_superbatches/",
    #     "./hipt_superbatches_remainder/"
    # ]
    # for path in paths:
    #     write_away_last_layer_features(path, "./hipt_last_layer/")