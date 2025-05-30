from typing import Union, Optional, Any, Dict
import json
from pathlib import Path

# helper function for reading feature information and creating patient vector
def read_feature_information(
    feature_information_path: Union[str, Path], 
    feature_directory: Optional[Union[str, Path]] = None, 
) -> Dict[int, Dict[str, Any]]:
    """
    Reads feature information

    Args:
        feature_information_path:  Path to feature with tile information.
        image_directory:  Path to folder where all images are stored.

    Returns:
        data_dict:  Dictionary with specimen information, feature paths,
            and corresponding tile position information.
    """
    # read feature information from file
    with open(feature_information_path, 'r') as f:
        lines = f.readlines()

    # check if the number of lines is a multiple of three
    if len(lines) % 3 != 0:
        raise ValueError(f'The number of lines in {feature_information_path} '
                         'must be a multiple of three.')

    # initialize dictionary for storing the feature information 
    data_dict = {}
    # loop over feature information per specimen
    for i in range(int(len(lines)/3)):
        image_filenames = eval(lines[i*3])
        specimen_information = json.loads(lines[i*3+1])
        feature_filenames = eval(lines[i*3+2])
        if feature_directory is not None:
            feature_paths = [(Path(feature_directory)/name) for name in feature_filenames]
        else:
            feature_paths = feature_filenames

        # add information to dictionary and indices to list
        specimen_index = specimen_information['specimen_index']
        if specimen_index in data_dict:
            data_dict[specimen_index]['feature_paths'].extend(feature_paths)
        else:
            data_dict[specimen_index] = {
                **specimen_information,
                'images': image_filenames,
                'feature_paths': feature_paths,
            }

    return data_dict