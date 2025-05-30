"""
Format for preprocessing data:

List[Dict[str, str]]

[
    {
        "image": "xxxxxx1.pth",
        "caption": str,
        "image_id": int 
    },
    ...
    {
        "image": "xxxxxxn.pth",
        "caption": str,
        "image_id": int
    }
]

ADDITIONAL EXPLANATION:
- image: only the name of the file, not the path. Path is defined in yaml file, 
    within configs/datasets folder (under features-storage).

IDEAS:
- image_id --> patient/case id. (image_id is unique, patient/case id not). 
    However, code should be adjusted to handle this. Currently only works for 
    unique image_id.
"""

from typing import List, Dict, Tuple
import pandas as pd
import json
import os
from utils import read_feature_information
from pathlib import Path

def read_patient_ids(file_name):
    with open(file_name, 'r') as f:
        patient_ids = f.read().split(",")
    return patient_ids

def read_patient_information(file_name: str) -> pd.DataFrame:
    df = pd.read_excel(file_name)
    df['specimen_adjusted'] = df['specimen'].apply(
        lambda x: x[:9] + "_" + x[9:]
        )
    return df
    
def read_report_information(file_name: str) -> Tuple[Dict[str, Dict[str, str]],
                                                      List[str]]:
    with open(file_name, 'r') as f:
        data = json.load(f)

    nested_reports = [list(patient.keys()) for patient in data.values()]
    med_report_ids = [item for sublist in nested_reports for item in sublist]

    return data, med_report_ids

def get_all_report_ids(patient_df: pd.DataFrame, 
                       patient_id: List[str]) -> List[str]:
    filtered_df = patient_df[patient_df['patient'].isin(patient_id)]
    return filtered_df['specimen_adjusted'].tolist()

def check_for_medical_report(med_report_ids: List[str], report_id: str):
    return report_id in med_report_ids

def check_for_image_features(report_id_specimen_map: Dict[str, str], 
                             report_id: str):
    return report_id in report_id_specimen_map.keys()

def find_feature_information_files(path: str) -> List[str]:
    feature_information_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == "feature_information.txt":
                feature_information_files.append(os.path.join(root, file))
    return feature_information_files

def mapping_report_id_to_specimen_id(paths: List[str], 
                                     special_treatment: bool = None
                                     ) -> Dict[str, str]:
    report_id_specimen_map = {}
    images_specimen_map = {}
    for path in paths:
        specimen_information = read_feature_information(path)
        for information in specimen_information.values():
            case_id = information['specimen'][:9] \
                    + "_"\
                    + information['specimen'][9:]
            images_specimen_map[case_id] = information['images']
            if not case_id in report_id_specimen_map.keys():
                if special_treatment:
                    report_id_specimen_map[case_id] = \
                        information['specimen_index'] + 100000    
                else:
                    report_id_specimen_map[case_id] = \
                        information['specimen_index'] 
            else:
                raise ValueError(f"Duplicate case_id: {case_id}")
    return report_id_specimen_map, images_specimen_map

def filter_reports(patient_df: pd.DataFrame, patient_ids: List[str], 
                   med_report_ids: List[str], 
                   report_id_specimen_map: Dict[str, str]) -> List[str]:
    """
    Function to filter out cases that do not have ALL necessary information.
    ALL:    1) Medical report with H&E and IHC+ information
            2) Image features

    Args:
    patient_df (pd.DataFrame): DataFrame containing patient information
    patient_ids (List[str]): List of patient ids to filter (train, val, test)
    med_report_ids (List[str]): List of medical report ids with H&E and IHC+ 
        information
    report_id_specimen_map (Dict[str, str]): Dictionary mapping report_id to 
        specimen_index

    Returns:
    usable_reports (List[str]): List of report_ids that contain all necessary 
        information
    unusable_reports (List[str]): List of report_ids that do not contain all 
        necessary information
    """
    report_ids = get_all_report_ids(patient_df, patient_ids)
    
    unusable_reports = []
    usable_reports = []
    for report_id in report_ids:
        # Check if a medical report has the relevant information (H&E and IHC+)
        # Check if the report has image features
        if check_for_medical_report(med_report_ids, report_id) and \
            check_for_image_features(report_id_specimen_map, report_id):
            usable_reports.append(report_id)
        else:
            unusable_reports.append(report_id)
    return usable_reports, unusable_reports

def create_data_structure(report_ids, report_id_specimen_map, 
                            text_data, text_data_all):
    data = []
    filtered_text_data = {
        med_report_id: text for instance in text_data.values() 
        for med_report_id, text in instance.items()
    }

    filtered_text_data_all = {
        med_report_id: text for instance in text_data_all.values() 
        for med_report_id, text in instance.items()
    }

    for report_id in report_ids:
        specimen_index = report_id_specimen_map[report_id]
        data.append({
            "image": f"{specimen_index}.pth",
            "caption": filtered_text_data[report_id],
            "caption_all": filtered_text_data_all[report_id],
            "image_id": specimen_index
            }
        )
    return data

###############################################################################
################################## VARIABLES ##################################
###############################################################################

BASE_DIR = Path().resolve()
# Path to the patient information
PATH_PATIENT_INFO = BASE_DIR / \
    "./patient_characteristics.xlsx"

# Path to the medical report information
PATH_MEDICAL_REPORT_INFORMATION = BASE_DIR / \
    "./H&E_IHC_plus_preprocessed.json"
PATH_MEDICAL_REPORT_INFORMATION_ALL = BASE_DIR / \
    "./H&E_IHC_plus_report_all_data_preprocessed.json"

# Path to the folder containing the image features
PATH_FEATURE_INFORMATION_1 = BASE_DIR / \
    "./hipt_superbatches/"
PATH_FEATURE_INFORMATION_2 = BASE_DIR / \
    "./hipt_superbatches_remainder/"

# Paths to the different splits
PATH_TRAIN_PATIENT_IDS = BASE_DIR / \
    "./patient_ids_train.txt"
PATH_VAL_PATIENT_IDS = BASE_DIR / \
    "./patient_ids_val.txt"
PATH_TEST_PATIENT_IDS = BASE_DIR / \
    "./patient_ids_test.txt"

# Path to save the data
SAVE_DIR = "./data/pathology/"

if not os.path.exists(BASE_DIR / SAVE_DIR):
    os.makedirs(BASE_DIR / SAVE_DIR)

###############################################################################
###############################################################################
###############################################################################

if __name__ == "__main__":
    # Read in the patient information
    patient_df = read_patient_information(PATH_PATIENT_INFO)

    # Read in the medical report information
    text_data, medical_report_ids = read_report_information(
        PATH_MEDICAL_REPORT_INFORMATION
    )
    text_data_all, medical_report_ids_all = read_report_information(
        PATH_MEDICAL_REPORT_INFORMATION_ALL
    )

    # Read in the patient ids for the different splits
    train_patient_ids = read_patient_ids(PATH_TRAIN_PATIENT_IDS)
    val_patient_ids = read_patient_ids(PATH_VAL_PATIENT_IDS)
    test_patient_ids = read_patient_ids(PATH_TEST_PATIENT_IDS)

    # Create mapping from report_id to specimen_index
    feature_information_files_1 = find_feature_information_files(
        PATH_FEATURE_INFORMATION_1
    )
    report_id_specimen_map_1, images_specimen_map_1 = mapping_report_id_to_specimen_id(
        feature_information_files_1
    )

    feature_information_files_2 = find_feature_information_files(
        PATH_FEATURE_INFORMATION_2
    )
    report_id_specimen_map_2, images_specimen_map_2 = mapping_report_id_to_specimen_id(
        feature_information_files_2,
        special_treatment=True
    )

    report_id_specimen_map = {**report_id_specimen_map_1,
                              **report_id_specimen_map_2}
    
    images_specimen_map = {**images_specimen_map_1,
                            **images_specimen_map_2}

    # Get all report ids for the different splits
    # TRAIN
    train_ids, unusable_train_ids = filter_reports(
        patient_df, train_patient_ids, medical_report_ids, 
        report_id_specimen_map
    )
    train_data = create_data_structure(
        train_ids, report_id_specimen_map, text_data, text_data_all
    )
    with open(BASE_DIR / SAVE_DIR /
              "train_data.json", 'w') as f:
        json.dump(train_data, f, indent=4)

    # VAL
    val_ids, unusable_val_ids = filter_reports(
        patient_df, val_patient_ids, medical_report_ids, 
        report_id_specimen_map
    )
    val_data = create_data_structure(
        val_ids, report_id_specimen_map, text_data, text_data_all
    )
    with open(BASE_DIR / SAVE_DIR /
              "val_data.json", 'w') as f:
        json.dump(val_data, f, indent=4)

    # TEST
    test_ids, unusable_test_ids = filter_reports(
        patient_df, test_patient_ids, medical_report_ids, 
        report_id_specimen_map
    )
    test_data = create_data_structure(
        test_ids, report_id_specimen_map, text_data, text_data_all
    )
    with open(BASE_DIR / SAVE_DIR /
              "test_data.json", 'w') as f:
        json.dump(test_data, f, indent=4)