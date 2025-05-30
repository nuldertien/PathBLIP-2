
import os 
import torch
from collections import OrderedDict

from torch import nn
from lavis.datasets.datasets.base_dataset import BaseDataset

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class PathologyDataset(BaseDataset,  __DisplMixin):
    def __init__(self, vis_processor: str, text_processor: str, vis_root: str, 
            ann_paths: str):
        """
        
        """
        super().__init__(vis_processor=vis_processor,
                            text_processor=text_processor,
                            vis_root=vis_root,
                            ann_paths=ann_paths) 

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids:
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        ann = self.annotation[index]

        # Get image name
        img_name = ann["image"]
        image_path = os.path.join(self.vis_root, img_name)

        # image is of the following format:
        # { 
        #   (specimen_nr, image_nr, x_patch, y_patch): torch.Tensor
        #   ...
        #   (specimen_nr, image_nr, x_patch, y_patch): torch.Tensor
        # }
        # With specimen_nr being the same for each image, the remainder will 
        # change and provide positional information
        image = torch.load(image_path)

        # Get caption
        caption = ann['caption']
        caption_all = ann['caption_all']

        # Process it with the text processor
        caption = self.text_processor(caption)
        caption_all = self.text_processor(caption_all)

        out = {
            "image": image,
            "text_input": caption,
            "text_input_all": caption_all,
            "image_id": ann["image_id"],
        }

        return out
     
    def extract_region_level_representation(self, image_tensor: torch.Tensor):
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
        specimen_last_layer = image_tensor[1]
        for position, feature in specimen_last_layer.items():
            new_features[position] = torch.tensor(feature['feature'])
        return new_features
    
    # def collater(self, samples):
    #     if samples and isinstance(samples[0]["image"], torch.Tensor):
    #         lengths = [sample["image"].shape[0] 
    #                    for sample in samples if sample is not None]
    #         if len(set(lengths)) == 1:
    #             return self.normal_collater(samples)
    #     return self.list_collater(samples)

    # def normal_collater(self, samples):
    #     """
    #     This is used when the number of regions per image is the same for all
    #     images, accomplished by padding missing regions or truncating 
    #     extra regions.
    #     """
    #     # Filter out None samples
    #     samples = [s for s in samples if s is not None]
    #     # Check if samples is empty after filtering
    #     if not samples:
    #         return {}
    #     collated_dict = {}
    #     keys = samples[0].keys() # Use the keys of the first sample as a reference
    #     for k in keys:
    #         values = [sample[k] for sample in samples]
    #         # If the value type for the key is torch.Tensor, stack them else return list
    #         collated_dict[k] = torch.stack(values, dim=0) \
    #             if isinstance(values[0], torch.Tensor) else values
    #     return collated_dict
    
    # def list_collater(self, samples):
    def collater(self, samples):
        """
        This is used for adaptive pooling, where the number of regions per 
        image is not the same for all images.
        """
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return {}
        collated_dict = {}
        keys = samples[0].keys() # Use the keys of the first sample as a reference
        for k in keys:
            values = [sample[k] for sample in samples]
            if k == "image":
                collated_dict[k] = values
            else:
                # If the value type for the key is torch.Tensor, stack them else return list
                collated_dict[k] = torch.stack(values, dim=0) \
                    if isinstance(values[0], torch.Tensor) else values
        return collated_dict

    def extract_patch_level_representation(self, image_tensor: torch.Tensor):
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
            (specimen, image_0, x, y): torch.tensor(feature_vector (256x384)),
            ...
            (specimen, image_n, x, y): torch.tensor(feature_vector (256x384))
        }
        """
        new_features = {}
        specimen_last_layer = image_tensor[0]
        for position, feature in specimen_last_layer.items():
            new_features[position] = torch.tensor(feature['feature'])
        return new_features

class PathologyEvalDataset(PathologyDataset):
    # Exactly the same as PathologyDataset, since no data augmentation is used
    pass
