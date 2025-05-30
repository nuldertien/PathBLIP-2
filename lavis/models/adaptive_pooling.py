import torch.nn as nn
from typing import Tuple
import torch

class AdaptivePool(nn.Module):
    """
    Model for adaptive naive pooling. It can be used for either max or average
    pooling.
    """
    def __init__(self, max_patches: int=None, technique: str="max"):
        """
        Args:
        max_patches (int): Maximum amount of patches, if None max per batch is 
            assumed
        technique (str): The technique to use for pooling. Either:
            "max" or "avg"
        """
        super(AdaptivePool, self).__init__()
        self.max_patches = max_patches
        self.technique = technique
        self.num_features = 192
        
    def forward(self, x):
        """
        x is a list, with following format:
        [
            {
            (specimen_nr1, image_nr, x_patch, y_patch): torch.Tensor
            ...
            (specimen_nr1, image_nr, x_patch, y_patch): torch.Tensor
            },
            ...
            {
            (specimen_nrN, image_nr, x_patch, y_patch): torch.Tensor
            ...
            (specimen_nrN, image_nr, x_patch, y_patch): torch.Tensor
            }
        ]

        The number of patches can be different for each image, but the feature
        embedding should be the same for all images. The pooling is applied to
        the patches of each image separately. 
        """
        image_features = []

        for image in x:    
            # Stack the features
            stacked_features = torch.stack(list(image.values()))

            # Now it is transformed from (n_patches, 1, 192) to (n_patches, 192)
            image_feature = nn.Flatten(start_dim=1)(stacked_features)
            image_features.append(image_feature)
        
        if not self.max_patches:
            max_patches = max([image.size(0) for image in image_features])
            self.max_patches = max_patches

        if self.technique == "max":
            self.aggregator = nn.AdaptiveMaxPool1d(self.max_patches)
        elif self.technique == "avg":
            self.aggregator = nn.AdaptiveAvgPool1d(self.max_patches)
        else:
            raise ValueError(f"Technique {self.technique} is not supported.") 
        
        batch_list = []

        for image in image_features: 
            # Reshape:
            # n_patches, features
            # to
            # 1, n_patches, features
            image = image.unsqueeze(0)
            
            # Reshape:
            # 1, n_patches, features
            # to
            # 1, features, n_patches
            image = image.permute(0, 2, 1)

            # Apply the pooling
            aggregated_features = self.aggregator(image)

            # Reshape:
            # 1, features, max_patches
            # to
            # 1, max_patches, features
            aggregated_features = aggregated_features.permute(0, 2, 1)

            # append to the batch list
            batch_list.append(aggregated_features)
        
        # Stack to a batch tensor
        batch_aggregated_features = torch.cat(batch_list, dim=0)

        return batch_aggregated_features
    
    def get_num_layer(self):
        """
        Since this is a non-learnable module, it does not have any layers. In 
        essence it is a fixed transformation and therefore 0 layers.
        """
        return 0
    
def create_adaptive_pooling(max_patches: int=None, 
                             technique: str="max"):
    #TODO, do we want to have a variable for precision, fp16
    return AdaptivePool(max_patches=max_patches,
                        technique=technique)

if __name__ == "__main__":
    model = create_adaptive_pooling()