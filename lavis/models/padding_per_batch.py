import torch.nn as nn
from typing import Tuple, List, Dict
import torch

class PaddingPerBatch(nn.Module):
    """
    Module for padding/truncation per batch. It is used within each batch to 
    ensure that all samples have the same number of 'patches'. This is useful
    when the number of patches is different for each sample in the batch. Which
    is the case for the PathologyDataset. Also the attention mask is created
    and returned.
    """
    def __init__(self, max_patches: int=None):
        """
        Args:
        max_patches (int): The maximum length of the sequence. If None, the 
            maximum length is calculated from the batch.
        """
        super(PaddingPerBatch, self).__init__()
        self.max_patches = max_patches
        self.num_features = 192    

    def forward(self, 
                x: List[Dict[Tuple[int, int, int, int], torch.Tensor]]) \
                                        -> Tuple[torch.Tensor, torch.Tensor]:
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
        embedding should be the same for all images. 
        """
        image_features = []

        for image in x:    
            # Stack the features
            stacked_features = torch.stack(list(image.values()))
            
            # Now it is transformed from (n_patches, 1, 192) to (n_patches, 192)
            image_feature = nn.Flatten(start_dim=1)(stacked_features)
            image_features.append(image_feature)

        if self.max_patches is None:
            max_patches = max([image.size(0) for image in image_features])
        else:
            # Currently this is neglected because otherwise r@k is not working
            # max_patches = min(self.max_patches, 
            #                  max([image.size(0) for image in image_features]))
            max_patches = self.max_patches
            
        padded_images = []
        attention_masks = []

        for image in image_features:
            pad_size = max_patches - image.size(0)

            if pad_size < 0:
                image = self.randomized_truncation(image, -pad_size)
                pad_size = 0

            padding = torch.full(
                (pad_size, *image.size()[1:]), 
                0, # 0 padding
                device=image.device  
            ) 
            padded_image = torch.cat([image, padding], dim=0)

            attention_mask = torch.cat(
                [torch.ones(image.size(0), device=image.device),
                 torch.zeros(pad_size, device=image.device)],
                dim=0
            )

            padded_images.append(padded_image)
            attention_masks.append(attention_mask)

        return torch.stack(padded_images), torch.stack(attention_masks)

    def randomized_truncation(self, image, truncate_size):
        random_indices = torch.randperm(image.size(0))[truncate_size:]
        # Put the random indices in ascending order
        random_indices = random_indices.sort()[0]
        return image[random_indices]
    
def create_padding_per_batch(max_patches: int=None) -> PaddingPerBatch:
    return PaddingPerBatch(max_patches=max_patches)

if __name__ == "__main__":
    random_length = torch.randint(1, 10, (1,)).item()
    print("Random Length:", random_length)
    model = create_padding_per_batch(max_patches=random_length)
    
    example_input = [
        # Creating dictionaries where each tuple key represents a unique patch identifier
        {
            (1, 0, 0, 0): torch.tensor([0] * 192, dtype=torch.float32),
            (1, 0, 0, 1): torch.tensor([1] * 192, dtype=torch.float32),
            (1, 0, 1, 0): torch.tensor([2] * 192, dtype=torch.float32)
        },
        {
            (2, 1, 0, 0): torch.rand(192),
            (2, 1, 0, 1): torch.rand(192),
            (2, 1, 1, 0): torch.rand(192),
            (2, 1, 1, 1): torch.rand(192),
            (2, 1, 1, 2): torch.rand(192),
        },
        {
            (3, 2, 0, 0): torch.rand(192),
            (3, 2, 0, 1): torch.rand(192)
        }
    ]

    padded_images, attention_masks = model(example_input)
    print("Padded Images Shape:", padded_images.shape)
    print("Attention Masks Shape:", attention_masks.shape)
    print("Attention Masks:")
    print(attention_masks)
    print("Padded Images:")
    print(padded_images)
