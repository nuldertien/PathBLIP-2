import torch.nn as nn
import torch

class IdentityVisionEncoder(nn.Module):
    """
    A vision encoder that returns exactly the input tensor. Therefore actually 
    it is a network that does nothing, but has the same functions as the other
    vision encoders such that it is compatible with the rest of the code.
    """
    def __init__(self, num_features: int=192):
        """
        num_features (int): 
            e.g. img [3, 224, 224] -> vision encoder -> [246, 768] -> num_features = 768
        """
        super(IdentityVisionEncoder, self).__init__()
        self.encoder = nn.Identity()
        self.num_features = num_features
               
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.encoder(x)
        elif isinstance(x, list):
            image_features = []
            for image in x:
                    
                # Stack the features
                stacked_features = torch.stack(list(image.values()))
                
                # Now it is transformed from (n_patches, 1, 192) to (n_patches, 192)
                image_feature = nn.Flatten(start_dim=1)(stacked_features)
                image_features.append(image_feature)
            return self.encoder(torch.stack(image_features))
        else:
            raise ValueError (f"Input type {type(x)} is not supported.")            

    def get_num_layer(self, var_name=""):
        return 0

def create_identity_model():
    #TODO, do we want to have a variable for precision, fp16
    return IdentityVisionEncoder()

if __name__ == "__main__":
    model = create_identity_model()