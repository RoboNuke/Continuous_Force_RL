import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def he_layer_init(layer, bias_const=0.0):
    torch.nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NatureCNN(nn.Module):
    def __init__(self, 
            obs_size,
            with_state=True, 
            with_rgb=False, 
            force_type=None
        ):#, nn.ReLU=nn.ReLU):
        super().__init__()
        with_force = force_type is not None

        # TODO: Handle n-step
        #if "-step" in force_type:
        #    force_type="FFN"

        extractors = {}
        self.out_features = 0
        feature_size = 256
        if with_rgb:
            #TODO: This doesn't work
            """
            in_channels=sample_obs["rgb"].shape[-1]
            image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


            # here we use a NatureCNN architecture to process images, but any architecture is permissble here
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )

            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors["rgb"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size
            """
            pass
        #self.cat_state_force = False
        if with_state and with_force:
            #self.cat_state_force = True
            state_size = obs_size
            if force_type == "FFN":
                force_size = 0
            else:
                raise NotImplementedError(f"Unexpected force type:{force_type}")
            
            extractors['state'] = nn.Sequential(
                nn.Linear(state_size + force_size, 256),
                nn.ReLU()
            )
            self.out_features += 256

        if with_state and not with_force:
            # for state data we simply pass it through a single linear layer
            #extractors["state"] = nn.Sequential(
            
            self.feat_extractor = nn.Sequential(
                nn.Linear(obs_size, 256),
                nn.ReLU()
            )
            self.out_features += 256

        elif with_force and not with_state:
            if force_type == "FFN":
                force_size = obs_size
                extractors["force"] = nn.Sequential(
                    nn.Linear(force_size, 256),
                    nn.ReLU()
                )
                self.out_features += 256
                
            elif force_type == "1D-CNN":
                raise NotImplementedError(f"1D-CNN not implemented yet")
            else:
                raise NotImplementedError(f"Unexpected force type:{force_type}")
            
        #self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        #print(observations)
        #encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        """
        #for key, extractor in self.extractors.items():
        #    obs = observations[key]
            
            TODO: Handle RGB + Force observations seperately
            if self.cat_state_force and (key == 'state'):
                f_obs = torch.tanh( observations['force'] * 0.0011)
                s_obs = observations['state']
                obs = torch.cat([s_obs, f_obs], dim=1) 
                encoded_tensor_list.append(extractor(obs))
            elif not self.cat_state_force:
                if key == "rgb":
                    obs = obs.float().permute(0,3,1,2)
                    obs = obs / 255
                if key == 'force':
                    obs = torch.tanh( obs.float() * 0.0011 )
            
        #    encoded_tensor_list.append(extractor(obs))
        #return torch.cat(encoded_tensor_list, dim=1)
        """
        return self.feat_extractor(observations['states'])