import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image


class LegibilityClassifier34(nn.Module):
    """ ResNet34 based model for binary classification """

    def __init__(self, train=False, finetune=False):
        super().__init__()
        self.model_ft = models.resnet34(pretrained=True)
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)
        self.model_ft.fc.requires_grad = True
        self.model_ft.layer4.requires_grad = True

    def forward(self, x):
        x = self.model_ft(x)
        x = F.sigmoid(x)
        return x
    

class LegibilityPredictor:
    """
    Wrapper for legibility classification inference.
    Based on mkoshkina/jersey-number-pipeline implementation.
    """
    
    def __init__(self, model_path, device='cuda', threshold=0.5, arch='resnet34'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.arch = arch

        self.model = LegibilityClassifier34()
       
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("Loaded checkpoint")

        except RuntimeError as e:
            print(f"Load failed: {e}")
            
            new_state_dict = {}
            model_keys = set(self.model.state_dict().keys())
            
            for k, v in state_dict.items():
                candidates = [
                    k,
                    k.replace('model.', ''),
                    k.replace('module.', ''),
                    'model_ft.' + k,
                    'model_ft.' + k.replace('model.', ''),
                    'model_ft.' + k.replace('module.', ''),
                ]
                
                for candidate in candidates:
                    if candidate in model_keys:
                        new_state_dict[candidate] = v
                        break
            
            loaded_keys = set(new_state_dict.keys())
            missing = model_keys - loaded_keys
            unexpected = loaded_keys - model_keys
            
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
            
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded {len(new_state_dict)}/{len(model_keys)} keys")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Legibility classifier loaded, threshold: {self.threshold}")
    
    
    def filter(self, crops_list, indices):
        """ Filter crops by legibility """

        if not crops_list or not crops_list[0]:
            return crops_list, indices
        
        legible_flags, _ = self.predict_batch(crops_list[0])
        
        filtered_crops_list = [
            [c for c, keep in zip(crops, legible_flags) if keep]
            for crops in crops_list
        ]
        filtered_indices = [i for i, keep in zip(indices, legible_flags) if keep]
        
        return filtered_crops_list, filtered_indices
    

    def predict_batch(self, crops, batch_size=32):
        legible_flags = []
        confidences = []
        
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            batch_flags, batch_confs = self.process_batch(batch_crops)
            legible_flags.extend(batch_flags)
            confidences.extend(batch_confs)
        
        return legible_flags, confidences
    

    def process_batch(self, crops):
        """ Process a single batch through the model """

        batch_tensors = []
        for crop in crops:
            pil_img = Image.fromarray(crop)
            tensor = self.transform(pil_img)
            batch_tensors.append(tensor)
        
        batch = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch).squeeze()
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
        
        confidences = outputs.cpu().numpy()
        
        legible_flags = confidences >= self.threshold
        
        return legible_flags.tolist(), confidences.tolist()