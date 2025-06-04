import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn


class BirdClassifierEnsemble:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.species_list = [
            'Ciconia_ciconia', 'Columba_livia', 'Streptopelia_decaocto',
            'Emberiza_calandra', 'Carduelis_carduelis', 'Serinus_serinus',
            'Delichon_urbicum', 'Hirundo_rustica', 'Passer_domesticus',
            'Sturnus_unicolor', 'Turdus_merula'
        ]

        self.yolo_model = YOLO('runs/detect/train/weights/best.pt')  

        binary_models = [ "saved_models//model_segemented_binary_"+specie for specie in self.species_list]
        
        self.num_classes = len(self.species_list)

        self.multiclass_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        self.multiclass_model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.multiclass_model.classifier[1].in_features, 11)
        )
        self.multiclass_model.load_state_dict(torch.load("saved_models//full_image_model//final_model_20250603.pth")["model_state_dict"])
        self.multiclass_model.eval()

        self.multiclass_model_output_transform = {0:4,
                                            1:0,
                                            2:1,
                                            3:6,
                                            4:3,
                                            5:7,
                                            6:8,
                                            7:5,
                                            8:2,
                                            9:9,
                                            10:10}


        self.head_model = models.efficientnet_b0(weights=None)
        self.head_model.classifier[1] = torch.nn.Linear(self.head_model.classifier[1].in_features, 11)
        self.head_model.load_state_dict(torch.load("saved_models//best_bird_head_model_11classes.pth", map_location='cuda'))
        self.head_model.eval()

        self.body_model = models.efficientnet_b0(weights=None)
        self.body_model.classifier[1] = torch.nn.Linear(self.body_model.classifier[1].in_features, 11)
        self.body_model.load_state_dict(torch.load("saved_models//best_bird_body_model_11classes.pth", map_location='cuda'))
        self.body_model.eval()


        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def _prepare_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict_multiclass(self, model, image_tensor):
        model = model.to(image_tensor.device)
        with torch.no_grad():
            
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
        return probs

    
    def classify(self, image_path, method='vote', top_k=3):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
    
        # Head and Body detection and crop
        results = self.yolo_model(image)
        image_np = np.array(image)
    
        head_crop = None
        body_crop = None
    
        for result in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = result
            part_class = int(cls)
            part_img = image_np[int(y1):int(y2), int(x1):int(x2)]
    
            if part_class == 0:  # 0 = head
                head_crop = part_img
            elif part_class == 1:  # 1 = body
                body_crop = part_img
    
        preds = {}

        
        if head_crop is not None and self.head_model:
            head_tensor = self._prepare_image(Image.fromarray(head_crop)).to(self.device)
            preds['head'] = self.predict_multiclass(self.head_model, head_tensor)
    
        if body_crop is not None and self.body_model:
            body_tensor = self._prepare_image(Image.fromarray(body_crop)).to(self.device)
            preds['body'] = self.predict_multiclass(self.body_model, body_tensor)
    
        if self.multiclass_model:
            img_tensor = self._prepare_image(image).to(self.device)
            preds['multiclass'] = self.predict_multiclass(self.multiclass_model, img_tensor)
            index_mapping = [k for k, _ in sorted(self.multiclass_model_output_transform.items(), key=lambda x: x[1])]
            preds['multiclass'] = preds['multiclass'][:, index_mapping]

        print(preds)
        
        if not preds:
            print("Classification error")
            return None
    
        # Aggregate probabilities using chosen method
        if method == 'vote':
            combined = self._vote_probs(preds)
        elif method == 'mean':
            combined = self._mean_probs(preds)
        elif method == 'max':
            combined = self._max_probs(preds)
        else:
            raise ValueError(f"Unknown method: {method}")

        if top_k:
            # Extract top-k
            top_probs, top_indices = torch.topk(combined, top_k)
            top_classes = [self.species_list[i] for i in top_indices[0].tolist()]
            top_probs = top_probs[0].tolist()
            return list(zip(top_classes, top_probs))
        else:
            return combined
        
    

    def _vote_probs(self, results):
        counts = torch.zeros((1, self.num_classes), device=self.device)
        for probs in results.values():
            pred = torch.argmax(F.pad(probs, (0, self.num_classes - probs.shape[1])))
            counts[0, pred] += 1
        return counts / counts.sum()
    
    def _mean_probs(self, results):
        total = torch.zeros((1, self.num_classes), device=self.device)
        for probs in results.values():
            total += F.pad(probs, (0, self.num_classes - probs.shape[1]))
        return total / len(results)
    
    def _max_probs(self, results):
        stacked = torch.stack([F.pad(p, (0, self.num_classes - p.shape[1])) for p in results.values()])
        return torch.max(stacked, dim=0, keepdim=True).values