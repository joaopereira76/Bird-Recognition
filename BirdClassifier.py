import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image

class BirdClassifierEnsemble:
    def __init__(self):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.species_list = [
            'Ciconia_ciconia', 'Columba_livia', 'Streptopelia_decaocto',
            'Emberiza_calandra', 'Carduelis_carduelis', 'Serinus_serinus',
            'Delichon_urbicum', 'Hirundo_rustica', 'Passer_domesticus',
            'Sturnus_unicolor', 'Turdus_merula'
        ]


        binary_models = [ "saved_models//model_segemented_binary_"+specie for specie in self.species_list]
        
        self.num_classes = len(self.species_list)

        self.multiclass_model = self._load_model(multiclass_model)
        
        self.head_model = self._load_model(head_model)
        self.body_model = self._load_model(body_model)

        

        self.onevsall_segmented_model = self._load_binary_models(binary_models)
        self.segmented_model = self._load_model(multiclass_model)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        if model_path is None:
            return None
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model

    def _load_binary_models(self, models_dict):
        loaded = {}
        if not models_dict:
            return loaded
        for species, path in models_dict.items():
            model = torch.load(path, map_location=self.device)
            model.eval()
            loaded[species] = model
        return loaded

    def _prepare_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict_multiclass(self, model, image_tensor):
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
        return probs

    def predict_binary(self, image_tensor, model_dict):
        probs = []
        with torch.no_grad():
            for species, model in model_dict.items():
                out = model(image_tensor)
                prob = torch.sigmoid(out).item()
                probs.append(prob)
        return torch.tensor(probs).unsqueeze(0).to(self.device)

    def classify(self, image, method='vote'):
        img_tensor = self._prepare_image(image)
        results = {}

        if self.multiclass_model:
            results['multiclass'] = self.predict_multiclass(self.multiclass_model, img_tensor)
        if self.head_model:
            results['head'] = self.predict_multiclass(self.head_model, img_tensor)
        if self.body_model:
            results['body'] = self.predict_multiclass(self.body_model, img_tensor)
        if self.binary_head_models:
            results['binary_head'] = self.predict_binary(img_tensor, self.binary_head_models)
        if self.binary_body_models:
            results['binary_body'] = self.predict_binary(img_tensor, self.binary_body_models)

        if method == 'vote':
            return self._vote(results)
        elif method == 'mean':
            return self._mean(results)
        elif method == 'max':
            return self._max(results)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _vote(self, results):
        votes = torch.zeros(self.num_classes)
        for _, probs in results.items():
            pred = torch.argmax(F.pad(probs, (0, self.num_classes - probs.shape[1])))
            votes[pred] += 1
        final_pred = torch.argmax(votes).item()
        return self.species_list[final_pred]

    def _mean(self, results):
        total = torch.zeros((1, self.num_classes))
        for probs in results.values():
            total += F.pad(probs, (0, self.num_classes - probs.shape[1]))
        final_pred = torch.argmax(total, dim=1).item()
        return self.species_list[final_pred]

    def _max(self, results):
        stacked = torch.stack([F.pad(p, (0, self.num_classes - p.shape[1])) for p in results.values()])
        max_probs, _ = torch.max(stacked, dim=0)
        final_pred = torch.argmax(max_probs, dim=1).item()
        return self.species_list[final_pred]