# Bird-Recognition - Specie Identification of the Most Common Birds in Portugal

## Project Description
This project aims to develop a deep learning model capable of accurately identifying images of the most common bird species in Portugal. The model will help monitor bird populations, track biodiversity changes, and can be integrated into mobile applications to assist birdwatchers in real-time species identification.

## Key Features
- **Dataset**: Includes images of 11 common bird species in Portugal:
  - *Ciconia ciconia*
  - *Columbia livia*
  - *Streptopelia decaocto*
  - *Emberiza calandra*
  - *Carduelis carduelis*
  - *Serinus serinus*
  - *Delichion urbicum*
  - *Hirundo rustica*
  - *Passer domesticus*
  - *Sturnus unicolor*
  - *Turdus merula*
- **Data Augmentation**: Techniques like rotation, inversion, and cropping will be applied to enhance dataset diversity and model robustness.


## Methods and Algorithms
- **Deep Learning Frameworks**: PyTorch.
- **Pre-trained Models**: EfficientNet-B0 and EfficientNet-V2-s 
- **Dataset Splitting**: Divided into Training (70), Validation (30), and Test (images captured by the group or people who volunteer, simulating an application)

## Evaluation Metrics
- **Quantitative**: Accuracy, Macro F1 score, Top-3 acc, Macro-AUPRC, ROC Curve
- **Qualitative**: Visual inspection of predictions and confusion matrices to identify misclassifications.

## 🧪 Results Summary - Segmented Approach

This section explores two classification strategies for recognizing the 11 most common bird species in Portugal using **EfficientNet**. 
The aim is to develop an image segmentation pipeline, isolating and analyzing birds from a collection of labeled species images. Initially, the goal was to perform fine-grained part-based segmentation, ideally identifying specific anatomical parts such as the head, beak, wings, and tail. This approach would allow for more biologically meaningful feature extraction, potentially improving the accuracy of species classification and enabling part-specific analysis.
However, in practice, detecting individual bird parts reliably proved to be a significant challenge. Even with advanced object detection models, the results were inconsistent due to factors such as pose variation, occlusion, lighting, and the relatively small size of some parts in certain images. Given these limitations, the part-specific detection did not provide robust or generalizable outputs.
As a practical alternative, the solution for a more consistent and automated strategy is to segment each detected bird into three vertical sections (top, middle, and bottom). This compromise allows for a coarse spatial understanding of the bird's structure while maintaining a uniform segmentation procedure across the dataset.

The pipeline works as follows:

- YOLOv8 is used to detect all birds in each image.

- For each bird (class ID 14), a bounding box is passed to SAM (Segment Anything Model) to generate a precise instance mask.

- The bird is isolated from the background using this mask.

- The isolated bird is then divided into three vertical segments, representing rough anatomical regions.

- Each segment is extracted, cropped, and stored as an individual image, labeled with its species ID.

This method ensures that all birds are treated consistently and that each segment is relevant enough to carry discriminative features, even in the absence of explicit part annotations.

Below it's possible to see the dataset distribution:
![dataset_segmented](https://github.com/user-attachments/assets/8679a783-b983-4b84-8f5f-8c8c16e34665)


With this segmentation, 2 strategies were considered:

1. **Multiclass classification** — a single model trained to classify all species.
2. **One-vs-All** — one binary model per species, trained to distinguish one species from all others.

---

### 🐦 Multiclass Classification Validation Results

| Config             | Value   |
|--------------------|---------|
| **learning_rate**  | 0.001   |
| **batch_size**     | 32      |
| **optimizer**      | adam    |
| **weight_decay**   | 0.0001  |
| **dropout_rate**   | 0.5     |
| **model**          | efficientnet-B0  |


| Metric             | Value   |
|--------------------|---------|
| **Macro F1-score** | ~0.74   |
| **Accuracy**       | ~0.75   |
| **Macro-AUPRC**    | ~0.83   |
| **Top-3 Accuracy** | ~0.91   |

![confsuin_matrix_segmented_1](https://github.com/user-attachments/assets/b53103f3-e985-4986-a086-a07362d1bd2c)


---


### 🔁 One-vs-All (Binary) Classifier Validation Results

| Metric             | Average (across 11 models) |
|--------------------|----------------------------|
| **Macro F1-score** |  0.8759                     |
| **Accuracy**       |  0.8760                     |
| **Macro-AUPRC**    |  0.9393                     |

All results in the report

---

### ⚙️ Training Environment

- **GPU**: NVIDIA GeForce RTX 3050 (local setup)
- **Max GPU memory usage**: ~2.0 GB (✅ fits within free-tier constraints)
- **Training time**: ~5 minutes per model (25 epochs)


## References
1. [Merlin Bird ID](https://merlin.allaboutbirds.org/photo-id/)
2. [eBird](https://ebird.org/about/ebird-mobile/)
3. [Wilder - Common Birds in Portugal](https://wilder.pt/primavera/estas-10-aves-estao-entre-as-mais-visas-e-ouvidas-na-primavera)
4. [ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)
5. [EfficientNet](https://pytorch.org/vision/main/models/efficientnet.html)

## Contacts
- Duarte Gonçalves (duarte.dapg@gmail.com)
- João Pereira (joao.pereira197606@gmail.com)


