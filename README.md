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
- **Models**: Utilizes pre-trained convolutional neural networks such as ResNet and EfficientNet, adapted for bird species identification.
- **Evaluation**: Performance assessed using accuracy, precision, recall, F1 score, and confusion matrices.

## Methods and Algorithms
- **Deep Learning Frameworks**: TensorFlow or PyTorch.
- **Pre-trained Models**: ResNet and EfficientNet will serve as the foundation, fine-tuned for the specific task.
- **Dataset Splitting**: Divided into training, validation, and test sets to ensure robust evaluation.

## Evaluation Metrics
- **Quantitative**: Accuracy, precision, recall, F1 score.
- **Qualitative**: Visual inspection of predictions and confusion matrices to identify misclassifications.

## References
1. [Merlin Bird ID](https://merlin.allaboutbirds.org/photo-id/)
2. [eBird](https://ebird.org/about/ebird-mobile/)
3. [Wilder - Common Birds in Portugal](https://wilder.pt/primavera/estas-10-aves-estao-entre-as-mais-visas-e-ouvidas-na-primavera)
4. [ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)
5. [EfficientNet](https://pytorch.org/vision/main/models/efficientnet.html)

- Duarte Gonçalves (duarte.dapg@gmail.com)

## Affiliation
Informatics Dept., Faculdade de Ciencias da Universidade de Lisboa, Lisbon, Portugal
