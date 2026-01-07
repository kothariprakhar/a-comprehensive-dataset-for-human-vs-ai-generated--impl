# A Comprehensive Dataset for Human vs. AI Generated Image Detection

Multimodal generative AI systems like Stable Diffusion, DALL-E, and MidJourney have fundamentally changed how synthetic images are created. These tools drive innovation but also enable the spread of misleading content, false information, and manipulated media. As generated images become harder to distinguish from photographs, detecting them has become an urgent priority. To combat this challenge, We release MS COCOAI, a novel dataset for AI generated image detection consisting of 96000 real and synthetic datapoints, built using the MS COCO dataset. To generate synthetic images, we use five generators: Stable Diffusion 3, Stable Diffusion 2.1, SDXL, DALL-E 3, and MidJourney v6. Based on the dataset, we propose two tasks: (1) classifying images as real or generated, and (2) identifying which model produced a given synthetic image. The dataset is available at https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset.

## Implementation Details

# Implementation Explanation: MS COCOAI Detection System

The research paper "A Comprehensive Dataset for Human vs. AI Generated Image Detection" primarily contributes a large-scale dataset (MS COCOAI) to benchmark detection algorithms. Since the paper focuses on the dataset, the core logic required to utilize this research is a **Multi-Task Classification Framework**. 

The implementation provided above creates a deep learning pipeline capable of performing the two key tasks defined in the paper using the described generator classes.

## 1. Architecture: Multi-Head ResNet50

To distinguish between real images and high-quality synthetic images (like those from MidJourney v6 or DALL-E 3), the model requires a robust feature extractor. We employ **Transfer Learning** using a ResNet50 backbone pre-trained on ImageNet.

### The `DeFactifyNet` Class
Instead of training two separate models, we implement a multi-head architecture for efficiency:
1.  **Shared Backbone:** The ResNet50 (minus the final classification layer) extracts high-level visual features. These features capture artifacts common in generative models (e.g., pixel grid irregularities, unnatural textures).
2.  **Head 1 (Binary Classification):** A linear layer projecting features to a single scalar output. This predicts the probability of the image being "Fake" (AI-generated). 
    *   *Math:* $P(y=Fake|x) = \sigma(W_b \cdot f(x) + b_b)$, where $\sigma$ is the sigmoid function.
3.  **Head 2 (Source Attribution):** A linear layer projecting features to 6 output logits (1 Real + 5 Generators).
    *   *Math:* $\hat{y}_{source} = \text{softmax}(W_m \cdot f(x) + b_m)$.

## 2. Dataset Simulation (`MS_COCOAI_Dummy`)

The paper introduces 96,000 datapoints. As we cannot access the physical files here, the `MS_COCOAI_Dummy` class simulates the data structure:
*   **Generators:** It randomly assigns labels corresponding to the 5 generators mentioned in the abstract: Stable Diffusion 3, SD 2.1, SDXL, DALL-E 3, and MidJourney v6.
*   **Tasks Mapping:** 
    *   If the source is Real (Index 0), the binary label is 0.
    *   If the source is any AI generator (Indices 1-5), the binary label is 1.

## 3. Training Logic

The training loop minimizes a joint loss function enabling the model to learn both tasks simultaneously:

$$ L_{total} = L_{binary} + L_{attribution} $$

*   **$L_{binary}$:** Modeled using `BCEWithLogitsLoss` (Binary Cross Entropy). This pushes the model to distinguish between the general distributions of natural vs. synthetic statistics.
*   **$L_{attribution}$:** Modeled using `CrossEntropyLoss`. This pushes the model to learn specific "fingerprints" of individual generative models (e.g., specific frequency artifacts unique to Latent Diffusion Models vs. Transformer-based models like DALL-E).

## 4. Usage in Research

This implementation serves as the baseline evaluation protocol. To replicate the full paper results:
1.  Replace `MS_COCOAI_Dummy` with a loader that reads the actual images from the HuggingFace repository.
2.  The `DeFactifyNet` can be swapped for other architectures (e.g., ViT or Swin Transformer) to benchmark performance differences as typically done in detection papers.

## Verification & Testing

### Analysis of the Implementation

The provided implementation is generally sound and follows standard PyTorch practices for transfer learning, specifically adhering to the architecture described (ResNet50 backbone with a multi-head design for binary and multi-class classification).

**Strengths:**
1.  **Architecture:** The `DeFactifyNet` correctly modifies the ResNet50 backbone by replacing the final fully connected layer with an `Identity` layer and attaching two separate heads. This ensures features are shared while tasks are learned independently.
2.  **Data Handling:** The `MS_COCOAI_Dummy` class and the transform pipeline correctly handle the shape conversion from `(H, W, C)` (numpy default) to `(C, H, W)` (PyTorch default) and normalize inputs according to ImageNet statistics.
3.  **Training Logic:** The handling of losses (combining BCE and CrossEntropy) and dimensions (unsqueezing binary labels) is correct.

**Weaknesses & Potential Issues:**
1.  **Class Imbalance in Dummy Data:** The random generation logic (`random.randint(0, 5)`) results in a dataset where 'Real' images (class 0) appear approx. 16% of the time, while 'AI' images (classes 1-5) appear 83% of the time. While acceptable for a dummy loader, this would lead to severe bias in a real scenario.
2.  **Backbone Freezing Logic:** While implemented correctly, the code does not provide a flag to unfreeze the backbone after X epochs (fine-tuning), which is often crucial for high accuracy in image forensics papers.
3.  **External Dependency in Constructor:** The `DeFactifyNet` initializes `models.resnet50(weights=...)` inside `__init__`. This requires an internet connection to download weights on the first run, which is bad practice for unit testing. It is better to inject the backbone or mock it during tests.
4.  **Reproducibility:** There are no seeds set for `torch`, `numpy`, or `random`, making debugging difficult.

**Verdict:** The code is functional and logically correct for a prototype/demonstration.