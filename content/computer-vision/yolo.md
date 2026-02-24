---
title: You Only Look Once
---

$$
    \newcommand{\loss}{\mathcal{L}}
    \newcommand{\one}{\mathbf{1}}
$$

## Introduction to the architecture

Before YOLO, models like R-CNN used a two-stage approach: first proposing potential regions where objects might be, and then classifying those regions. YOLO changed the game by doing it all at once using a single Convolutional Neural Network (CNN). The main concepts behind YOLO:

- First the input image is divided into smaller regions of size $S \times S$.

- If the center of an object lands inside a specific grid cell, it is responsible for detecting it.

- Each cell predicts $B$ bounding boxes and confidence scores for these bouding boxes.

For a given grid cell, the network outputs a tensor containing the bounding box attributes and class probabilities. If you are predicting $C$ classes and $B$ bounding boxes per cell, the output dimension for each cell is $B \times 5 + C$. The values in each bounding box are:

- $x, y$: coordinates relative the the center of the bounding box.

- $w, h$: width and height of each bounding box relative the the whole image.

- $C_\text{conf}$: the confidence score (the probability that an object exists multiplied by the Intersection over Union (IoU) between the predicted box and the ground truth).

## Math

To understand the internal processes, we need to look at how the model decodes bounding boxes and how it calculates its loss during training.

### Bounding Box Prediction

Starting with YOLOv2, the model predicts bounding box coordinates using anchor boxes (predefined shapes). The network outputs parameterized coordinates ($t_x, t_y, t_w, t_h$), which are transformed into bounding box coordinates ($b_x, b_y, b_w, b_h$) using the following equations:

$$
    \begin{split}
        b_x &= \sigma(t_x) + c_x, \;\; b_y = \sigma(t_y) + c_y, \\
        b_w &= p_w e^{t_w}, \;\;\; b_h = p_h e^{t_h},
    \end{split}
$$

where:

- $\sigma$ is the sigmoid function assuring normalization to the interval $[-1, 1]$.

- $c_x, c_y$ are the top-left coordinates of the current grid cell.

- $p_w, p_h$ are the width and height of the predefined anchor box.

### Original Loss Function

The YOLO training process optimizes a massive, multi-part loss function. It computes the Sum of Squared Errors (SSE) across localization, objectness, and classification.

$$
    \begin{split}
        \loss_\text{loc} &= \lambda_\text{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^\text{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\
        \loss_\text{obj} &= \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^\text{obj} (C_i - \hat{C}_i)^2 + \lambda_\text{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^\text{noobj} (C_i - \hat{C}_i)^2 \\
        \loss_\text{class} &= \sum_{i=0}^{S^2} \mathbb{1}_{i}^\text{obj} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2 \\
        \loss_{total} &= \loss_\text{loc} + \loss_\text{obj} + \loss_\text{class}
    \end{split}
$$

> [!info]+ Details of the loss equations
>
> - $\mathbb{1}_{ij}^\text{obj}$ is a binary mask that is 1 if the $j$-th bounding box in the $i$-th cell is responsible for detecting the object, and 0 otherwise.
> - The square root of width/height is used so that small deviations in small boxes are penalized more heavily than the same deviations in large boxes.

### Training

Forward Pass: The image is passed through the backbone (feature extractor) and the neck (feature aggregator), finally reaching the head where the grid predictions are made.

Target Assignment: The algorithm matches ground-truth objects to the specific grid cells and anchor boxes that have the highest IoU (Intersection over Union).

Loss Calculation: The model evaluates how far its predictions were from the ground truth using the loss functions (modern YOLOs use variations like CIoU loss for bounding boxes and Focal Loss for classification).

Backpropagation: The weights are updated using optimizers (like AdamW or, in YOLO26, MuSGD).

Post-Processing (Historically): Older YOLO models predicted thousands of boxes. They used Non-Maximum Suppression (NMS) to filter out overlapping boxes, keeping only the one with the highest confidence. Newer versions have engineered this step out entirely.

## Evolution of SOTA


