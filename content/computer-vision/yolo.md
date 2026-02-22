---
title: You Only Look Once
---

$$
    \newcommand{\loss}{\mathcal{L}}
    \newcommand{\one}{\mathbf{1}}
$$

## Introduction to YOLO architecture

Before YOLO, models like R-CNN used a two-stage approach: first proposing potential regions where objects might be, and then classifying those regions. YOLO changed the game by doing it all at once using a single Convolutional Neural Network (CNN). The main concepts behind YOLO:

- First the input image is divided into smaller regions of size $S \times S$.

- If the center of an object lands inside a specific grid cell, it is responsible for detecting it.

- Each cell predicts $B$ bounding boxes and confidence scores for these bouding boxes.

For a given grid cell, the network outputs a tensor containing the bounding box attributes and class probabilities. If you are predicting $C$ classes and $B$ bounding boxes per cell, the output dimension for each cell is $B \times 5 + C$. The values in each bounding box are:

- $x, y$: coordinates relative the the center of the bounding box.

- $w, h$: width and height of each bounding box relative the the whole image.

- $C_\text{conf}$: the confidence score (the probability that an object exists multiplied by the Intersection over Union (IoU) between the predicted box and the ground truth).

## Math behind YOLO

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
        \loss_\text{loc} &= \lambda_\text{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \one_{ij}^\text{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\
        \loss_\text{obj} &=
    \end{split}
$$

hejka
