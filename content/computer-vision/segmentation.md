---
title: Segmentation
---

## Introduction to Segmentation

![intro](computer-vision/res-segmentation/intro.png)

In digital image processing and computer vision, image segmentation is the process of partitioning a digital image into multiple image segments, also known as image regions or image objects (sets of pixels). The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. Image segmentation is typically used to locate objects and boundaries (lines, curves, etc.) in images. More precisely, image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics.

## History of Segmentation

Before 2015, segmentation models relied on patch-by-patch classification, which was incredibly slow.

**FCN (2015)** Fully Convolutional Networks (FCNs) completely changed the paradigm. FCNs proved that you could train neural networks to make dense, pixel-wise predictions end-to-end. It replaced the dense (fully connected) layers of standard classification networks (like VGG16) with 1x1 convolutions, allowing the network to output spatial heatmaps instead of single class probabilities. It uses a standard CNN backbone to downsample the image and extract features (encoder), followed by transpose convolutions (often called deconvolutions) to upsample the feature maps back to the original image size.

**U-Net (2015)** Developed initially for biomedical image segmentation, U-Net remains one of the most famous and widely adapted architectures today. FCNs lost a lot of fine spatial details during downsampling. U-Net solved this by passing high-resolution feature maps from the encoder directly to the decoder. A symmetric, U-shaped network. The "contracting path" (encoder) captures context through convolutions and pooling. The "expansive path" (decoder) uses up-convolutions. Skip connections concatenate the encoder's feature maps with the decoder's upsampled maps, fusing "what" (semantic context) with "where" (spatial precision).

**Expanding the Receptive Field: DeepLab Series (2014–2018)** The DeepLab models (culminating in DeepLabv3+) tackled the issue of segmenting objects at multiple scales. Atrous (Dilated) Convolutions and Atrous Spatial Pyramid Pooling (ASPP). Pooling layers reduce image resolution drastically. Dilated convolutions allowed the network to "widen its view" (receptive field) without losing resolution or increasing computational cost.

**The Instance Segmentation King: Mask R-CNN (2017)** While previous models did semantic segmentation (labeling all "car" pixels as one blob), Mask R-CNN brought SOTA instance segmentation (distinguishing "car 1" from "car 2"). It extended the Faster R-CNN object detection model by adding a parallel branch that outputs a binary mask for each detected bounding box. It also introduced RoIAlign, a quantization-free layer that preserves exact spatial locations (crucial for pixel-perfect masks). A ResNet backbone + Feature Pyramid Network (FPN) extracts features. A Region Proposal Network (RPN) suggests bounding boxes. RoIAlign extracts fixed-size feature maps for each box. The network then splits into three heads: classification, bounding box regression, and mask prediction.

**The Vision Transformer Era: SegFormer & Mask2Former (2021–2022)** Transformers, initially built for natural language processing, were adapted for vision (ViT). Global context via self-attention. CNNs are biased toward local features (the convolution window). Transformers look at the relationship between all patches of an image simultaneously. Mask2Former unified semantic, instance, and panoptic segmentation into a single architecture based on mask classification rather than pixel classification. A hierarchically structured Transformer encoder that outputs multi-scale features (without positional encoding, making it robust to different resolutions), paired with a lightweight Multi-Layer Perceptron (MLP) decoder.

**The Current SOTA Foundation Models: SAM & SAM 2 (2023–Present)** Meta's Segment Anything Model (SAM) shifted segmentation from highly specific, dataset-bound tasks to a "foundation model" paradigm. You can prompt SAM with a click (point), a bounding box, or text, and it will segment the object perfectly, even if it has never seen that specific class of object before. SAM 2 (2024) extended this to real-time video segmentation.
