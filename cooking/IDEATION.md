# Time to Cook
## General Ideas
**Committing to ConvNeXt backbone!!** 

Will be taking the shared encoder approach for now for the following reasons:
- Limited time to train
- Small dataset, pretraining and squeezing as much as I can from data crucial

In Class, MIM was described as the "best" technique for SSL.

## Pre-Training Ideas
**Some training pipeline thoughts!** FCMAE on 50k unlabeled + 7.5k classification labeled WITHOUT labels to treat it as unlabeled + 3k semantic segmentation masked WITHOUT masks to treat it as unlabeled
    - =60.5k images...?

**From Gemini 3.1 Pro:**
> **Approach 1: Pretraining Only on the 50k Unlabeled Samples:** The model learns entirely from the 50k pool and is later fine-tuned on the separate 7.5k (classification) and 3k (segmentation) datasets.
>
> **Domain Shift (Distribution Mismatch):** This is the biggest risk. If your 50k unlabeled images were scraped from a different source or collected under different conditions than your labeled data, the MAE will learn a representation space that doesn't align with your downstream tasks. When you move to fine-tuning, the model will struggle to adapt.
>
> **Underutilization of Data:** MAEs scale exceptionally well with data volume. By completely walling off 10.5k high-quality images during the pretraining phase, you are leaving roughly 17% of your visual diversity on the table.
>
> **Overfitting During Fine-Tuning:** With 300 target classes, 7.5k labeled images leaves you with an average of only 25 images per class (and only 10 per class for segmentation). If the pretrained model hasn't learned robust, universal features from the 50k set, it will rapidly overfit to these tiny labeled subsets during fine-tuning.
>
> **Approach 2: Pretraining on All 60.5k Samples (Ignoring Labels)** You pool everything together for the MAE pretext task, maximizing your pretraining data, and then fine-tune on the labeled subsets.
>
> **Memorization vs. Generalization:** If the model sees the exact same images in pretraining that it will later be fine-tuned on, it risks memorizing specific image patches rather than learning generalized, semantic features. When fine-tuned, the model is essentially acting on "confirmation bias"—relying on memorized textures of those specific images rather than learning the underlying features of the 300 classes.
> 
> **Class Imbalance Amplification:** If the 10.5k labeled images are perfectly balanced across your 300 classes, but the 50k unlabeled images are heavily skewed towards just 10 classes, combining them means your pretraining distribution is heavily skewed. The MAE will become excellent at reconstructing the majority classes while remaining blind to the long-tail minority classes.
>
> **The Verdict: In modern Self-Supervised Learning (SSL), Approach 2 is generally the standard and most effective strategy, but with one absolute condition: You must rigorously separate your validation and test splits first. You should pool the 50k unlabeled images with only the training splits of your 7.5k and 3k datasets. This maximizes your data volume for the MAE to learn good patch reconstructions without leaking your evaluation data into the model's memory.**

앗싸, FCMAE 쓰자. 

## Fine-Tuning Ideas
Directly running end-to-end task on both respective architectures is likely the best bet. Use the 7.5k labeled to train up the classifier model, and use the 3k to train up the semantic segmentation model.

Risk is of overfit/inability to gneralize with such few samples, but later this can be addressed with pesudolabeling on the untrained dataset.

## Model Architecture Ideas
### ConvNeXt Backbone

### UPerNet Decoder for Semantic Segmentation


## High Level Project Architecture Ideas
### 

## Implementation Ideas
### Pytorch my beloved


## Roadmap
**Mission Critical: MUST COMPLETE ASAP**
1. Implement ConvNeXt backbone
    - 
2. Implement FCMAE pre-training
    - 
3. Pre-train the backbone with simple decoder
    - Potentially send to Google Cloud VM?
4. 