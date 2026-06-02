# Notes on ConvNeXt
## Goals
- Get a firm grasp on the model architecture
    - Be able to recreate in PyTorch!!!
    - What features does it have?
    - How conducive is it to downstream goals?
- Establish how to accomplish both tasks
    - Classification task
    - Semantic Segmentation task
- Estabilshed if Semi and Self supervised learning are possible
    - Masked Autoencoder
    - If not, then other Self supervised learning methods
- Other metrics
    - Feasibility
    - Complexity
    - Train time

## Paper 1 - "A ConvNet for the 2020s"
*https://arxiv.org/pdf/2201.03545*
### Abstract
- ViT introduction took over from ConvNet
- But Vanilla ViT struggled with computer vision tasks
- How did it take over? -> Hierarchial Transformers (e.g. Swin)
- Reintroduced some of the assumptions that made ConvNets do so well
- ViT being good is the intrinsic superiority of Transformers
- ConvNeXt is modernized ConvNet - starting from ResNet

### Introduction
- ConvNet Background
    - 2010s brought huge advancements in DL, mainly through ConvNets
    - AlexNet -> ImageNet age
    - Key was "sliding window" strategy to visual processing
        - Especially for high-resolution images
    - ConvNets have built-in inductive biases for CV applications
        - Translation Invariance
    - Shared computations in ConvNets - inherently efficient
    - Region-based detectors further made CNN better
- Rise of Transformers
    - NLP task: transformers took over from RNN
    - 2020 - ViT introduced
    - Originally the same as NLP besides patchify layer
    - No inductive biases!
- Even without inductive biases - ViT blew up because of scaling behavior
    - Bigger model, bigger dataset, significantly outperformed ResNets
- BUT! only for classification
    - ViT at first not good on all
- ViT quadratic complexity w.r.t. input size
    - High res = gg
- Hierarchial Transformers
    - Briedging gap with some inductive bias
    - "sliding window"
        - Swin
- Key point: **Convolution not irrelevant**
    - Are ConvNets really fading?
    - Transformers outperforming on many tasks
    - Usually due to Transformer scaling behavior
- What does ConvNeXt do?
    - ResNet modernized like Swin

### ConvNet Modernization Roadmap
- Paper starts with ResNet-50 and Swin-T as comparison (similar FLOPs)
- **NOTE: for this project will likely also use tiny!**
- Stages:
    - macro design
    - ResNeXt
    - Inverted Bottleneck
    - Large Kernel Size
    - Layer Wise Micro Designs

#### Training Techniques
- **Training procedure matters, on top of model architecture**
    - **Important to remember for my case**
- Transformers have different training techniques than ConvNets
    - AdamW optimizer
- Training recipe similar to Swin Transformer/DeiT

- AdamW optimizer
- 90 Epochs -> 300 Epochs
    *is more epochs always the solution...? What about overfitting and validation loss?*
- Data augmentation techniques
    - Mixup
    - Cutmix
    - RandAugment
    - Random Erasing
- Regularization Techniques
    - Stochastic Depth
    - Label Smoothing
- Complete set of techniques in Appendix A.1
- **Training techniques alone took ResNet-50 from 76.1% to 78.8% (+2.7)**

#### Macro Design
- Swin Transformer macro network
    - Follows ConvNets for multi-stage design (hierarchial)
    - Each stage has different feature map resolution
    - 2 design considerations:
- **Changing Stage Compute Ratio**
    | Model | Compute Ratio |
    | --- | --- |
    | ResNet-50 | 3 : 4 : 6 : 3 |
    | Swin-T | 1 : 1 : 3 : 1  |
    | Swin-B and larger | 1 : 1 : 9 : 1 |
    | Updated ResNet-50 | 3 : 3 : 9 : 3 |

    - Changing stage compute ratio improved model accuracy from **78.8% to 79.4% (+0.6)**
    - *"A more optimal design is likely to exist"...*
- **Changing Stem to "Patchify"**
    - *changing "stem-cell" structure*
    - How input images will be processed at network beginning
    - ResNet's Stem Cell:
        - 7 $\times$ 7 convolution layer, stride 2
        - 3 $\times$ 3 max pool
        - Result is 4x downsampling of input images
    - Vision Transformers: more aggressive "patchify" strategy used
        - Corresponds to large kernel size, non-overlapping convolution
    - Swin Transformer: similar "patchify"
        - Smaller patch size of 4 - for multi-stage design (resolution decreases with hierarchial features)
    - ConvNeXt: replace ResNet stem cell with patchify layer
        - 4 $\times$ 4, stride 4 convoutional layer
        - **Accuracy 79.4% to 79.5% (+0.1)**
    - *How is patchify less aggressive? Answered by Gemini 3.1 Pro: "This is considered less aggressive because it is mathematically closer to just reshaping the raw data. It does not explicitly discard values the way max pooling does. It simply bundles the raw pixels together and hands them off to the rest of the network, essentially saying, 'Here is all the raw data, nicely packaged. You figure out what is important.' TL;DR: Patchify closer to raw data."*

#### ResNeXt-ify
- ResNeXt has better FLOPs/accuracy trade-off than vanilla ResNet
- Core is **grouped convolution**
- High level: *"use more groups, expand width"*

```
A recap on how convolutions work.

For every channel:
1. A filter (e.g. 3x3, 5x5) is multiplied elementwise to the feature map
2. This operation is repeated across the entire 2D feature map
   Padding adds edges to the feature map that the filters cannot cross
   Stride changes how many points to travel with every filter application
3. The resulting activation maps for EVERY CHANNEL are summed

Note: this blurb's wording is like hella weird, might want to rewrite and fix later on
```

- ResNeXt uses **grouped convolution** for 3 $\times$ 3 conv layer in a bottleneck.
    - **grouped convolution** is a technique where the convolution is only applied to a given number of channels at a time. This saves FLOPs.
    - FLOPs reduced, network width is expanded to compensate for capacity loss
- ConvNeXt goes further - **depthwise convolution**
    - **depthwise convolution** is a technique where number of groups equals number of channels (1 per channel)
    - Similar to weighted sum operation in self attention!
    - $1\times 1$ convolutions used afterwards
    - Separation of spatial and channel mixing (ViT technique)
    - FLOPs reduced, network width increased 64 $\rightarrow$ 96, to 5.3G FLOPs
    - **Accuracy 79.5% to 80.5% (+1.0)**

#### Inverted Bottleneck
- Every Transformer block creates an **inverted bottleneck**
    - **inverted bottleneck:** MLP block hidden dimension is 4x wider than input dimension
- Inverted bottleneck - expansion ratio of 4 in MobileNetV2
- Reduces FLOPs to 4.6G
- **Slightly Improved Performance 80.5% to 80.6% (+0.1)**

#### Large Kernel Sizes
- ViT feature is non-local self-attention
    - each layer can have **global receptive field**
        - **global receptive field:** a single layer can "see" info from entire input image at once
- Gold standard (VGGNet madee popular) is stacked small kernel-sized conv layers ($3\times 3$)
- Swin uses 7 $\times$ 7 windows
- **Depthwise conv layer moved up**
    - Transformer inspired design decision
        - MSA $\rightarrow$ MLP
    - Results in fewer channels for complex/ineffecient modules
    - Dense 1 $\times $ 1 layers od heavy lifting
    - Brings FLOPs down to 4.1G, performance temporarily to 79.9%
- **Kernel Size Increased**
    - Experimented: 3, 5, 7, 9, 11
    - Kernel size saturates at 7 $\times$ 7, performance back to 80.6%
    - Most ViT design decisions can be mapped to ConvNet

#### Micro Design
- **ReLU $\rightarrow$ GELU**
    - NLP vs vision is architecture discrepancy
    - ReLU used in ConvNets - simple & efficient
        - ReLU also used in original Transformer paper!
    - GELU - smoother ReLU, used in most advnaced transformers
    - GELU swap has same accuracy
    - *Could be a consideration for our implementation - swap between ReLU and GELU to see if differences arise*
- **Fewer Activation Functions**
    - Transformers only have one activation function in MLP block
    - Common practice to have activation function each convolutional layer (including 1 $\times$ 1 conv)
    - Eliminate all GELU except between two 1 $\times$ 1 layers
    - **Performance 80.6% to 81.3% (+0.7)** - matches Swin-T!!
- **Fewer Normalization Layers**
    - Remove two BatchNorm layers, only 1 BN layer before conv 1 $\times$ 1 layer
    - **Performance 81.3% to 81.4% (+0.1)** - better than Swin-T
- **Substitute BN with LN**
    - Layer Norm used in Transformers
    - In ResNet, breaks it
    - In ConvNeXt, it makes performance better. What.
    - **Performance 81.4% to 81.5% (+0.1)**
- **Separate Downsampling Layers**
    - ResNet, downsampling by residual block @ start of each stage (3 $\times$ 3 conv, stride 2, 1 $\times$ 1 conv, stride 2 @ shortcut connection)
    - Swin separate downsampling between stages
        - *clarification on what this means: ResNet does downsampling as part of the convolutions, but Swin has MSA and downsampling separated.*
    - Similar, ConvNeXt uses 2 $\times$ 2 conv layers, stride 2 for spatial downsampling
    - Adding norm when resolution changed helps stabilize
        - Add LN before downsampling, one after stem, one after final global average pool
    - **Performance 81.5% to 82.0% (+0.5)**

#### Summary:
- ConvNeXt made, design choices taken from ViT
- Designs are not novel in ConvNet literature
    - But never used collectively

### Empirical Evaluations
- **How they trained on ImageNet - 1K**
    - 300 epochs
    - AdamW
        - lr 4e-3
    - 20 epoch linear warmup
        - -> cosine decay schedule
    - Batch size 4096
        - *my computer will likely get NOWHERE near this*
    - Weight decay 0.05
    - Augmentations
        - Mixup
        - Cutmix
        - RandAugment
        - Random Erasing
    - Regularize
        - Stochastic Depth
        - Label Smoothing
    - Layer Scale
        - Init value 1e-6
    - EMA used
        - Helps larger model overfitting
- **Pre-Training ImageNet-22K**
    - 90 epochs
        - Warmup 5 epochs
    - no EMA
    - other settings same as ImageNet 1K
- **Fine Tuning Imagenet-1K**
    - 30 epochs
    - AdamW
        - lr 5e-5
    - cosine lr schedule
    - layer-wise learning rate decay
    - no warmup
    - batch size 512
    - weight decay 1e-8
    - default pre-train, fine-tune, test resolution 224<sup>2</sup>
    - fine tune at 384<sup>2</sup> for both
    - **Compared to ViT/Swin, ConvNeXt is simpler to fine tune at different resolutions, no need to adjust input patch size**

#### Results
- Note: they use UperNet for semantic segmentation!!

#### Appendix/Implementations
**(pre-)training configurations**
| (pre-)training config | ConvNeXt-T/S/B/L, ImageNet-1K 224<sup>2</sup> | ConvNeXt-T/S/B/L/XL, ImageNet-22K 224<sup>2</sup> |
| --- | --- | --- |
| weight init | trunc. normal (0.2) | trunc. normal (0.2) |
| optimizer | AdamW | AdamW |
| base lr | 4e-3 | 4e-3 |
| weight decay | 0.05 | 0.05 |
| optimizer momentum | $\beta_1, \beta_2=0.9,0.999$ | $\beta_1, \beta_2=0.9,0.999$ |
| batch size | 4096 | 4096 |
| training epochs | 300 | 90 |
| learning rate schedule | cosine decay | cosine decay |
| warmup epochs | 20 | 5 |
| warmup schedule | linear | linear |
| layer-wise lr decay | None | None |
| randaugment | (9, 0.5) | (9, 0.5) |
| mixup | 0.8 | 0.8 |
| cutmix | 1.0 | 1.0 |
| random erasing | 0.25 | 0.25 |
| label smoothing | 0.1 | 0.1 |
| stochastic depth | 0.1/0.4/0.5/0.5 | 0.0/0.0/0.1/0.1/0.2 |
| layer scale | 1e-6 | 1e-6 |
| head init scale | None | None |
| gradient clip | None | None |
| EMA | 0.9999 | None |

**fine-tuning configurations**
| pre-training config | ConvNeXt-B/L, ImageNet-1K 224<sup>2</sup> | ConvNeXt-T/S/B/L/XL ImageNet-22K 224<sup>2</sup> |
| --- | --- | --- |
| **fine-tuning config** | **ImageNet-1K 384<sup>2</sup>** | **ImageNet-2K 224<sup>2</sup> and 384<sup>2</sup>** |
| optimizer | AdamW | AdamW |
| base lr | 5e-5 | 5e-5 |
| weight decay | 1e-8 | 1e-8 |
| optimizer momentum | $\beta_1, \beta_2=0.9,0.999$ | $\beta_1, \beta_2=0.9,0.999$ |
| batch size | 512 | 512 |
| training epochs | 30 | 30 |
| learning rate schedule | cosine decay | cosine decay |
| layer-wise lr decay | 0.7 | 0.8 |
| warmup epochs | None | None |
| warmup schedule | N/A | N/A |
| randaugment | (9, 0.5) | (9, 0.5) |
| mixup | None | None |
| cutmix | None | None |
| random erasing | 0.25 | 0.25 |
| label smoothing | 0.1 | 0.1 |
| stochastic depth | 0.8/0.95 | 0.0/0.1/0.2/0.3/0.4 |
| layer scale | pre-trained | pre-trained |
| head init scale | 0.001 | 0.001 |
| gradient clip | None | None |
| EMA | None | None(T-L)/0.9999(XL) |

**downstream tasks**

## Paper 2 - ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
### Abstract
- Convnets built for supervised learning with ImageNet labels
- Simple combination of MAE not good
- New FCMAE framework
- Introduced Global Response Normalization layer to ConvNeXt
    - Enhance inter-channel feature competition
- -> ConvNeXt V2
- **Note: ConvNeXt V2 atto, and other smaller may be better suited for our relatively small dataset**

### Introduction
- Performance of vision learning system 3 factors:
    - network architecture chosen
    - method used for training
    - data used for training
- Most common approach of neural net design is still through supervised learning performance on ImageNet
    - Separate line of research - focus of visual representation learning
        - Shfited from supervised learning to self-supervised pre-training with pretext objectives
- Masked Autoencoder brought succeses
    - SSL common practice is use *predetermined architecture*
        - e.g. MAE developed ViT architecture
    - Challenging for ConvNeXt
- **MAE ConvNet compatability issues**
    - MAE has encoder-decoder design optimized for transformers
        - Standard ConvNet dense sliding windows - incompatible
    - Transformers and ConvNets have different feature learning behaviors
        - Architecture and training objective relationship needs to be considered
- **Solution: Network Architecture MAE co-design**
    - Make Mask-based SSL effective for ConvNeXt
    - Treat input as set of sparse patches
        - Use sparse convolutions to process only visible parts
        - Inspired by use of sparse convolutions in 3D point clouds
    - Implement ConvNeXt with sparse convolutions $\rightarrow$ back to standard at fine-tuning
    - Transformer decoder replaced with single ConvNeXt block
        - Entire design is fully convolutional
            - **Note: this will likely get dropped after pre-training**
                - **Replace with fully connected layer for classification**
                - **Replace with UperNet for segmentation** 
- Initial implementation findings:
    - Learned features are useful
    - Improvement on baseline
    - Fine-Tuning performance not as good as transformer-based
        - Feature collapse issue on MLP layer with masked input
        - Response: **Global Response Normalization layer**
            - Most effective when model is pre-trained with MAE
            - Indicates reusing fixed architecture from supervised learning may be suboptimal

### Related Work
...

### Fully Convolutional Masked Autoencoder
- Conceptually simple - fully convolutional
#### Masking
- Ratio of 0.6
- **IMPORTANT NOTES ON HOW THIS WORKS**
> "As the convolutional model has a hierarchial design, where features are downsampled in different stages, the mask is generated in the **last stage** and upsampled recursively up to the finest resolution. To implement this in practice, we manually remove 60% of the 32 x 32 patches from the original input image. **We use minimal data augmentation, only including random resized cropping.**
- **What this means:**
    - **Mask stencil creation**
    - Need to make sure the masked chunks perfectly align with hierarchial layers of ConvNeXt
    - Define the masks matrix at lowest resolution, then scale up
    - Guarantees one pixel of deep mask expands to one 32x32 block at input
    - THEN applies this found stencil to mask out the original image
#### Encoder Design
- ConvNeXt aas encoder
- Challenge: keep model from learning shortcuts to copy and paste
    - ViT easy: visible patches only input to encoder
    - ConvNet hard: 2D image structure needs to be preserved
        - Learnable mask tokens not good - decrease efficiency
        - Train/test time inconsistency 
        - Issue especially with high mask ratio
- **Sparse Data Perspective**
    - Inspired by sparse point clouds
    - Masked image - 2D sparse array of pixels
    - Incorporate sparse convolution
        - **Submanifold Sparse Convolution:** only apply convolution to area with data
        - Only operate in visible training points
        - Can be easily converted back to regular convolution!
        - **ALTERNATIVE:** apply binary masking operation before & after dense convolution operation
            - Numerically same effect as sparse convolutions
            - Theoretically more computationally expensive
            - More friendly on AI accelerators like TPU
#### Decoder Design
- Plain ConvNeXt block as decoder

#### Reconstruction Target
- Mean squared error (MSE) between reconstructed and target images
- Similar to MAE, target is patch-wise normalized image of original input
- Loss only on masked patches

#### FCMAE
- Fully Convolutional Masked AutoEncoder
- Sparse conv needed! - 79.3% vs 83.7%
- **Pretraining Regime**
    - Use ImageNet 1K
        - 800 epochs for pre
        - 100 epochs for fine tune
- **Decoder Design**
    - A single ConvNext block does the best for fastest
- **Decoder Depth**
    - 1 or 4 blocks give best performance, but like why use two if you can use one
- **Decoder Width**
    -  256 or 512 give best results
- **Self-Supervised vs Supervised:**
    - Two baseline experiments:
        - Supervised 100 epoch baseline
        - 30 epoch supervised training in og ConvNeXt

    | supervised 100 epoch | supervised 300 epoch | FCMAE (800 pre, 100 fine epoch) |
    | --- | --- | --- |
    |  82.7 | 83.8 | 83.7 |

    - Better initialization than 100 epoch baseline, but not as good
        - ViT pretrain significantly outperform

### Global Response Normalization (GRN)
- Makes training more effective
#### Feature Collapse
- Qualitative analysis in the feature space
    - Visualize FCMAE activations
    - **"Feature Collapse"** phenomenon
        - Many dead/saturated feature maps
        - Mainly in dimension-expansion MLP layers
#### Feature Cosine Distance Analysis 
- Cosine distance
- Activation tensor $X \in R^{H\times W\times C}$
    - $\rightarrow X_i\in R^{H\times W}$ is feature map of $i$-th channel
    - Reshape to $HW$ dimensional vector
    ...
- TL;DR, ConvNeXt V1 experiences feature collapse with FCMAE
> Not yet in the paper, but this makes a lot of sense. Given the mask, it makes sense that certain filters are getting updates at certain points, leading to feature collapse.
#### Approach
- Human brain:
    - Lateral inhibition
- Global Response Normalization (GRN)
    1. Global feature aggregation
    2. Feature normalization
    3. Feature calibration
- **TODO: add the math here**
- Pseudocode:
    ```
    # gamma, beta: learnable affine transform parameters
    # X: input of shape (N, H, W, C)
    
    gx = torch.norm(X, p=2, dim=(1,2), keepdim=True)
    nx = gx / (gx.mean(dim=-1, keepdim=True+1e-6))
    return gamma * (X * nx) + beta + X
    ```
- L2 norm resulted in better performance than just global average pooling
- **TODO: write more before putting this into Obsidian**
- TLDR: GRN makes it better
#### ConvNeXt V2
-  Incorporated GRN into ConvNeXt block (before MLP layer)
-  LayerScale becomes unnecessary, removed
#### Impact of GRN
- V2 + FCMAE gives +0.9 over V1 + FCMAE
#### Relation to feature normalization methods
- Can other layers perform as well?
- GRN outperforms
#### Relation to feature gating methods
- **didn't read**
#### Role of GRN in pre-training/fine tuning
- When GRN is taken from fine tuning or pre-training, performance drops. Keep GRN in whole time.
### ImageNet Experiments
- ConvNeXt V2 popped off
#### Co-Design Matters
- Self-supervised framework and model architecture go hand-in-hand
- FCMAE without architecture change gives little impact
- GRN itself doesn't help either
- Model and learning framework must be considered together
- **Good to know for my project: as now I can use learning framework and model meant for each other.**
#### Model Scaling
- Atto size to Huge
    - Atto has 3.7M
    - **Start with Atto for my project?**
#### Previous Method Comparison
- TL;DR ConvNeXt V2 mogs
- ViT model surpassed in Huge
    - **Note for my project, scaling doesn't matter too much - dataset is kinda limited. Small scale gains matter more.**
#### ImageNet-22K Intermediate Fine-Tuning
- More fine-tuning
### Transfer Learning Experiments
#### COCO Object Detection & Segmentation
- Kinda hard to follow what's being said in this paragraph
#### Semantic Segmentation on ADE20k
**MOST IMPORTANT FOR MY PROJECT**
- Using UperNet framework
- Significantly improves over V1
- On par with Swin base, large, outperform huge
    - Base does not do as well as Swin on base with mIoU, but is still very close (-0.7).
### Conclusion
- ConvNeXt V2 performs exceptionally well in the FCMAE regime, and shows the key insight that training regime and model architecture should be co-designed.
### Appendix
#### Implementation Details

| Atto | Femto | Pico | Nano | Tiny | Big | Large | Huge | 
| --- | --- | --- | --- | --- | --- | --- | --- |
| C=40 | C=48 | C=64 | C=80 | C=96 | C=128 | C=192 | C=352 |
| B=(2, 2, 6, 2) | B=(2, 2, 6, 2) | B=(2, 2, 6, 2) | B=(2, 2, 8, 2) | B=(3, 3, 9, 3) | B=(3, 3, 27, 3) | B=(3, 3, 27, 3) | B=(3, 3, 27, 3) |

#### Imagenet Experiments
**Pre-Training**
- Same pre-training setup

| config | value |
| --- | --- |
| optimizer | AdamW |
| base learning rate | 1.5e-4 |
| weight decay | 0.05 |
| optimizer momenetum | $\beta_1, \beta_2=$ 0.9, 0.5 |
| batch size | 4096 | 
| learning rate schedule | cosine decay |
| warmup epochs | 40 |
| training epochs | 800 or 1600 |
| augmentation | RandomResizedCrop | 

- Linear scaling rule $lr=baselr\times batchsize / 256$ (for warmup epochs)

**ImageNet-1K Fine Tuning**
- Fine tuning recipe varied by model size
- **Longer fine-tuning epochs help small models**
- 2 different learning-rate layer decay
    - Group-wise
        - 3 sequential layers is one "layer" - use same decaying value for them
    - Layer-wise
        - Distinct value for each layer
    - Both standard decaying rule
        - Default is layer-wise, group-wise applied to Base and Large
            **Why??**
**ImageNet-22K Intermediate Fine Tuning**
- Setups + larger layer-wise learning rate decay values for small models:

**Unfortunately, no info on integration with UperNet**.


