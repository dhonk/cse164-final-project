# Time To Cook
## Lecture Note Insights - Brainstorm!
- Entropy Based Regularization looks very very promising! 
    - I know that classifier will be somewhat straightforward to do
    - I wonder if semantic segmenter would also be possible with entropy-based regularization?
- I FINALLY GET HOW ENCODERS/DECODERS WORK RAHHHH
    - ITS ALL TENSORS
    ![alt text](src/alwaystensors.jpg)
    - ALL THE WAY DOWN
    - U-NET MAKES SO MUCH SENSE
    - RAHHHHHHH
- I GET HOW LLMS WORK OMG
    - after ALL THESE YEARS
    - attention finally makes sense
- Data Augmentations
    - augmenting data can maybe get me even further!!
    - dataset is still relatively small, even with unlabeled samples
    - evaluate performance as is first, then maybe augment data for even more samples
- ... bruh...
why is it so hard to get sparse convolutions... PAIN
- Let's see if Docker works right!!
- Ok, now the other way...?
    - OKAYYYYYY

- WAIT IDEA
    - The model is being kinda forced to learn a "useless" class - the background class
    - Remember UNet toy results - background saturated the results
        - Separation of foreground and background?

## First Presentation
> "Good! Get it working, I'm sure you'll have strong performance. But compatability issues sound non-trivial".

BOOYEAHHHH THANK YOU LORD JESUS

### Notes from other presenters
- Model ensembling
- Sebastian used boundary head for decoder!!!
    - Maybe copy...?
- TTA!!!

- If first round of pseudolabeling doens't work - try again!

- Segmentation + Boundary - Binary CE loss, dice loss for segmentation
    - This is moreso auxillary
    - Not as reliant on good overlap
    - Not core

## The Future?
- TTA
- Multi cropping during inference!
- Figure out better way to handle image sizes

## WOW. PSEUDOSPARSE. FTW
MinkowskiEngine would've been the best.
However, performance on the overfit sample:

spconv: 1hr 45m 41s.

pseudodense: 1m 6s