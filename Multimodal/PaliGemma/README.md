# Vision Language model
Vision language model can extract information from the images.

## Text and image encoding
We need a vision and text encoder, we can do this with CLIP

### Problem with CLIP
As we are calculating attention of two different sequences, one of text and the other of image we need to calculate all the values in the attention matrix as it is not symetrical across the diagonal. That's why we use SigLip, for it we substitute the CEL with Sigmoid loss. For SigLip instead of doing the softmax for all the attention matrix columns and rows we are using Sigmoid for each element independently, this is especially useful as we can parallelize the attention matrix calculation.

## Segmentation and object detection with tokens
PaliGemma is a single-turn vision language model not meant for conversational use, and it works best when fine-tuning to a specific use case.

You can configure which task the model will solve by conditioning it with task prefixes, such as “detect” or “segment”. Each detection is represented by four location coordinates in the order $y_{min}$, $x_{min}$, $y_{max}$, $x_{max}$.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/paligemma/detect.png)

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/paligemma/segment.png)