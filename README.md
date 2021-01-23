# A Perceptual Prediction Framework for Self Supervised Event Segmentation
Code for the paper A Perceptual Prediction Framework for Self Supervised Event Segmentation

## Preparation:
  All videos must be extracted into individual frames. One folder per video with each folder containing the frames for the video.
  
   Example code for extracting frames is shown in preprocessVideo.py for the Breakfast Actions dataset. Download the videos from https://uni-bonn.sciebo.de/index.php/s/QRQuGAtNfOYi3yM. After extraction, run the ```preprocessVideo.py``` code for extracting frames. Takes approximately 3-6 hours depending on your computation.
  
  Requirements: Python 2.7, Tensorflow 1.6+, scipy and numpy
  
  A JSON file with th video names to process as keys also needs to be created.
  
  
## Training
  Run Zacks_VGG_RNN.py for RNN based prediction model
  
  Run Zacks_VGG_RNN_EventLayer.py for LSTM based prediction model
  
  Arguments for the program are the same:
  
    ```python Zacks_VGG_RNN.py <jsonData path> <video frames root directory> <path to save model> <use active learning (1) or not (0)>```
  
## Inference

  Run ```Zacks_VGG_RNN_Inference.py``` for RNN based prediction model
  
  Run ```Zacks_VGG_RNN_EventLayer_Inference.py``` for LSTM based prediction model
  
  Arguments for the program are the same:
   ``` python Zacks_VGG_RNN.py <jsonData path> <video frames root directory> <path to restore model> <output file name to write loss characteristics> ```

  An output text file will be written into each folder with the prediction loss at each time instance t.
  
  Run the ```getBoundaries.py``` script to transform the loss file into boundaries and visualize the predictions.
  
## Evaluation Remarks
  Note: Due to the self-supervised nature of the approach, the output classes will be in serial order (i.e. 0,1, ..., n). TO evaluate, transform ground truth to same format and run evaluation scripts for the dataset. 
  
  Example Evaluation for Breakfast Actions Dataset is shown in ```evaluateBreakfastActions.py```.
  
  Note: The original script was written in Tensorflow 1.3. Newer versions of Tensorflow and Python have shown variations in performance. Running this script should lead to the reported accuracy of 42.8% MoF on Breakfast Actions.

Please cite if this has been useful:
```
@article{aakur2018perceptual,
  title={A Perceptual Prediction Framework for Self Supervised Event Segmentation},
  author={Aakur, Sathyanarayanan N and Sarkar, Sudeep},
  journal={arXiv preprint arXiv:1811.04869},
  year={2018}
}
```
## Quick Start Guide for Processing Furniture Data

### Conda Setup

Set up conda environment with Python 2.7. Here are a list of necessary packages:

- tensorflow 1
- pillow
- numpy
- matplotlib
- opencv-python
- tqdm

### Download VGG16 Pre-trained Model

Here: https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models
Download VGG16 and put it under this repository

### A Note about Batch Sampling

In the training and inference scripts, you may see code like this :

```python
batch = loadFurnitureData(vidPath)
batch = batch[:10]
print('Selected Sequences: ')
for b in batch:
    print(b)

```

Due to limited computational capacity, I only sampled 10 sequences. Feel free to modify this code to train for all or a subset of all sequences.

### Training

Run ```Zacks_VGG_RNN.py``` (RNN) or ```Zacks_VGG_RNN_EventLayer.py``` (LSTM).
Modify the args at the top of the files. Make sure to change ```vidPath``` to the parent directory of all different sequences of videos.
The active learning argument is set to True because the paper mentiones that active learning improves the performance.

The trained model will be saved at ```models/LSTM(or RNN)/trial...```
### Inference

Run ```Zacks_VGG_RNN_Inference.py``` (RNN) or ```Zacks_VGG_RNN_EventLayer_Inference.py``` (LSTM). Modify the args at the top of the files. In particular, change ```vidPath``` accordingly and change ```trialName``` to where the model is saved.

The results will be saved under the folder of each sequence under ```vidPath```.

### Evaluation

Run ```evaluateFurniture.py```. Modify the args at the top of the file, including ```vidPath``` and ```trialName```. Specify what sequences are evaluated using ```seqs```.

The plots will be saved under the folder of each specified sequence under ```vidPath```.

