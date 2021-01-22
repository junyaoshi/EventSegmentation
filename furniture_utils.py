from __future__ import print_function
import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir
from random import shuffle, choice
from PIL import Image
import sys
import json
import collections
import cv2

input_width = 224
input_height = 224


def addDateTime(s = ""):
    """
    Adds the current date and time at the end of a string.
    Inputs:
        s -> string
    Output:
        S = s_Dyymmdd_HHMM
    """
    import datetime
    date = str(datetime.datetime.now())
    date = date[2:4] + date[5:7] + date[8:10] + '_' + date[11:13] + date[14:16] + date[17:19]
    return s + '_D' + date


def loadFurnitureData(vidPath, camNum=0):
    batchPaths = [str(join(vidPath, f, str(camNum))) for f in listdir(vidPath) if isdir(join(vidPath, f))]
    return batchPaths


def loadFurnitureMiniBatch(vidFilePath, mode='RNN'):
	vidName = vidFilePath.split('/')[-2]
	frameList = sorted(
		[join(vidFilePath, f) for f in listdir(vidFilePath) if isfile(join(vidFilePath, f)) and f.endswith('.png')])
	its = [iter(frameList), iter(frameList[1:])]
	segments = zip(*its)
	minibatch = []
	for segment in segments:
		im = []
		for j, imFile in enumerate(segment):
			img = Image.open(imFile)
			img = img.resize((input_width, input_height), Image.ANTIALIAS)
			if mode == 'LSTM':
				img = np.array(img)
				img = img[:, :, ::-1].copy()
				img = cv2.GaussianBlur(img, (5, 5), 0)
			im.append(np.array(img))
		minibatch.append(np.stack(im))
	return vidName, minibatch


def loadMiniBatch(vidFilePath):
	vidName = vidFilePath.split('/')[-3]
	frameList = sorted([join(vidFilePath, f) for f in listdir(vidFilePath) if isfile(join(vidFilePath, f)) and f.endswith('.png')])
	frameList = sorted(frameList, key=lambda x: int(x.split('/')[-1].split('.')[0]))
	its = [iter(frameList), iter(frameList[1:])]
	segments = zip(*its)
	minibatch = []
	for segment in segments:
		im = []
		numFrames = 0
		for j, imFile in enumerate(segment):
			img = Image.open(imFile)
			img = img.resize((input_width, input_height), Image.ANTIALIAS)
			im.append(np.array(img))
			numFrames += 1
		minibatch.append(np.stack(im))
	return vidFilePath, minibatch


if __name__ == '__main__':
	vidPath = '/Users/jackshi/furniture_assembly/assembly/dataset/camera_data/camera_0_50_224_224_20_D210117_160343'
	batch = loadFurnitureData(vidPath)
	print('Loaded Batch')
	shuffle(batch)

	for miniBatchPath in batch:
		vidName, minibatches = loadFurnitureMiniBatch(miniBatchPath)
		print('Loaded minibatch {}'.format(vidName))
		print()
