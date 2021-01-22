import numpy as np
from sys import argv, exit
from sklearn.metrics import jaccard_similarity_score, adjusted_rand_score, precision_score, f1_score, \
    adjusted_rand_score, fowlkes_mallows_score
from scipy import signal
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
from math import factorial
from collections import Counter
from math import log
# from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings


# ------------------------------------------- args ------------------------------------------- #
mode = 'LSTM'  # 'RNN'
vidPath = '/Users/jackshi/furniture_assembly/assembly/dataset/camera_data/camera_0_50_224_224_20_D210117_160343'
trialName = 'trial_D210122_013514'

predFileName = 'ES_out_{}_{}.txt'.format(mode, trialName[6:])
gtFileName = 'labels.npy'
errorPlotName = 'error_{}.png'.format(mode)
segmentsPlotName = 'segments_{}.png'.format(mode)
seqs = [50]  # 50, 32, 35, 34, 33, 20, 18, 27, 9, 11

relativeExtremaOrder = 20
movingAverageWinSize = 5


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


# predFilePath = argv[1]
# gtFilePath = argv[2]

# predFiles = sorted(
#     [join(predFilePath, f) for f in listdir(predFilePath) if isfile(join(predFilePath, f)) and f.endswith('.txt')])
predFiles = sorted([join(vidPath, str(seq), predFileName) for seq in seqs])
gtFiles = sorted([join(vidPath, str(seq), gtFileName) for seq in seqs])

files = zip(predFiles, gtFiles)

# fps = 30
# winSize = np.ceil(fps / 2) // 2 * 2 + 1

numVideos = 0
for predFile, gtFile in files:
    seq = gtFile.split('/')[-2]
    numVideos += 1

    predFrames = []
    predFrames1 = []
    predErrors = []
    avgFr = []
    classNo = 0
    BGClass = []

    gtLabels = np.load(gtFile)
    gtLabels = gtLabels[1:]  # crop away the first frame to keep length same as event segmentation
    gtBoundaries = np.where(gtLabels == 1)[0]

    with open(predFile, 'rb') as file:
        lineNo = 0
        for line in file:
            data = line.replace('\n', '').split('\t')
            frameNo, predError = data
            frameNo, predError = int(frameNo), float(predError)
            predErrors.append(predError)
    # print predErrors

    predErrors_Ori = predErrors
    # predErrors = movingaverage(predErrors, 80)
    predErrors = movingaverage(predErrors, movingAverageWinSize)
    predErrors = np.gradient(np.array(predErrors)).tolist()

    # predBoundaries = signal.argrelextrema(np.array(predErrors), np.greater, order=int(0.57899 * 200))[0].tolist()
    predBoundaries = signal.argrelextrema(np.array(predErrors), np.greater, order=int(relativeExtremaOrder))[0].tolist()
    # predBoundaries.append(len(gtFrames) - 1)

    x = np.arange(len(predErrors))
    predErrors_Ori = np.array(predErrors_Ori)
    # plot pred error
    plt.figure(num=1, figsize=(8, 6))
    plt.plot(x, predErrors_Ori)
    plt.scatter(x[gtBoundaries], predErrors_Ori[gtBoundaries], label='Ground Truth', c='r')
    plt.xlabel("Frames", fontsize=20)
    plt.ylabel("Prediction Error", fontsize=20)
    lgd = plt.legend()
    plt.title("Sequence {} Self-Supervised Prediction Error ({})".format(seq, mode), fontsize=20)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(join(vidPath, str(seq), errorPlotName))
    plt.close()
    print('Plot saved to {}'.format(join(vidPath, str(seq), errorPlotName)))

    # plot segments
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Sequence {} Segments ({})".format(seq, mode), fontsize=20)
    ax1.hlines(1, 0, len(x))
    ax1.eventplot(gtBoundaries, orientation='horizontal', colors='b')
    ax1.axis('off')
    ax1.set_title('Ground Truth Segments')
    ax2.hlines(1, 0, len(x))
    ax2.eventplot(predBoundaries, orientation='horizontal', colors='b')
    ax2.axis('off')
    ax2.set_title('Prediction Segments')
    # fig.tight_layout()
    fig.savefig(join(vidPath, str(seq), segmentsPlotName))
    plt.close(fig)
    print('Plot saved to {}'.format(join(vidPath, str(seq), segmentsPlotName)))


#     outFile = predFile.replace('.txt', '_predBoundaries.txt')
#     with open(outFile, 'w') as of:
#         for p in predBoundaries:
#             of.write('%d\n' % p)
# print "Fin"
