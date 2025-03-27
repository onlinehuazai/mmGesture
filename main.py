from ReadData import getadcDataFromDCA1000, ADC_frame
from CFAR import ca_cfar
from FFT import DopplerFFT, rangeFFT, angleFFT, specific_range, specific_Doppler
from IntraFrameProcess import statistical_outlier_removal_intra_frame
from InterFrameProcess import statistical_outlier_removal_inter_frame
from utils import clutterRemoval, MediaFilter, kalman_filter_on_lists, smooth_savgol_filter, plot_trajectory
import numpy as np
import os

c = 3.0e8
B = 3997.04e6
K_S = 68.654e12
T = B / K_S
Tc = 40e-3
fs = 5e6
f0 = 77e9
lamb = c / f0
d_l = lamb / 2
idxProcChirp = 128
numADCSamples = 256
numRX = 4
numTX = 3
numLanes = 2
numChirp = idxProcChirp * 3

numGuard = 2
numTrain = numGuard * 2
P_fa = 1e-5
SNR_OFFSET = -3

path = "adc_data\\adcData.bin"
image_path = f'imgs\\'
padded_image_path = f'imgs\\'

if not os.path.exists(image_path):
    os.makedirs(image_path)
if not os.path.exists(padded_image_path):
    os.makedirs(padded_image_path)

filesize, adcData = getadcDataFromDCA1000(
    fileName=path, numLanes=numLanes, numRX=numRX, numADCSamples=numADCSamples, idxProcChirp=idxProcChirp)

numFrame = filesize // idxProcChirp // numADCSamples // numRX // numTX // numLanes

resultAllFrame = []
coors = []
for idxFrame in range(0, numFrame):
    frame = adcData[:, numChirp*(idxFrame):numChirp*(idxFrame+1), 0:numADCSamples]
    outputframe = ADC_frame(frame, numRX=numRX, numADCSamples=numADCSamples, idxProcChirp=idxProcChirp)
    resultAllFrame.append(outputframe)

for idxFrame in range(0, numFrame):
    R_list = []
    range_x = 0
    range_y = 0
    data_radar = resultAllFrame[idxFrame]
    range_profile = rangeFFT(data_radar, numADCSamples=numADCSamples, idxProcChirp=idxProcChirp, numRX=numRX)
    range_profile_clutterRemoval = clutterRemoval(range_profile, axis=1)
    speed_profile = DopplerFFT(range_profile_clutterRemoval, numADCSamples=numADCSamples, idxProcChirp=idxProcChirp, numRX=numRX)
    speed_profile = specific_range(speed_profile, range_min=0, range_max=32)
    speed_profile = specific_Doppler(speed_profile, idxProcChirp=128, spped_K=72)

    magnitudes = np.sqrt(np.real(speed_profile)**2 + np.imag(speed_profile)**2)
    total_magnitudes = np.mean(magnitudes, axis=2)
    RDM_dB = 10 * np.log10(total_magnitudes / np.max(total_magnitudes))
    RDM_dB[:3, :] = -100

    RDM_mask, cfar_ranges, cfar_dopps, K = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET)
    angle_profile = angleFFT(speed_profile[cfar_ranges, cfar_dopps, :], lamb=lamb, d_l=d_l, fft_size=180)
    RDM_mask = RDM_mask.astype(bool)
    magnitude_profile = total_magnitudes[RDM_mask]
    magnitude_sum = np.sum(magnitude_profile)

    each_frame = []
    for k in range(len(angle_profile)):
        fb = ((cfar_ranges[k] - 1) * fs) / numADCSamples
        fd = (cfar_dopps[k] + 28 - idxProcChirp//2 - 1) / (idxProcChirp * Tc)
        R = c * (fb - fd) / (2 * K_S)
        R_list.append(R)
        each_frame.append([-R_list[k] * np.sin(angle_profile[k]), R_list[k] * np.cos(angle_profile[k])])
    R_list.clear()

    if each_frame == []:
        continue
    else:
        each_frame, inlier = statistical_outlier_removal_intra_frame(each_frame, k=len(each_frame), std_dev_multiplier=0.5)
    
    magnitude_profile_new = magnitude_profile[inlier]
    magnitude_sum_new = np.sum(magnitude_profile_new)
    for k in range(len(each_frame)):
        range_x += each_frame[k][0] * magnitude_profile_new[k] / magnitude_sum_new
        range_y += each_frame[k][1] * magnitude_profile_new[k] / magnitude_sum_new
    coors.append([range_x, range_y])

data = list(coors)
data = statistical_outlier_removal_inter_frame(data, std_dev_multiplier=1.5)

x_coords, y_coords = MediaFilter(data, k=2, threshold=0.02, mindistance=0.05)
x_coords, y_coords = kalman_filter_on_lists(x_coords, y_coords)
x_smooth, y_smooth = smooth_savgol_filter(x_coords, y_coords, window_length=9, polyorder=5)
plot_trajectory(x_coords, y_coords, image_path, padded_image_path, long_edge=196)