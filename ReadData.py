import numpy as np


def getadcDataFromDCA1000(fileName, numLanes=2, numRX=4, numADCSamples=256, idxProcChirp=128):
    adcData = np.fromfile(fileName, dtype=np.int16)
    numFrame_cal = adcData.shape[0]
    fileSize = adcData.shape[0]
    
    adcData = adcData.reshape(-1, numLanes*2).transpose()
    fileSize = int(fileSize/2)
    LVDS = np.zeros((2, fileSize))

    temp = np.empty((adcData[0].size + adcData[1].size), dtype=adcData[0].dtype)
    temp[0::2] = adcData[0]
    temp[1::2] = adcData[1]
    LVDS[0] = temp
    
    temp = np.empty((adcData[2].size + adcData[3].size), dtype=adcData[2].dtype)
    temp[0::2] = adcData[2]
    temp[1::2] = adcData[3]
    LVDS[1] = temp

    adcData = np.zeros((numRX, int(fileSize/numRX)), dtype='complex_')
    iter = 0
    
    for i in range(0, fileSize, numADCSamples * 4):
        adcData[0][iter:iter+numADCSamples] = LVDS[0][i:i+numADCSamples] + np.sqrt(-1+0j)*LVDS[1][i:i+numADCSamples]
        adcData[1][iter:iter+numADCSamples] = LVDS[0][i+numADCSamples:i+numADCSamples*2] + np.sqrt(-1+0j)*LVDS[1][i+numADCSamples:i+numADCSamples*2]
        adcData[2][iter:iter+numADCSamples] = LVDS[0][i+numADCSamples*2:i+numADCSamples*3] + np.sqrt(-1+0j)*LVDS[1][i+numADCSamples*2:i+numADCSamples*3]
        adcData[3][iter:iter+numADCSamples] = LVDS[0][i+numADCSamples*3:i+numADCSamples*4] + np.sqrt(-1+0j)*LVDS[1][i+numADCSamples*3:i+numADCSamples*4]
        iter = iter + numADCSamples

    adcDataReshape = adcData.reshape(numRX, -1, numADCSamples)
    return numFrame_cal, adcDataReshape

def ADC_frame(adcDataReshape_frame, numRX=4, numADCSamples=256, idxProcChirp=128):
    dataRadar = np.zeros((numRX * 2, idxProcChirp, numADCSamples), dtype='complex_')
    dataRadar2 = np.zeros((numRX, idxProcChirp, numADCSamples), dtype='complex_')
    numChirp = idxProcChirp * 3
    
    for idxRX in range(numRX):
        for idxChirp in range(numChirp):
            if idxChirp % 3 == 0:
                dataRadar[idxRX, idxChirp // 3] = adcDataReshape_frame[idxRX, idxChirp]
            if idxChirp % 3 == 1:
                dataRadar2[idxRX, idxChirp // 3] = adcDataReshape_frame[idxRX, idxChirp]
            elif idxChirp % 3 == 2:
                dataRadar[idxRX + 4, idxChirp // 3] = adcDataReshape_frame[idxRX, idxChirp]
    
    dataRadar = np.transpose(dataRadar, (2, 1, 0))
    return dataRadar