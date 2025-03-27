import numpy as np

def ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET):
    numTrain2D = numTrain * numTrain - numGuard * numGuard
    RDM_mask = np.zeros_like(RDM_dB)

    for r in range(numTrain + numGuard + 1, RDM_dB.shape[0] - (numTrain + numGuard)):
        for d in range(numTrain + numGuard + 1, RDM_dB.shape[1] - (numTrain + numGuard)):
            Pn = (np.sum(RDM_dB[r - (numTrain + numGuard):r + (numTrain + numGuard),
                              d - (numTrain + numGuard):d + (numTrain + numGuard)]) -
                 np.sum(RDM_dB[r - numGuard:r + numGuard,
                              d - numGuard:d + numGuard])) / numTrain2D

            a = numTrain2D * (P_fa**(-1 / numTrain2D) - 1)
            threshold = a * Pn

            if RDM_dB[r, d] > threshold and RDM_dB[r, d] > SNR_OFFSET:
                RDM_mask[r, d] = 1

    cfar_ranges, cfar_dopps = np.where(RDM_mask == 1)
    return RDM_mask, cfar_ranges, cfar_dopps, len(cfar_dopps)