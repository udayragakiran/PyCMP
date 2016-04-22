# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:50:22 2016

@author: OHNS
"""

import pickle
import os
import sys
import traceback
import datetime

from PyQt4 import QtCore, QtGui, uic

import numpy as np
import scipy.signal

import AudioHardware
import xlsxwriter
import CMPCommon

import matplotlib.pylab as plt
import pickle

class DPOAEData:
    def __init__(self, freqArray, ampArray, nunInputPts):
        self.freqArray = freqArray
        
        numFreq = len(freqArray)
        numAmp = len(ampArray)
        
        self.numFreq = numFreq
        self.numAmp = numAmp
        
        self.ampArray = ampArray
        
        self.F2Resp = np.zeros((numFreq, numAmp))
        self.F1Resp = np.zeros((numFreq, numAmp))
        self.DPResp = np.zeros((numFreq, numAmp))
        self.noise_std = np.zeros((numFreq, numAmp))
        self.noise_mean = np.zeros((numFreq, numAmp))
        self.mic_data = np.zeros((numFreq, numAmp, nunInputPts))
        self.thresholds = 80*np.ones(numFreq)

        self.F2Resp[:, :] = np.nan        
        self.F1Resp[:, :] = np.nan
        self.DPResp[:, :] = np.nan
        self.noise_std[:, :] = np.nan
        self.noise_mean[:, :] = np.nan
        self.mic_data[:, :, :] = np.nan
        self.mic_fft_mag = None  # will be generated on first processDPOAEata iteration
        
class DPOAEPointData:
    def __init__(self):
        self.mic_data_avg = None   # averaged mic response
        
        # self.filtBioamp = None   # filtered bioamp response
        self.t = None   # time
        self.mic_fft_mag = None
        self.mic_fft_freq = None
        self.mic_freq_mag = None
        
        
def makeDPOAEOutput(freq, trialDur, audioHW):
    outV = 1
    outputRate = audioHW.DAQOutputRate
    
    trialPts = np.ceil(trialDur * outputRate)
    stimEnv = 1e-3
    envPts = np.ceil(stimEnv * outputRate)
    t = np.linspace(0, trialDur, trialPts)
    F2 = freq
    F1 = freq/1.22
    sig1 = outV*np.sin(2*np.pi*F2*t)
    sig2 = outV*np.sin(2*np.pi*F1*t)
    
    envFcn = np.ones((trialPts))
    envFcn[0:envPts] = np.linspace(0, 1, envPts)
    envFcn[trialPts-envPts:] = np.linspace(1, 0, envPts)
    sig1 = sig1*envFcn
    sig2 = sig2*envFcn
    
    return sig1, sig2
    
def plotDPOAEdata(appObj, DPOAEptData, DPOAEdata):
    pl = appObj.DPOAE_micInput
    pl.clear()
    pl.plot(DPOAEptData.t, DPOAEptData.mic_data_avg, pen='b')
    labelStyle = appObj.xLblStyle
    pl.setLabel('bottom', 'Time', 's', **labelStyle)
    labelStyle = appObj.yLblStyle
    pl.setLabel('left', 'Resp', 'Pa', **labelStyle)
    
    pl = appObj.DPOAE_micInput
    pl.clear()
    pl.plot(DPOAEptData.t, DPOAEptData.mic_data_avg, pen='b')
    labelStyle = appObj.xLblStyle
    pl.setLabel('bottom', 'Time', 's', **labelStyle)
    labelStyle = appObj.yLblStyle
    pl.setLabel('left', 'Resp', 'V', **labelStyle)
                
    pl = appObj.DPOAE_micFFT
    pl.clear()
    pl.plot(DPOAEptData.mic_fft_freq, DPOAEptData.mic_fft_mag, pen='b')
    pl.plot([DPOAEptData.DP_freq], [DPOAEptData.DP_mag], pen='r', symbol='o')

    labelStyle = appObj.xLblStyle
    pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
    labelStyle = appObj.yLblStyle
    pl.setLabel('left', 'Magnitude', 'V', **labelStyle)
    freq = DPOAEdata.freqArray
    if len(freq) == 1:
        pl.setXRange(0.25*freq[0], 2*freq[-1])
    else:
        pl.setXRange(0.9*freq[0], 1.1*freq[-1])
    
    pl = appObj.DPOAE_response
    pl.clear()

    numFreq = len(DPOAEdata.freqArray)
    numAmp = len(DPOAEdata.ampArray)
    if numFreq == 1:
        DPOAEresp = DPOAEdata.DPResp[0, :]
        pl.plot(DPOAEdata.ampArray, DPOAEresp, pen='b', symbol='o')
        noise_mean = DPOAEdata.noise_mean[0, :]
        noise_std = DPOAEdata.noise_std[0, :]
        pl.plot(DPOAEdata.ampArray,  noise_mean + 3*noise_std, pen='r', symbol='o')   
        labelStyle = appObj.xLblStyle
        pl.setLabel('bottom', 'Amplitude', 'dB', **labelStyle)
    else:
        for n in range(0, numAmp):
            DPOAEresp = DPOAEdata.DPResp[:, n]
            pl.plot(DPOAEdata.freqArray, DPOAEresp, pen=(n, numAmp), symbol='o')
        labelStyle = appObj.xLblStyle
        pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
        
    labelStyle = appObj.yLblStyle
    pl.setLabel('left', 'Response', 'V', **labelStyle)
    
    pl = appObj.DPOAE_threshold
    pl.clear()
    pl.plot(DPOAEdata.freqArray, DPOAEdata.thresholds, pen='b', symbol='s')
    
def calcThreshold(DPOAEdata, freq_idx):
    sig = DPOAEdata.DPResp
    numSD = 2.5
    noise = DPOAEdata.noise_mean[freq_idx, :] + numSD*DPOAEdata.noise_std[freq_idx, :]
    threshold = 80
    DPOAEdata.thresholds[freq_idx] = threshold
    
def processDPOAEData(mic_data, freq, freq_idx, amp_idx, freqArray, ampArray, inputRate, DPOAEdataIn):
    # print("SpeakerCalProtocol: processData: mic_data=" + repr(mic_data))
    # ensure data is 1D
    numpts = len(mic_data)
    ptsPerRep = numpts
    #ptsPerRep = numpts // nReps
    #print("processDPOAEData: numpts= %d ptsPerRep= %d" % (numpts, ptsPerRep))
    
    # mic_data = np.reshape(mic_data, (ptsPerRep, nReps))
    print("processDPOAEData: mic_data.shape= ", mic_data.shape)
    
    #mic_data = np.mean(mic_data, 1)
    #print("processDPOAEData: (after averaging) mic_data.shape= ", mic_data.shape)

    # t = np.linspace(0, ptsPerRep/inputRate, ptsPerRep)
    t = np.linspace(0, numpts/inputRate, ptsPerRep)
    
    print("processDPOAEData: t.shape=", t.shape)
    zero_pad_factor = 2
    numfftpts = ptsPerRep*zero_pad_factor
    numfftpts = int(2 ** np.ceil(np.log2(numfftpts)))
    winfcn = np.hanning(ptsPerRep)
    win_corr = 2  # magnitude correction for hanning window
    mic_fft = np.fft.fft(winfcn*mic_data, numfftpts)
    endIdx = np.ceil(numfftpts/2)
    mic_fft = mic_fft[0:endIdx]
    mic_fft_mag = 2*np.abs(mic_fft)
    
    # convert to dB, correctting for RMS and FFT length
    fftrms_corr = 1/(ptsPerRep*np.sqrt(2))
    mic_fft_mag = win_corr*fftrms_corr*mic_fft_mag 
    mic_fft_mag_log = 20*np.log10(mic_fft_mag/20e-6 )  # 20e-6 pa
    
    mic_freq = np.linspace(0, inputRate/2, endIdx)
    fIdx = int(np.floor(freq*numfftpts/inputRate))
    print("processDPOAEData:  freq= %f "  % (freq))

    mic_stim_freq_mag = np.NAN
        
    F2_freq = freq
    F1_freq = freq/1.22
    DP_freq = 2*F1_freq - F2_freq
    try:            
        npts = zero_pad_factor
        
        fIdx = int(np.floor(F2_freq*numfftpts/inputRate))
        mag_rgn = mic_fft_mag_log[fIdx-npts:fIdx+npts]
        maxIdx = np.argmax(mag_rgn)
        F2_freq_mag = mag_rgn[maxIdx]
        
        fIdx = int(np.floor(F1_freq*numfftpts/inputRate))
        F1_idx = fIdx
        mag_rgn = mic_fft_mag_log[fIdx-npts:fIdx+npts]
        maxIdx = np.argmax(mag_rgn)
        F1_freq_mag = mag_rgn[maxIdx]

        fIdx = int(np.floor(DP_freq*numfftpts/inputRate))
        DP_idx = fIdx
        mag_rgn = mic_fft_mag_log[fIdx-npts:fIdx+npts]
        maxIdx = np.argmax(mag_rgn)
        DP_freq_mag = mag_rgn[maxIdx]

        noise_rgn = mic_fft_mag_log[DP_idx+npts:F1_idx-npts]         
        mic_fft_noise_mean = np.mean(noise_rgn)
        mic_fft_noise_std = np.std(noise_rgn)
    except Exception as ex:
        traceback.print_exc()
    
    # process the bioamp
    #Wn = [300/inputRate, 3e3/inputRate]
    #(b, a) = scipy.signal.butter(5, Wn=Wn, btype='bandpass')
    #filt_bioamp = scipy.signal.lfilter(b, a, bioamp_data) 
    
    print("processDPOAEData: DP= %0.3f noise_mean=%0.3f noise_std=%0.3f " % (DP_freq_mag, mic_fft_noise_mean, mic_fft_noise_std))
    pData = DPOAEPointData()
    pData.mic_data_avg = mic_data
    
    #pData.filtBioamp = filt_bioamp
    pData.t = t
    pData.mic_fft_mag = mic_fft_mag_log
    pData.mic_fft_freq = mic_freq
    pData.F2_mag = F2_freq_mag
    pData.F1_mag = F1_freq_mag
    pData.DP_mag = DP_freq_mag
    pData.F2_freq = F2_freq
    pData.F1_freq = F1_freq
    pData.DP_freq = DP_freq
    
    if DPOAEdataIn is None:
        DPOAEdataIn = DPOAEData(freqArray, ampArray, len(mic_data))
    
    DPOAEdataIn.F2Resp[freq_idx, amp_idx] = F2_freq_mag
    DPOAEdataIn.F1Resp[freq_idx, amp_idx] = F1_freq_mag
    DPOAEdataIn.DPResp[freq_idx, amp_idx] = DP_freq_mag
    DPOAEdataIn.noise_mean[freq_idx, amp_idx] = mic_fft_noise_mean
    DPOAEdataIn.noise_std[freq_idx, amp_idx] = mic_fft_noise_std
    DPOAEdataIn.t = t
    DPOAEdataIn.mic_data[freq_idx, amp_idx, :] = mic_data
    DPOAEdataIn.mic_freq = mic_freq
    if DPOAEdataIn.mic_fft_mag is None:
        DPOAEdataIn.mic_fft_mag = np.zeros((len(freqArray), len(ampArray), len(mic_fft_mag_log)))
        DPOAEdataIn.mic_fft_mag[:, :, :] = np.nan
        
    DPOAEdataIn.mic_fft_mag[freq_idx, amp_idx, :] = mic_fft_mag_log
    
    calcThreshold(DPOAEdataIn, freq_idx)
        
    return pData, DPOAEdataIn
    
    
# save the processed data as excel spreadsheet
def saveDPOAEDataXLS(DPOAEdata, trialDuration, ws, saveOpts):
    try:
        numFreq = len(DPOAEdata.freqArray)
        ws.write(2, 0, "Trial duration")
        ws.write(2, 1, "%0.3g" % trialDuration)
        #ws.write(3, 0, "Trial reps")
        #ws.write(3, 1, "%d" % trialReps)
       
        # writeExcelFreqAmpHeader(ws, freq, amp, row=0, col=1):
        row = 4
        ws.write(row, 0, 'DP Response')
        CMPCommon.writeExcelFreqAmpHeader(ws, DPOAEdata.freqArray, DPOAEdata.ampArray, row+1, 0)
        CMPCommon.writeExcel2DData(ws, DPOAEdata.DPResp, row+2, 1)

        row = row + numFreq + 2     
        ws.write(row, 0, 'F2 Response')
        CMPCommon.writeExcelFreqAmpHeader(ws, DPOAEdata.freqArray, DPOAEdata.ampArray, row+1, 0)
        CMPCommon.writeExcel2DData(ws, DPOAEdata.F2Resp, row+2, 1)

        row = row + numFreq + 2     
        ws.write(row, 0, 'F1 Response')
        CMPCommon.writeExcelFreqAmpHeader(ws, DPOAEdata.freqArray, DPOAEdata.ampArray, row+1, 0)
        CMPCommon.writeExcel2DData(ws, DPOAEdata.F1Resp, row+2, 1)

        row = row + numFreq + 2     
        ws.write(row, 0, 'Noise Mean')
        CMPCommon.writeExcelFreqAmpHeader(ws, DPOAEdata.freqArray, DPOAEdata.ampArray, row+1, 0)
        CMPCommon.writeExcel2DData(ws, DPOAEdata.noise_std, row+2, 1)
        
        row = row + numFreq + 2       
        ws.write(row, 0, 'Noise Stdev')
        CMPCommon.writeExcelFreqAmpHeader(ws, DPOAEdata.freqArray, DPOAEdata.ampArray, row+1, 0)
        CMPCommon.writeExcel2DData(ws, DPOAEdata.noise_mean, row+2, 1)

        freqArray = DPOAEdata.freqArray
        ampArray = DPOAEdata.ampArray
        row = row + numFreq + 1
        
        # save tracings if user has checked box        
        if saveOpts.saveMicData:
            ws.write(row, 0, 'Averaged Mic')
            row += 1
            ws.write(row, 1, 'Freq (kHz) / Amp')
            ws.write(row, 0, 't (ms)')
            col = 0
            row += 1
            r = row + 1
            for t in DPOAEdata.t:
                ws.write(r, col, t*1000)
                r += 1
            
#            ws.write(row+1, 1, 'Amp')
            col = 1
            for f_idx in range(0, len(freqArray)):
                for a_idx in range(0, len(ampArray)):
                    f = freqArray[f_idx]
                    a = ampArray[a_idx]
                    f_div_1k = f/1000
                    ws.write(row, col, "%0.3g" % f_div_1k + '/' + "%0.3g" % a)
                    r = row + 1
                    tr = DPOAEdata.mic_data[f_idx, a_idx, :]
                    tr = np.round(tr*1000)/1000
                    for pt in tr:
                        if not (np.isinf(pt) or np.isnan(pt)):
                            ws.write(r, col, pt)
                        r += 1
                    col += 1
            row = r
            
        if saveOpts.saveMicFFT:
            ws.write(row, 0, 'Mic FFT Mag (dB SPL)')
            row += 1
            ws.write(row, 1, 'Freq (kHz) / Amp')
            ws.write(row, 0, 'f (kHz)')
            col = 0
            row += 1
            r = row+1
            for f in DPOAEdata.mic_freq:
                ws.write(r, col, f/1000)
                r += 1
                
            col = 1
            for f_idx in range(0, len(freqArray)):
                for a_idx in range(0, len(ampArray)):
                    f = freqArray[f_idx]
                    a = ampArray[a_idx]
                    f_div_1k = f/1000
                    ws.write(row, col, "%0.3g" % f_div_1k + '/' + "%0.3g" % a)
                    tr = DPOAEdata.mic_fft_mag[f_idx, a_idx, :]
                    tr = np.round(tr*100)/100
                    r = row + 1
                    for pt in tr:
                        if not (np.isinf(pt) or np.isnan(pt)):
                            ws.write(r, col, pt)
                        r += 1
                    col += 1
    except:
        print('DPOAE.saveDPOAEDataXLS: Exception writing data to file')        
        traceback.print_exc()
        
        
# save the processed data of this protocol
def saveDPOAEData(DPOAEdata, trialDuration, trialReps, saveOpts, timeStr):
    f = open(filepath, 'a')
    
    try:
        f.write('[data]')
        f.write('\nType= DPOAE')
        f.write('\nTime= ' + timeStr)
        f.write('\nNote= ' + saveOpts.note)
        f.write("\nTrial duration= %0.3g" % trialDuration)
        f.write("\nTrial reps=  %0.3g" % trialReps)

        f.write("\nF= ")
        for freq in DPOAEdata.freqArray:
            f.write(" %0.3f" % freq)
        f.write("\nA= ")
        for a in DPOAEdata.ampArray:
            f.write(" %0.1f" % a)
    
        
        f.write('\nResp=')        
        for f_i in range(0, DPOAEdata.numFreq):
            for a_i in range(0, DPOAEdata.numAmp):
                f.write(' %0.5g' % DPOAEdata.DPResp[f_i, a_i])
                if a_i < DPOAEdata.numAmp-1:
                    f.write(',')
            f.write('\n')                
            
        f.write('\nNoise STD=')        
        for f_i in range(0, DPOAEdata.numFreq):
            for a_i in range(0, DPOAEdata.numAmp):
                    f.write(' %0.5g' % DPOAEdata.noise_std[f_i, a_i])
                    if a_i < DPOAEdata.numAmp-1:
                        f.write(',')
            f.write('\n')
            
        f.write('\nNoise Mean=')        
        for f_i in range(0, DPOAEdata.numFreq):
            for a_i in range(0, DPOAEdata.numAmp):
                    f.write(' %0.5g' % DPOAEdata.noise_mean[f_i, a_i])
                    if a_i < DPOAEdata.numAmp-1:
                        f.write(',')
            f.write('\n')
            
        if saveOpts.saveMicData:
            f.write('\nt= ')
            for t in DPOAEdata.t[0:-1]:
                f.write(' %0.5g,' % t)
            f.write(' %0.5g ' % DPOAEdata.t[-1])
                
            f.write('\nF\tA\tTracing')
            freqArray = DPOAEdata.freqArray
            ampArray = DPOAEdata.ampArray
            for f_idx in range(0, len(freqArray)):
                for a_idx in range(0, len(ampArray)):
                    f.write('\n%0.3f\t%0.1f\t' % (freqArray[f_idx], ampArray[f_idx]))
                    tr = DPOAEdata.tracings[f_idx, a_idx, :]
                    for pt in tr[0:-1]:
                        f.write(' %0.5g,' % pt)
                    f.write(' %0.5g' % tr[-1])
                        
        f.write('\n[/data]\n')
    except:
        print('DPOAE.saveDPOAEData: Exception writing data to file')        
        traceback.print_exc()
        
    f.close()
    
def saveDPOAEDataPickle(DPOAEdata, trialDuration, fileName, saveOpts, timeStr):
    DPOAEdata.trialDuration = trialDuration
    DPOAEdata.note = saveOpts.note
    DPOAEdata.timeStamp = timeStr
    mic_data = DPOAEdata.mic_data
    if not saveOpts.saveMicData:
        DPOAEdata.mic_data = None
        
    if not saveOpts.saveMicFFT:
        DPOAEdata.mic_fft_mag = None
        
    filepath = os.path.join(saveOpts.saveBaseDir, '%s.pickle' % fileName)
    f = open(filepath, 'wb')
    pickle.dump(DPOAEdata, f)
    f.close()

    DPOAEdata.mic_data = mic_data
    
   
def saveDPOAEDataFig(DPOAEdata, trialDuration, saveDir, plotName, timeStr):
    plt.figure(1, figsize=(14, 11), dpi=80)
    plt.clf()
    
    numFreq = len(DPOAEdata.freqArray)
    numAmp = len(DPOAEdata.ampArray)
    if numFreq == 1:
        DPOAEresp = DPOAEdata.DPResp[0, :]
        plt.plot(DPOAEdata.ampArray, 1e6*DPOAEresp,'-bo', label='Signal')
        noise_mean = DPOAEdata.noise_mean[0, :]
        noise_std = DPOAEdata.noise_std[0, :]
        plt.plot(DPOAEdata.ampArray,  1e6*(noise_mean + 3*noise_std), '-ro', label='Noise')   
        plt.xlabel('Amplitude (dB)')
        plt.legend(loc='upper left')
    else:
        # clr_lbls = ['b', 'g', 'r', 'c', 'm',  'y', 'k']
        nGraphs = numFreq + 1
        numRows = int(np.ceil(nGraphs ** 0.5))
        numCols = int(np.ceil(nGraphs / numRows))
        numSD = 2.5
        for n in range(0, numFreq):
            plt.subplot(numRows, numCols, n+1)
            DPresp = DPOAEdata.DPResp[n, :]
            noiseMean = DPOAEdata.noise_mean[n, :]
            noise = noiseMean + numSD*DPOAEdata.noise_std[n, :]
            plt.plot(DPOAEdata.ampArray, DPresp, '-bo', label='Signal')
            plt.hold('on')
            plt.plot(DPOAEdata.ampArray, noiseMean, ':r', label='Noise mean')
            plt.plot(DPOAEdata.ampArray, noise, '-rs', label='+ %g SD' % numSD)
            plt.xlabel('Amplitude (dB)', fontsize=10)
            if n == (numFreq-1):
                plt.legend(loc='upper left', fontsize=10)
            #plt.pcolormesh(t, f, Sxx)
            plt.ylabel('dB SPL', fontsize=10)
            plt.title('%0.2f kHz' % (DPOAEdata.freqArray[n]/1e3), x=0.15, fontsize=10)
            
    plt.show()
    fname = os.path.join(saveDir, plotName)
    plt.savefig(fname)
    
    return fname
    
def runDPOAE(appObj, testMode=False):
    print("runDPOAE")
    
    try:
        appObj.tabWidget.setCurrentIndex(2)
        appObj.doneFlag = False
        appObj.isCollecting = True
        # trigRate = octfpga.GetTriggerRate()
        audioHW = appObj.audioHW
        bioamp = appObj.bioamp
        outputRate = audioHW.DAQOutputRate
        inputRate = audioHW.DAQInputRate
    
        # freq_array2 = audioParams.freq[1, :]
        freqArray = appObj.getFrequencyArray()
        
        if testMode:
            testDataDir = os.path.join(appObj.basePath, 'exampledata', 'DPOAE')
    #        filePath = os.path.join(testDataDir, 'AudioParams.pickle')
    #        f = open(filePath, 'rb')
    #        audioParams = pickle.load(f)
    #        f.close()
        else:
            # freqArray = appObj.getFrequencyArray()
            i1 = appObj.DPOAE_freqLow_comboBox.currentIndex()
            i2 = appObj.DPOAE_freqHigh_comboBox.currentIndex()
            print("runDPOAE: i1= ", i1, "i2= ", i2)
    
            ampLow = appObj.DPOAE_ampLow_spinBox.value()
            ampHigh = appObj.DPOAE_ampHigh_spinBox.value()
            ampDelta = appObj.DPOAE_ampDelta_spinBox.value()
            
            # ampArray = np.arange(ampLow, ampHigh, ampDelta)
            #numSteps = np.floor((ampHigh - ampLow)/ampDelta) + 1
            #ampArray = np.linspace(ampLow, ampHigh, numSteps)
            ampArray = np.arange(ampLow, ampHigh, ampDelta)
            if ampArray[-1] != ampHigh:
                ampArray = np.hstack((ampArray, ampHigh))
            
            freqArray = freqArray[i1:i2+1]
    
        # numSpk = audioParams.getNumSpeakers()
        if not testMode:
            from DAQHardware import DAQHardware
            daq = DAQHardware()
    
        chanNamesIn= [ audioHW.mic_daqChan, bioamp.daqChan]
        micVoltsPerPascal = audioHW.micVoltsPerPascal
        trialDur = appObj.DPOAE_stimDuration_dblSpinBox.value() * 1e-3
        # nReps = appObj.DPOAEtrialReps_spinBox.value()
    
        # set input rate multiple the highest output frequency, a little more than Nyquest so stim frequency is more towards center 
        inputRate = 4*freqArray[-1]
        inputRate = outputRate / int(np.floor(outputRate / inputRate))  # pick closest input rate that evenly divides output rate
        
        frameNum = 0
        isSaveDirInit = False

        attenLines1 = audioHW.attenL_daqChan
        attenLines2 = audioHW.attenR_daqChan
        
        freq_idx = 0
        DPOAEdata = None
        numSpk = appObj.speaker_comboBox.currentIndex()+1
        chanNameOut = audioHW.speakerL_daqChan 
        if numSpk > 1:
            chanNameOut = [audioHW.speakerL_daqChan, audioHW.speakerR_daqChan ]
        print("runDPOAE numSpk=", numSpk)
        
        for freq in freqArray:
            sp1, sp2 = makeDPOAEOutput(freq, trialDur, audioHW)
            # spkOut = np.tile(spkOut_trial, nReps)
            
            npts = len(sp1)
            tOut = np.linspace(0, npts/outputRate, npts)
            print("runDPOAE npts=%d len(spkOut)= %d len(tOut)= %d" % (npts, len(sp1), len(tOut)))
            amp_idx = 0
            # ptsPerRep = inputRate
            
            for amp in ampArray:
                print("runDPOAE freq=" + repr(freq), " amp= ", + amp, " freq_idx= ", freq_idx, " amp_idx= ", amp_idx)
                vOut1, attenLvl1 = audioHW.getCalibratedOutputVoltageAndAttenLevel(freq, amp, 0)
                spkNum = numSpk - 1
                vOut2, attenLvl2 = audioHW.getCalibratedOutputVoltageAndAttenLevel(freq/1.22, amp, spkNum)
                if vOut1 > 0 and vOut2 > 0:
                    # attenSig = AudioHardware.makeLM1972AttenSig(0)
                    if not testMode:
                        if numSpk > 1:
                            AudioHardware.Attenuator.setLevel(attenLvl1, attenLines1)
                            AudioHardware.Attenuator.setLevel(attenLvl2, attenLines2)
                        else:
                            if attenLvl1 > attenLvl2:
                                dbDiff = attenLvl1 - attenLvl2
                                attenLvl1 = attenLvl2
                                vOut2 = vOut2*(10**(dbDiff/20))
                            elif attenLvl1 < attenLvl2:
                                dbDiff = attenLvl2 - attenLvl1
                                attenLvl2 = attenLvl1
                                vOut1 = vOut1*(10**(dbDiff/20))
                            AudioHardware.Attenuator.setLevel(attenLvl1, attenLines1)
                        # daq.sendDigOutDPOAEd(attenLines, attenSig)
                        # appObj.oct_hw.SetAttenLevel(0, attenLines)
                    
                    pl = appObj.DPOAE_output
                    pl.clear()
                    endIdx = int(5e-3 * outputRate)        # only plot first 5 ms
                    #pl.plot(t[0:endIdx], spkOut[0:endIdx], pen='b')
                    pl.plot(tOut, sp1 + sp2, pen='b')
                    
                    labelStyle = appObj.xLblStyle
                    pl.setLabel('bottom', 'Time', 's', **labelStyle)
                    labelStyle = appObj.yLblStyle
                    pl.setLabel('left', 'Output', 'V', **labelStyle)
                    
                    numInputSamples = int(inputRate*len(sp1)/outputRate)
                    
                    if testMode:
                        # mic_data = OCTCommon.loadRawData(testDataDir, frameNum, dataType=3)                    
                        pass
                    else:
        
                        # setup the output task
                        if numSpk > 1:
                            spkOut = np.vstack((vOut1*sp1, vOut2*sp2))
                        else:
                            spkOut = vOut1*sp1 + vOut2*sp2
                            
                        daq.setupAnalogOutput([chanNameOut], audioHW.daqTrigChanIn, int(outputRate), spkOut)
                        daq.startAnalogOutput()
                        
                        # setup the input task
                        daq.setupAnalogInput(chanNamesIn, audioHW.daqTrigChanIn, int(inputRate), numInputSamples) 
                        daq.startAnalogInput()
                    
                        # trigger the acquiisiton by sending ditital pulse
                        daq.sendDigTrig(audioHW.daqTrigChanOut)
                        
                        dataIn = daq.readAnalogInput()
                        mic_data = dataIn[0, :]
                        
                        mic_data = mic_data/micVoltsPerPascal
                    
                        daq.stopAnalogInput()
                        daq.stopAnalogOutput()
                        daq.clearAnalogInput()
                        daq.clearAnalogOutput()
                    
                    npts = len(mic_data)
                    t = np.linspace(0, npts/inputRate, npts)
                    pl = appObj.spCal_micInput
                    pl.clear()
                    pl.plot(t, mic_data, pen='b')
                    
                    labelStyle = appObj.xLblStyle
                    pl.setLabel('bottom', 'Time', 's', **labelStyle)
                    labelStyle = appObj.yLblStyle
                    pl.setLabel('left', 'Response', 'Pa', **labelStyle)
        
                    DPOAEptData, DPOAEdata = processDPOAEData(mic_data, freq, freq_idx, amp_idx, freqArray, ampArray, inputRate, DPOAEdata)
    
                    print("runDPOAE: plotting data")
                    plotDPOAEdata(appObj, DPOAEptData, DPOAEdata)
                    
                QtGui.QApplication.processEvents() # check for GUI events, such as button presses
                
                # if done flag, break out of loop
                if appObj.doneFlag:
                    break
                
                frameNum += 1
                amp_idx += 1
                
            # if done flag, break out of loop
            if appObj.doneFlag:
                break
            
            freq_idx += 1


        saveOpts = appObj.getSaveOpts()
        workbook = appObj.excelWB
        note = saveOpts.note
        number = appObj.DPOAEnumber
        name = 'DPOAE'
        d = datetime.datetime.now()
        timeStr = d.strftime('%H_%M_%S')
        
        saveOpts.saveMicData = appObj.DPOAE_saveMicData_checkBox.isChecked()
        saveOpts.saveMicFFT = appObj.DPOAE_saveMicFFT_checkBox.isChecked()
        saveDir = appObj.saveDir_lineEdit.text()
        
        plotName = 'DPOAE %d %s %s' % (number, timeStr, saveOpts.note)
        plotFilePath = saveDPOAEDataFig(DPOAEdata, trialDur, saveDir, plotName, timeStr)
        
        reply = QtGui.QMessageBox.question(appObj, 'Save', "Keep data?" , QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.Yes)
        if reply == QtGui.QMessageBox.Yes:
            excelWS = CMPCommon.initExcelSpreadsheet(workbook, name, number, timeStr, note)
            saveDPOAEDataXLS(DPOAEdata, trialDur, excelWS, saveOpts)
            #saveDPOAEData(DPOAEdata, trialDur, nReps, appObj.saveFileTxt_filepath, saveOpts, timeStr)
            
            saveDPOAEDataPickle(DPOAEdata, trialDur, plotName, saveOpts, timeStr)
            appObj.DPOAEnumber += 1                
            
        else:
            os.remove(plotFilePath)
            
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        QtGui.QMessageBox.critical (appObj, "Error", "Error during collection. Check command line output for details")           
        
    8# update the audio hardware speaker calibration                     
    appObj.isCollecting = False
    QtGui.QApplication.processEvents() # check for GUI events, such as button presses
    appObj.finishCollection()