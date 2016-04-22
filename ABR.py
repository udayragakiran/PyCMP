# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:31:37 2016

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

import matplotlib.pylab as mlab
import matplotlib.pylab as plt
import pickle

class ABRData:
    def __init__(self, freqArray, ampArray, ptsPerTrial):
        self.freqArray = freqArray
        
        numFreq = len(freqArray)
        numAmp = len(ampArray)
        
        self.numFreq = numFreq
        self.numAmp = numAmp
        
        self.ampArray = ampArray
        
        self.ABRResp = np.zeros((numFreq, numAmp))   # PP ABR response
        self.noise_std = np.zeros((numFreq, numAmp))
        self.tracings = np.zeros((numFreq, numAmp, ptsPerTrial))
        self.threshold = ampArray[-1]*np.ones(numFreq)
        
        self.ABRResp[:, :] = np.nan
        self.noise_std[:, :] = np.nan
        
class ABRParams:
    def __init__(self, appObj=None):
        if appObj is not None:
            self.trialDur = appObj.ABRtrialDuration_dblSpinBox.value() * 1e-3
            self.stimDur = appObj.ABRstimDuration_dblSpinBox.value() * 1e-3
            self.stimEnv = appObj.ABRstimEnvelope_dblSpinBox.value() * 1e-3
            self.stimOffset = appObj.ABRstimOffset_dblSpinBox.value() * 1e-3
            self.click = appObj.ABR_click_checkBox.isChecked()
            
            nReps = appObj.ABRtrialReps_spinBox.value()
            nReps = 2 * (nReps // 2)  # ensure number of reps is even number
            self.nReps = nReps

class ABRPointData:
    def __init__(self):
        self.mic_data_avg = None   # averaged mic response
        self.bioamp_data_avg = None   # averaged bioamp response, after filter
        
        # self.filtBioamp = None   # filtered bioamp response
        self.t = None   # time
        self.mic_fft_mag = None
        self.mic_fft_freq = None
        self.mic_freq_mag = None
        
        self.bioamp_PP_resp = None
        
def makeABROutput(freq, ABRparams, audioHW):
    outV = 1
    outputRate = audioHW.DAQOutputRate
    
#    stimEnv = 1e-3
#    stimOffset = 5e-3
    
    trialDur = ABRparams.trialDur
    stimDur = ABRparams.stimDur
    stimEnv = ABRparams.stimEnv
    stimOffset = ABRparams.stimOffset
    
    trialPts = np.ceil(trialDur * outputRate)
    stimPts = np.ceil(stimDur * outputRate)
    
    sig = np.zeros(trialPts)
    
    if ABRparams.click:
        stim = outV*np.ones(stimPts)
    else:
        t = np.linspace(0, stimDur, stimPts)
        stim = outV*np.sin(2*np.pi*freq*t)
    
    envPts = np.ceil(stimEnv * outputRate)
    envFcn = np.ones((stimPts))
    if ABRparams.click:
        env = np.linspace(0, 1, envPts)
    else:
        t = np.linspace(0, np.pi/2, envPts)
        env = np.sin(t) ** 2
        
    envFcn[0:envPts] = env
    envFcn[stimPts-envPts:] = 1 - env  # invert the envelope function
    stim = stim * envFcn
    
    offsetPt = np.round(stimOffset*outputRate)
    print('makeABRoutput: offsetPt= ', offsetPt, ' len(stim)= ', len(stim), ' stimPts= ', stimPts, ' len(sig)= ', len(sig))
    sig[offsetPt:(offsetPt+stimPts)] = stim
    
    return sig
    
import scipy
    
def calcThreshold(ABRdata, fIdx):
    sig = ABRdata.ABRResp[fIdx, :]
    noise = 5*ABRdata.noise_std[fIdx, :]
    print("ABR.calcThreshold sig=", sig, " noise=", noise)
    
    # remove nans
    nonnanidx = np.argwhere(~np.isnan(sig))
    print("ABR.calcThreshold nonnanidx=", nonnanidx)
    nonnanidx = nonnanidx[:, 0]  # acces sfirst element wince arguwhere returns tuple
    print("ABR.calcThreshold nonnanidx=", nonnanidx)
    sig = sig[nonnanidx]
    noise = noise[nonnanidx]
    
    print("ABR.calcThreshold sig > noise=", sig > noise)
    posABRidx = np.argwhere(sig > noise)  # boolean array if 
    posABRidx = posABRidx[:, 0]  # flatten to 1D array since argwhere returns 2DD array
    print("ABR.calcThreshold posABRidx=", posABRidx)
    if posABRidx.size == 0:  # no signal above threshold
        tHold = ABRdata.ampArray[-1]
        
    else:
        idx = posABRidx[0] 
        print("ABR.calcThreshold idx=", idx)
        
        if nonnanidx[idx] == 0:  # all signals above threshold - extrapolate to find threshold
            # tHold = ABRdata.ampArray[nonnanidx[idx]]
            a1 = ABRdata.ampArray[0]
            a2 = ABRdata.ampArray[1]
            ns = noise[idx]
            s1 = sig[idx]
            s2 = sig[idx+1]
            # linear fit last two points s = m*a + b
            m = (s2-s1)/(a2-a1)
            b = s2-m*a2
            # find intercept: n = m*a + b -> a = (n - b)/m
            a = (ns-b)/m
            tHold = a
            
        else: # interpolate to find response
            print("ABR.calcThreshold ampArray(nonnanidx)=", ABRdata.ampArray[nonnanidx])
            aIdx = nonnanidx[idx]
            print("ABR.calcThreshold aIdx=", aIdx)
            a1 = ABRdata.ampArray[aIdx-1]
            a2 = ABRdata.ampArray[aIdx]
            n1 = noise[idx-1]
            n2 = noise[idx]
            s1 = sig[idx-1]
            s2 = sig[idx]
            print("ABR.calcThreshold a1=", a1, " a2=", a2, " n1=", n1, " n2=", n2, " s1=", s1, " s2=", s2)
            a = np.linspace(a1, a2, 100)
            s = np.interp(a, (a1, a2), (s1, s2))   # signal interpolated 
            n = np.interp(a, (a1, a2), (n1, n2))   # noise interpolated
            print("ABR.calcThreshold a=", a)
            print("ABR.calcThreshold s=", s)
            print("ABR.calcThreshold n=", n)
            tHold_idx = np.argmin(np.abs(s - n))
            tHold = a[tHold_idx]
        
    ABRdata.threshold[fIdx] = tHold    
    
def plotABRdata(appObj, ABRptData, ABRdata):
    pl = appObj.ABR_bioampInput
    pl.clear()
    pl.plot(ABRptData.t, ABRptData.bioamp_data_avg, pen='b')
    labelStyle = appObj.xLblStyle
    pl.setLabel('bottom', 'Time', 's', **labelStyle)
    labelStyle = appObj.yLblStyle
    pl.setLabel('left', 'Resp', 'V', **labelStyle)
                
    pl = appObj.ABR_micInput
    pl.clear()
    # pl.plot(ABRptData.mic_fft_freq, ABRptData.mic_fft_mag, pen='b')
    pl.plot(ABRptData.t, ABRptData.mic_data_avg, pen='b')
    labelStyle = appObj.xLblStyle
    #pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
    pl.setLabel('bottom', 'Time', 's', **labelStyle)
    labelStyle = appObj.yLblStyle
    pl.setLabel('bottom', 'Response', 'Pa', **labelStyle)
    #pl.setLabel('left', 'dB SPL', 'dB', **labelStyle)
    freq = ABRdata.freqArray
#    if len(freq) == 1:
#        pl.setXRange(0.25*freq[0], 2*freq[-1])
#    else:
#        pl.setXRange(0.9*freq[0], 1.1*freq[-1])
    
    pl = appObj.ABR_response
    pl.clear()
    labelStyle = appObj.yLblStyle
    pl.setLabel('left', 'Response', 'V', **labelStyle)
    
    numFreq = len(ABRdata.freqArray)
    numAmp = len(ABRdata.ampArray)
    if numFreq == 1:
        labelStyle = appObj.xLblStyle
        pl.setLabel('bottom', 'Amplitude', 'dB', **labelStyle)

        ABRresp = ABRdata.ABRResp[0, :]
        pl.plot(ABRdata.ampArray, ABRresp, pen='b', symbol='o')
#        noise_mean = ABRdata.noise_mean[0, :]
        noise_std = ABRdata.noise_std[0, :]
        pl.plot(ABRdata.ampArray,  5*noise_std, pen='r', symbol='o')   
    else:
        labelStyle = appObj.xLblStyle
        pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
        
        for n in range(0, numAmp):
            ABRresp = ABRdata.ABRResp[:, n]
            pl.plot(ABRdata.freqArray, ABRresp, pen='b', symbol='o')
        
    
def processABRData(mic_data, bioamp_data, freq, freq_idx, amp_idx, freqArray, ampArray, inputRate, ABRdataIn, ABRparams):
    # print("SpeakerCalProtocol: processData: mic_data=" + repr(mic_data))
    # ensure data is 1D
    numpts = len(mic_data)
    ptsPerRep = numpts // ABRparams.nReps
    print("processABRData: numpts= %d ptsPerRep= %d" % (numpts, ptsPerRep))
    
    mic_data = np.reshape(mic_data, (ABRparams.nReps, ptsPerRep))
    # invert the mic data because the output was inverted
    for n in range(1, ABRparams.nReps, 2):
        mic_data[n, :] = -mic_data[n, :]
    bioamp_data = np.reshape(bioamp_data, (ABRparams.nReps, ptsPerRep))
    
    print("processABRData: mic_data.shape= ", mic_data.shape)
    
    mic_data = np.mean(mic_data, 0)
    bioamp_data = np.mean(bioamp_data, 0)
    
    print("processABRData: (after averaging) mic_data.shape= ", mic_data.shape)

    t = np.linspace(0, ptsPerRep/inputRate, ptsPerRep)
    
    print("processABRData: t.shape=", t.shape)
    zero_pad_factor = 2
    numfftpts = ptsPerRep*zero_pad_factor
    winfcn = np.hanning(ptsPerRep)
    win_corr = 2
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
    print("processABRData:  freq= %f "  % (freq))
    mic_stim_freq_mag = np.NAN
        
    try:            
        npts = zero_pad_factor
        mag_rgn = mic_fft_mag_log[fIdx-npts:fIdx+npts]
        fIdx = int(np.floor(freq*numfftpts/inputRate))
        print("processABRData: freq= %f" % (freq))
        
        maxIdx = np.argmax(mag_rgn)
        mic_stim_freq_mag = mag_rgn[maxIdx]
    except Exception as ex:
        traceback.print_exc()
        
    # process the bioamp data
    print("processABRData:  freq= %f "  % (freq))

    Wn = [300, 3000] # mouse
    # Wn = [500, 10000]
    Wn = np.array(Wn)/inputRate
    #Wn = [0.001, 0.01]
    #(b, a) = scipy.signal.butter(5, Wn=Wn, btype='bandpass')
    (b, a) = scipy.signal.iirfilter(2, Wn,  btype='bandpass', ftype='bessel')

    
    bioamp_data = bioamp_data - np.mean(bioamp_data)
    #b = scipy.signal.firwin(21, Wn)
    #a = [1.0]
    #bioamp_filt = scipy.signal.lfilter(b, a, bioamp_data - bioamp_data[0]) 
    # bioamp_filt = scipy.signal.lfilter(b, a, bioamp_data) 
    # bioamp_filt = bioamp_filt + bioamp_data[0]
    bioamp_filt = bioamp_data
    
    lagTime = 5e-3  # time to account for speaker and neural response delay
    idx1 = int(np.round(ABRparams.stimOffset*inputRate))
    idx2 = idx1 + int(np.round((ABRparams.stimDur + lagTime)*inputRate)) 
    sigRgn = bioamp_filt[idx1:idx2]
    noiseRgn = bioamp_filt[idx2:]
    bioamp_ptp = np.ptp(sigRgn)
    bioamp_noise_std = np.std(noiseRgn)
    
    # process the bioamp
    #Wn = [300/inputRate, 3e3/inputRate]
    #(b, a) = scipy.signal.butter(5, Wn=Wn, btype='bandpass')
    #filt_bioamp = scipy.signal.lfilter(b, a, bioamp_data) 
    
    print("processABRData: mic_stim_freq_mag= %f " % (mic_stim_freq_mag))
    pData = ABRPointData()
    pData.mic_data_avg = mic_data
    pData.bioamp_data_avg = bioamp_data
    
    #pData.filtBioamp = filt_bioamp
    pData.t = t
    pData.mic_fft_mag = mic_fft_mag_log
    pData.mic_fft_freq = mic_freq
    pData.mic_stim_freq_mag = mic_stim_freq_mag
    pData.bioamp_data_avg = bioamp_filt
    pData.bioamp_PP_resp = bioamp_ptp
    
    if ABRdataIn is None:
        ABRdataIn = ABRData(freqArray, ampArray, len(bioamp_data))
    
    ABRdataIn.ABRResp[freq_idx, amp_idx] = bioamp_ptp
    ABRdataIn.noise_std[freq_idx, amp_idx] = bioamp_noise_std
    ABRdataIn.tracings[freq_idx, amp_idx, :] = bioamp_filt
    ABRdataIn.t = t
    
    return pData, ABRdataIn
    
    
# save the processed data as excel spreadsheet
def saveABRDataXLS(ABRdata, ABRparams, ws, saveOpts):
    try:
        numFreq = len(ABRdata.freqArray)
        ws.write(2, 0, "Trial duration")
        ws.write(2, 1, "%0.3g" % ABRparams.trialDur)
        ws.write(3, 0, "Stim duration")
        ws.write(3, 1, "%0.3g" % ABRparams.stimDur)
        ws.write(4, 0, "Stim offset")
        ws.write(4, 1, "%0.3g" % ABRparams.stimOffset)
        ws.write(5, 0, "Stim envelope")
        ws.write(5, 1, "%0.3g" % ABRparams.stimEnv)
        ws.write(6, 0, "Trial reps")
        ws.write(6, 1, "%d" % ABRparams.nReps)
        ws.write(7, 0, "Click")
        ws.write(7, 1,  repr(ABRparams.click))
        
       
        # writeExcelFreqAmpHeader(ws, freq, amp, row=0, col=1):
        row = 9
        ws.write(row, 0, 'Response')
        CMPCommon.writeExcelFreqAmpHeader(ws, ABRdata.freqArray, ABRdata.ampArray, row+1, 0)
        CMPCommon.writeExcel2DData(ws, np.round(1e9*ABRdata.ABRResp)/1e3, row+2, 1)

        row = row + numFreq + 2     
        ws.write(row, 0, 'Noise SD')
        CMPCommon.writeExcelFreqAmpHeader(ws, ABRdata.freqArray, ABRdata.ampArray, row+1, 0)
        CMPCommon.writeExcel2DData(ws, np.round(1e9*ABRdata.noise_std)/1e3, row+2, 1)
        
#        row = row + numFreq + 2       
#        ws.write(row, 0, 'Noise Stdev')
#        CMPCommon.writeExcelFreqAmpHeader(ws, ABRdata.freqArray, ABRdata.ampArray, row+1, 0)
#        CMPCommon.writeExcel2DData(ws, ABRdata.noise_mean, row+2, 1)
        
        # save tracings if user has checked box        
        if saveOpts.saveTracings:
            row = row + numFreq + 3
            ws.write(row, 0, 't (ms)')
            col = 2
            for t in ABRdata.t:
                t = round(t*1e6)/1e3
                ws.write(row, col, t)
                col += 1
                
            row += 2
            ws.write(row, 0, 'Freq')
            ws.write(row, 1, 'Amp')
            ws.write(row, 2, 'Averaged Tracing')
            freqArray = ABRdata.freqArray
            ampArray = ABRdata.ampArray
            row += 1
            for f_idx in range(0, len(freqArray)):
                for a_idx in range(0, len(ampArray)):
                    f = freqArray[f_idx]
                    a = ampArray[a_idx]
                    ws.write(row, 0, f)
                    ws.write(row, 1, a)
                    tr = ABRdata.tracings[f_idx, a_idx, :]
                    col = 2
                    for pt in tr:
                        pt = round(pt*1e9)/1e3 # round to nearest nV and conver to uV 
                        ws.write(row, col, pt)
                        col += 1
                    row += 1

    except:
        print('ABR.saveABRDataXLS: Exception writing data to file')        
        traceback.print_exc()
        
        
# save the processed data of this protocol
def saveABRData(ABRdata, ABRparams, filepath, saveOpts, timeStr):
    f = open(filepath, 'a')
    
    try:
        f.write('[data]')
        f.write('\nType= ABR')
        f.write('\nTime= ' + timeStr)
        f.write('\nNote= ' + saveOpts.note)
        f.write("\nTrial duration= %0.3g" % ABRparams.trialDur)
        f.write("\nTrial reps=  %0.3g" % ABRparams.nReps)
        f.write("\nStim envelope %0.3g" % ABRparams.stimEnv)
        f.write("\nTrial reps %d" % ABRparams.nReps)
        f.write("\nClick %s" % repr(ABRparams.click))

        f.write("\nF= ")
        for freq in ABRdata.freqArray:
            f.write(" %0.3f" % freq)
        f.write("\nA= ")
        for a in ABRdata.ampArray:
            f.write(" %0.1f" % a)
    
        
        f.write('\nResp=')        
        for f_i in range(0, ABRdata.numFreq):
            for a_i in range(0, ABRdata.numAmp):
                f.write(' %0.5g' % ABRdata.ABRResp[f_i, a_i])
                if a_i < ABRdata.numAmp-1:
                    f.write(',')
            f.write('\n')                
            
        f.write('\nNoise STD=')        
        for f_i in range(0, ABRdata.numFreq):
            for a_i in range(0, ABRdata.numAmp):
                    f.write(' %0.5g' % ABRdata.noise_std[f_i, a_i])
                    if a_i < ABRdata.numAmp-1:
                        f.write(',')
            f.write('\n')
            
        f.write('\nNoise Mean=')        
        for f_i in range(0, ABRdata.numFreq):
            for a_i in range(0, ABRdata.numAmp):
                    f.write(' %0.5g' % ABRdata.noise_mean[f_i, a_i])
                    if a_i < ABRdata.numAmp-1:
                        f.write(',')
            f.write('\n')
            
        if saveOpts.saveTracings:
            f.write('\nt (ms)= ')
            for t in ABRdata.t[0:-1]:
                f.write(' %0.3f,' % round(1e6*t)/1e3)
            f.write(' %0.3f ' % round(1e6*ABRdata.t[-1])/1e3)
                
            f.write('\nF\tA\tTracing (uV)')
            freqArray = ABRdata.freqArray
            ampArray = ABRdata.ampArray
            for f_idx in range(0, len(freqArray)):
                for a_idx in range(0, len(ampArray)):
                    f.write('\n%0.3f\t%0.1f\t' % (freqArray[f_idx], ampArray[f_idx]))
                    tr = ABRdata.tracings[f_idx, a_idx, :]
                    tr = round(tr*1e9)/1e6
                    for pt in tr[0:-1]:
                        f.write(' %0.3f,' % pt)
                    f.write(' %0.3f' % tr[-1])
                        
        f.write('\n[/data]\n')
    except:
        print('ABR.saveABRData: Exception writing data to file')        
        traceback.print_exc()
        
    f.close()
    
def saveABRDataPickle(ABRdata, ABRparams, fileName, saveOpts, timeStr):
    ABRdata.params = ABRparams
    ABRdata.note = saveOpts.note
    ABRdata.timeStamp = timeStr
    tracings = ABRdata.tracings
    if not saveOpts.saveTracings:
        ABRdata.tracings = None
    
    filepath = os.path.join(saveOpts.saveBaseDir, '%s.pickle' % fileName)
    f = open(filepath, 'wb')
    pickle.dump(ABRdata, f)
    f.close()

    ABRdata.tracings = tracings   # need to reassign because we would be modifiying the ABRdata object 
    
   
def saveABRDataFig(ABRdata, ABRparams, saveDir, plotName, timeStr):
    plt.figure(1, figsize=(14, 11), dpi=80) 
    plt.clf()
    
    numFreq = len(ABRdata.freqArray)
    numAmp = len(ABRdata.ampArray)
    numSD = 5
    # clr_lbls = ['b', 'g', 'r', 'c', 'm',  'y', 'k']
    nGraphs = numFreq + 1
    numRows = int(np.ceil(nGraphs ** 0.5))
    numCols = int(np.ceil(nGraphs / numRows))
    for n in range(0, numFreq):
        plt.subplot(numRows, numCols, n+1)
        ABRresp = 1e6*ABRdata.ABRResp[n, :]
        noise = 1e6*numSD*ABRdata.noise_std[n, :]
        plt.plot(ABRdata.ampArray, ABRresp, '-bo', label='Signal')
        plt.hold('on')
        plt.plot(ABRdata.ampArray, noise, '-r', label='Noise (%0.1f SD)' % numSD)
        if n == (0):
            plt.ylabel('Resp PP (uV)', fontsize=10)
        if n == (numFreq-1):
            plt.legend(loc='upper left', fontsize=10)
            plt.xlabel('Amplitude (dB)', fontsize=10)
        #plt.pcolormesh(t, f, Sxx)
        
        plt.title('%0.2f kHz' % (ABRdata.freqArray[n]/1e3), x=0.15, fontsize=10)

    plt.show()
    fname = os.path.join(saveDir, plotName)
    plt.savefig(fname)
    
    plt.figure(2, figsize=(14, 11), dpi=80)
    plt.clf()
    
    # clr_lbls = ['b', 'g', 'r', 'c', 'm',  'y', 'k']
    t = ABRdata.t
    for n in range(0, numFreq):
        plt.subplot(numRows, numCols, n+1)
        offset = 0
        for a in range(numAmp):
            tr = 1e6*ABRdata.tracings[n, a, :]
            if a > 0:
                offset = offset + np.abs(np.min(tr))

            plt.plot(t*1e3, tr+offset, '-b', label='Signal')
            plt.hold('on')
            offset = offset + np.max(tr)
            
        if n == (0):
            plt.ylabel('Resp (uV)', fontsize=10)
        if n == (numFreq-1):
            plt.xlabel('Time (ms)', fontsize=10)
        #plt.pcolormesh(t, f, Sxx)
        
        plt.title('%0.2f kHz' % (ABRdata.freqArray[n]/1e3), x=0.15, fontsize=10)
    
    plt.show()
    fname = os.path.join(saveDir, plotName + ' tracings')
    plt.savefig(fname)
    
def runABR(appObj, testMode=False):
    print("runABR")
    
    appObj.tabWidget.setCurrentIndex(3)
    appObj.doneFlag = False
    appObj.isCollecting = True
    # trigRate = octfpga.GetTriggerRate()
    audioHW = appObj.audioHW
    bioamp = appObj.bioamp
    outputRate = audioHW.DAQOutputRate
    inputRate = audioHW.DAQInputRate

    # freq_array2 = audioParams.freq[1, :]
    freqArray = appObj.getFrequencyArray()
    ABRparams = ABRParams(appObj)
    
    if testMode:
        testDataDir = os.path.join(appObj.basePath, 'exampledata', 'Speaker Calibration')
#        filePath = os.path.join(testDataDir, 'AudioParams.pickle')
#        f = open(filePath, 'rb')
#        audioParams = pickle.load(f)
#        f.close()
    else:
        # freqArray = appObj.getFrequencyArray()
        i1 = appObj.ABR_freqLow_comboBox.currentIndex()
        i2 = appObj.ABR_freqHigh_comboBox.currentIndex()
        print("runABR: i1= ", i1, "i2= ", i2)
        ampLow = appObj.ABRampLow_spinBox.value()
        ampHigh = appObj.ABRampHigh_spinBox.value()
        ampDelta = appObj.ABRampDelta_spinBox.value()
        
        # ampArray = np.arange(ampLow, ampHigh, ampDelta)
        #numSteps = np.floor((ampHigh - ampLow)/ampDelta) + 1
        #ampArray = np.linspace(ampLow, ampHigh, numSteps)
        if ampLow == ampHigh:
            ampArray = np.array([ampLow])
        else:
            ampArray = np.arange(ampLow, ampHigh, ampDelta)
            if ampArray[-1] != ampHigh:
                ampArray = np.hstack((ampArray, ampHigh))
        
        freqArray = freqArray[i1:i2+1]
    
    if ABRparams.click:
        freqArray = freqArray[0:1]  # only use single freqeucny
        clickRMS = appObj.ABRclick_RMS
        
    # numSpk = audioParams.getNumSpeakers()
    if not testMode:
        from DAQHardware import DAQHardware
        daq = DAQHardware()

    chanNamesIn= [ audioHW.mic_daqChan, bioamp.daqChan]
    micVoltsPerPascal = audioHW.micVoltsPerPascal
    
    

    # set input rate to three times the highest output frequency, to allow plus a 
    
#    inputRate = 3*freqArray[-1]
#    print("runABR: outputRate= ", outputRate, " inputRate= ", inputRate)
#    inputRate = np.max((inputRate, 6e3))   # inpute rate should be at least 6 kHz because ABR responses occur 0.3 - 3 kHz
#    inputRate = outputRate / int(np.floor(outputRate / inputRate))  # pick closest input rate that evenly divides output rate
#    print("runABR: inputRate(final)= ", inputRate)
    
    try:
        frameNum = 0
        numFrames = len(freqArray)*len(ampArray)
        isSaveDirInit = False
        chanNameOut = audioHW.speakerL_daqChan 
        attenLines = audioHW.attenL_daqChan
        
        freq_idx = 0
        ABRdata = None
        appObj.status_label.setText("Running")
        appObj.progressBar.setValue(0)
        
        for freq in freqArray:
            spkOut_trial = makeABROutput(freq, ABRparams, audioHW)
            npts = len(spkOut_trial)
            spkOut = np.tile(spkOut_trial, ABRparams.nReps)
            # invert every other trial, necessary for ABR/CAP output 
            for n in range(1, ABRparams.nReps, 2):
                idx1 = n*npts
                idx2 = (n+1)*npts
                spkOut[idx1:idx2] = -spkOut[idx1:idx2]
#            plt.figure(5)
#            plt.clf()
#            plt.plot(spkOut)
            tOut = np.linspace(0, npts/outputRate, npts)
            print("runABR npts=%d len(spkOut_trial)= %d len(tOut)= %d" % (npts, len(spkOut_trial), len(tOut)))
            amp_idx = 0
            ptsPerRep = int(inputRate*ABRparams.trialDur)
            
            for amp in ampArray:
                print("runABR freq=" + repr(freq), " amp= ", + amp, " freq_idx= ", freq_idx, " amp_idx= ", amp_idx)
                if ABRparams.click:
                    clickRMS = appObj.ABRclick_RMS
                    attenLvl = 0
                    vOut = 10**((amp - clickRMS)/20)
                    minV = audioHW.speakerOutputRng[0]
                    if vOut < minV:
                        attenLvl = int(round(20*np.log10(minV/vOut)))
                        vOut = minV
                else:
                    vOut, attenLvl = audioHW.getCalibratedOutputVoltageAndAttenLevel(freq, amp, 0)
                
                print("runABR vOut= ", vOut, " atenLvl=", attenLvl)
                
                if vOut > audioHW.speakerOutputRng[1]:
                    print("runABR vOut= ", vOut, "  out of range")
                    continue
                elif attenLvl > audioHW.maxAtten:
                    print("runABR attenLvl= ", attenLvl, "  gerater than maximum attenuation")
                    continue
                    
                # attenSig = AudioHardware.makeLM1972AttenSig(0)
                if not testMode:
                    AudioHardware.Attenuator.setLevel(attenLvl, attenLines)
                    # daq.sendDigOutABRd(attenLines, attenSig)
                    # appObj.oct_hw.SetAttenLevel(0, attenLines)
                
                pl = appObj.ABR_output
                pl.clear()
                endIdx = int(5e-3 * outputRate)        # only plot first 5 ms
                #pl.plot(t[0:endIdx], spkOut[0:endIdx], pen='b')
                pl.plot(tOut, spkOut_trial, pen='b')
                
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Time', 's', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Output', 'V', **labelStyle)
                
                numInputSamples = ABRparams.nReps*int(inputRate*len(spkOut_trial)/outputRate)
                
                if testMode:
                    # mic_data = OCTCommon.loadRawData(testDataDir, frameNum, dataType=3)                    
                    pass
                else:
    
                    # setup the output task
                    daq.setupAnalogOutput([chanNameOut], audioHW.daqTrigChanIn, int(outputRate), vOut*spkOut)
                    daq.startAnalogOutput()
                    
                    # setup the input task
                    daq.setupAnalogInput(chanNamesIn, audioHW.daqTrigChanIn, int(inputRate), numInputSamples) 
                    daq.startAnalogInput()
                
                    # trigger the acquiisiton by sending ditital pulse
                    daq.sendDigTrig(audioHW.daqTrigChanOut)
                    
                    timeout = numInputSamples/inputRate + 2
                    dataIn = daq.readAnalogInput(timeout)
                    mic_data = dataIn[0, :]
                    bioamp_data = dataIn[1, :]
                    
                    mic_data = mic_data/micVoltsPerPascal
                    bioamp_data = bioamp_data/bioamp.gain
                
                    daq.waitDoneInput()
                    daq.stopAnalogInput()
                    daq.clearAnalogInput()
                    
                    daq.waitDoneOutput(stopAndClear=True)
                
#                npts = len(mic_data)
#                t = np.linspace(0, npts/inputRate, npts)
#                pl = appObj.ABR_micInput
#                pl.clear()
#                pl.plot(t, mic_data, pen='b')
#                
#                labelStyle = appObj.xLblStyle
#                pl.setLabel('bottom', 'Time', 's', **labelStyle)
#                labelStyle = appObj.yLblStyle
#                pl.setLabel('left', 'Response', 'Pa', **labelStyle)
    
    # def processABRData(mic_data, bioamp_data, nReps, freq, amp_idx, inputRate, ABRdataIn):            
                ABRptData, ABRdata = processABRData(mic_data, bioamp_data, freq, freq_idx, amp_idx, freqArray, ampArray, inputRate, ABRdata, ABRparams)

                print("runABR: plotting data")
                plotABRdata(appObj, ABRptData, ABRdata)
                
    #                if appObj.getSaveState():
    #                    if not isSaveDirInit:
    #                        saveDir = OCTCommon.initSaveDir(saveOpts, 'Speaker Calibration', audioParams=audioParams)
    #                        isSaveDirInit = True
    #    
    #                    if saveOpts.saveRaw:
    #                        OCTCommon.saveRawData(mic_data, saveDir, frameNum, dataType=3)
                idx1 = round(inputRate*ABRparams.stimOffset)
                idx2 = idx1 + round(inputRate*ABRparams.stimDur)
                
                mic_data = mic_data[idx1:idx2] 
                rms = np.mean(mic_data ** 2) ** 0.5
                rms = 20*np.log10(rms/2e-5)
                
                appObj.ABR_rms_label.setText("%0.1f dB" % rms)                    
                
                QtGui.QApplication.processEvents() # check for GUI events, such as button presses
                
                # if done flag, break out of loop
                if appObj.doneFlag:
                    break
                
                frameNum += 1
                amp_idx += 1
                appObj.progressBar.setValue(frameNum/numFrames)
                
            # if done flag, break out of loop
            if appObj.doneFlag:
                break
            
            freq_idx += 1


        saveOpts = appObj.getSaveOpts()
        workbook = appObj.excelWB
        note = saveOpts.note
        number = appObj.ABRnumber
        name = 'ABR'
        d = datetime.datetime.now()
        timeStr = d.strftime('%H_%M_%S')
        excelWS = CMPCommon.initExcelSpreadsheet(workbook, name, number, timeStr, note)
    
        appObj.ABRnumber += 1                
        #saveOpts.saveTracings = appObj.ABR_saveTracings_checkBox.isChecked()
        saveOpts.saveTracings = True
        saveDir = appObj.saveDir_lineEdit.text()
        saveABRDataXLS(ABRdata, ABRparams, excelWS, saveOpts)
        #saveABRData(ABRdata, trialDur, nReps, appObj.saveFileTxt_filepath, saveOpts, timeStr)
        
        plotName = 'ABR %d %s %s' % (number, timeStr, saveOpts.note)
        saveABRDataFig(ABRdata, ABRparams, saveDir, plotName, timeStr)
        saveABRDataPickle(ABRdata, ABRparams, plotName, saveOpts, timeStr)
            
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        QtGui.QMessageBox.critical (appObj, "Error", "Error during collection. Check command line output for details")           
        
    # update the audio hardware speaker calibration                     
    appObj.isCollecting = False
    QtGui.QApplication.processEvents() # check for GUI events, such as button presses
    appObj.finishCollection()


def calibrateClick(appObj, testMode=False):
    print("ABR.calibrateClick")
    
    appObj.tabWidget.setCurrentIndex(3)
    appObj.doneFlag = False
    appObj.isCollecting = True
    # trigRate = octfpga.GetTriggerRate()
    audioHW = appObj.audioHW
    outputRate = audioHW.DAQOutputRate
    inputRate = audioHW.DAQInputRate

    # freq_array2 = audioParams.freq[1, :]
    freqArray = appObj.getFrequencyArray()
    
    if testMode:
        testDataDir = os.path.join(appObj.basePath, 'exampledata', 'Speaker Calibration')
#        filePath = os.path.join(testDataDir, 'AudioParams.pickle')
#        f = open(filePath, 'rb')
#        audioParams = pickle.load(f)
#        f.close()
    else:
        pass

    # numSpk = audioParams.getNumSpeakers()
    if not testMode:
        from DAQHardware import DAQHardware
        daq = DAQHardware()

    chanNamesIn= [ audioHW.mic_daqChan]
    micVoltsPerPascal = audioHW.micVoltsPerPascal
    ABRparams = ABRParams(appObj)
    ABRparams.click = True
    ABRparams.nReps = 20
    print("ABR.calibrateClick ABRparams=", ABRparams.__dict__)
    # set input rate to three times the highest output frequency, to allow plus a 
    
#    inputRate = 3*freqArray[-1]
#    print("runABR: outputRate= ", outputRate, " inputRate= ", inputRate)
#    inputRate = np.max((inputRate, 6e3))   # inpute rate should be at least 6 kHz because ABR responses occur 0.3 - 3 kHz
#    inputRate = outputRate / int(np.floor(outputRate / inputRate))  # pick closest input rate that evenly divides output rate
#    print("runABR: inputRate(final)= ", inputRate)
    
    try:
        chanNameOut = audioHW.speakerL_daqChan 
        attenLines = audioHW.attenL_daqChan
        
        spkOut_trial = makeABROutput(4e3, ABRparams, audioHW)
        spkOut = np.tile(spkOut_trial, ABRparams.nReps)
        npts = len(spkOut_trial)
        tOut = np.linspace(0, npts/outputRate, npts)
            
        # attenSig = AudioHardware.makeLM1972AttenSig(0)
        if not testMode:
            AudioHardware.Attenuator.setLevel(0, attenLines)
                
        pl = appObj.ABR_output
        pl.clear()
        endIdx = int(5e-3 * outputRate)        # only plot first 5 ms
        #pl.plot(t[0:endIdx], spkOut[0:endIdx], pen='b')
        pl.plot(tOut, spkOut_trial, pen='b')
                
        labelStyle = appObj.xLblStyle
        pl.setLabel('bottom', 'Time', 's', **labelStyle)
        labelStyle = appObj.yLblStyle
        pl.setLabel('left', 'Output', 'V', **labelStyle)
                
        numInputSamples = int(inputRate*len(spkOut)/outputRate)
        
        if testMode:
            # mic_data = OCTCommon.loadRawData(testDataDir, frameNum, dataType=3)                    
            pass
        else:
            # setup the output task
            daq.setupAnalogOutput([chanNameOut], audioHW.daqTrigChanIn, int(outputRate), spkOut)
            daq.startAnalogOutput()
            
            # setup the input task
            daq.setupAnalogInput(chanNamesIn, audioHW.daqTrigChanIn, int(inputRate), numInputSamples) 
            daq.startAnalogInput()
        
            # trigger the acquiisiton by sending ditital pulse
            daq.sendDigTrig(audioHW.daqTrigChanOut)
            
            timeout = numInputSamples/inputRate + 2
            dataIn = daq.readAnalogInput(timeout)
            mic_data = dataIn[0, :]
            
            mic_data = mic_data/micVoltsPerPascal
        
            daq.waitDoneInput()
            daq.stopAnalogInput()
            daq.clearAnalogInput()
            
            daq.waitDoneOutput(stopAndClear=True)
        
        print("ABR.calibrateClick: plotting data")
        npts = len(mic_data)
        
        # reshape and average the mic data
        ptsPerRep = npts // ABRparams.nReps
        mic_data = np.reshape(mic_data, (ABRparams.nReps, ptsPerRep))
        mic_data = np.mean(mic_data, 0)
        
        # plot mic data
        npts = len(mic_data)
        t = np.linspace(0, npts/inputRate, npts)
        pl = appObj.ABR_micInput
        pl.clear()
        pl.plot(t, mic_data, pen='b')
        
        labelStyle = appObj.xLblStyle
        pl.setLabel('bottom', 'Time', 's', **labelStyle)
        labelStyle = appObj.yLblStyle
        pl.setLabel('left', 'Response', 'Pa', **labelStyle)
        
        idx1 = round(inputRate*ABRparams.stimOffset)
        idx2 = idx1 + round(inputRate*ABRparams.stimDur)
        
        mic_data = mic_data[idx1:idx2] 
        # apply high pass filter to get rid of LF components
#        (b, a) = scipy.signal.butter(5, 100/inputRate, 'high')
#        mic_data = scipy.signal.lfilter(b, a, mic_data) 

        rms = np.mean(mic_data ** 2) ** 0.5
        rms = 20*np.log10(rms/2e-5)
        appObj.ABRclick_RMS = rms
        
        appObj.ABR_rms_label.setText("%0.1f dB" % rms)
        print("ABR.calibrateClick: RMS= ", rms)
        
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        QtGui.QMessageBox.critical (appObj, "Error", "Error during collection. Check command line output for details")           
        
    appObj.isCollecting = False
    QtGui.QApplication.processEvents() # check for GUI events, such as button presses
    appObj.finishCollection()
    
    
if __name__ == "__main__":
    F = np.array([4, 8, 16, 32])
    #A = np.arange(10, 90, 10)
    # A = np.arange(50, 85, 5)
    #A = np.arange(40, 85, 5)
    #A = np.arange(40, 50, 5)
    A = np.arange(60, 70, 5)
    abrdata = ABRData(F, A, 100)
    
    #abrdata.ABRResp[0, :] = np.array([np.nan, 0.4, 0.5, 1.2, 1.5, 2.2, 0.9, 3.1])
    # abrdata.noise_std[0, :] = 0.2
    #abrdata.noise_std[0, :] = np.array([np.nan, 0.2, 0.23, 0.17, 0.25, 0.18, 0.23, 0.2])
    # Threshold (0.5 uv)
    abrdata.ABRResp[0, :] = np.array([0.352,	0.947])
    abrdata.noise_std[0, :] = 0.1    
    calcThreshold(abrdata, 0)
    print("threshold= ", abrdata.threshold[0])