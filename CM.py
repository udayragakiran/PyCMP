# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:09:56 2016

@author: OHNS

CM.py: routines for running Cochlear Microphonic, intended to be ysed with PyCMP


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

class CMData:
    def __init__(self, freqArray, ampArray, ptsPerTrial):
        self.freqArray = freqArray
        
        numFreq = len(freqArray)
        numAmp = len(ampArray)
        
        self.numFreq = numFreq
        self.numAmp = numAmp
        
        self.ampArray = ampArray
        
        self.CMResp = np.zeros((numFreq, numAmp))
        self.noise_std = np.zeros((numFreq, numAmp))
        self.noise_mean = np.zeros((numFreq, numAmp))
        self.tracings = np.zeros((numFreq, numAmp, ptsPerTrial))
        self.micTracings = np.zeros((numFreq, numAmp, ptsPerTrial))
        
        self.CMResp[:, :] = np.nan
        self.noise_std[:, :] = np.nan
        self.noise_mean[:, :] = np.nan
        

class CMPointData:
    def __init__(self):
        self.mic_data_avg = None   # averaged mic response
        self.bioamp_data_avg = None   # averaged bioamp response
        
        # self.filtBioamp = None   # filtered bioamp response
        self.t = None   # time
        self.mic_fft_mag = None
        self.mic_fft_freq = None
        self.mic_freq_mag = None
        
        self.bioamp_fft_mag = None
        self.bioamp_fft_freq = None
        self.bioamp_freq_mag = None
        
        
def makeCMOutput(freq, trialDur, stimOffset, audioHW):
    outV = 1
    outputRate = audioHW.DAQOutputRate
    
    trialPts = np.ceil(trialDur * outputRate)
    stimDur = trialDur - stimOffset
    stimPts = np.ceil(stimDur * outputRate)
    offsetPt = trialPts - stimPts
    stimEnv = 1e-3
    envPts = np.ceil(stimEnv * outputRate)
    t = np.linspace(0, stimDur, stimPts)
    sig = np.zeros(trialPts)
    stimSig = outV*np.sin(2*np.pi*freq*t)
    envFcn = np.ones((stimPts))
    envFcn[0:envPts] = np.linspace(0, 1, envPts)
    envFcn[stimPts-envPts:] = np.linspace(1, 0, envPts)
    stimSig = stimSig * envFcn
    sig[offsetPt:] = stimSig
    
    return sig
    
def plotCMdata(appObj, CMptData, CMdata):
    pl = appObj.CM_micPlot
    pl.clear()
    pl.plot(CMptData.t, CMptData.mic_data_avg, pen='b')
    labelStyle = appObj.xLblStyle
    pl.setLabel('bottom', 'Time', 's', **labelStyle)
    labelStyle = appObj.yLblStyle
    pl.setLabel('left', 'Resp', 'Pa', **labelStyle)
    
    pl = appObj.CM_bioampPlot
    pl.clear()
    pl.plot(CMptData.t, CMptData.bioamp_data_avg, pen='b')
    labelStyle = appObj.xLblStyle
    pl.setLabel('bottom', 'Time', 's', **labelStyle)
    labelStyle = appObj.yLblStyle
    pl.setLabel('left', 'Resp', 'V', **labelStyle)
                
    pl = appObj.CM_bioampFFTPlot
    pl.clear()
    pl.plot(CMptData.bioamp_fft_freq, CMptData.bioamp_fft_mag, pen='b')
    labelStyle = appObj.xLblStyle
    pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
    labelStyle = appObj.yLblStyle
    pl.setLabel('left', 'Magnitude', 'V', **labelStyle)
    freq = CMdata.freqArray
    if len(freq) == 1:
        pl.setXRange(0.25*freq[0], 2*freq[-1])
    else:
        pl.setXRange(0.9*freq[0], 1.1*freq[-1])
    
    pl = appObj.CM_responsePlot
    pl.clear()
    labelStyle = appObj.yLblStyle
    pl.setLabel('left', 'Response', 'V', **labelStyle)
    
    numFreq = len(CMdata.freqArray)
    numAmp = len(CMdata.ampArray)
    if numFreq == 1:
        labelStyle = appObj.xLblStyle
        pl.setLabel('bottom', 'Amplitude', 'dB', **labelStyle)

        CMresp = CMdata.CMResp[0, :]
        pl.plot(CMdata.ampArray, CMresp, pen='b', symbol='o')
        noise_mean = CMdata.noise_mean[0, :]
        noise_std = CMdata.noise_std[0, :]
        pl.plot(CMdata.ampArray,  noise_mean + 3*noise_std, pen='r', symbol='o')   
    else:
        labelStyle = appObj.xLblStyle
        pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
        
        for n in range(0, numAmp):
            CMresp = CMdata.CMResp[:, n]
            pl.plot(CMdata.freqArray, CMresp, pen='b', symbol='o')
        
    
    
    
    
def processCMData(mic_data, bioamp_data, nReps, freq, freq_idx, amp_idx, freqArray, ampArray, inputRate, CMdataIn):
    # print("SpeakerCalProtocol: processData: mic_data=" + repr(mic_data))
    # ensure data is 1D
    numpts = len(mic_data)
    ptsPerRep = numpts // nReps
    print("processCMData: numpts= %d ptsPerRep= %d" % (numpts, ptsPerRep))
    
    mic_data = np.reshape(mic_data, (nReps, ptsPerRep))
    bioamp_data = np.reshape(bioamp_data, (nReps, ptsPerRep))
    
    print("processCMData: mic_data.shape= ", mic_data.shape)
    
    mic_data = np.mean(mic_data, 0)
    bioamp_data = np.mean(bioamp_data, 0)
    
    print("processCMData: (after averaging) mic_data.shape= ", mic_data.shape)

    t = np.linspace(0, ptsPerRep/inputRate, ptsPerRep)
    
    print("processCMData: t.shape=", t.shape)
    zero_pad_factor = 2
    numfftpts = ptsPerRep*zero_pad_factor
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
    print("processCMData:  freq= %f "  % (freq))

    mic_stim_freq_mag = np.NAN
        
    try:            
        npts = zero_pad_factor
        mag_rgn = mic_fft_mag_log[fIdx-npts:fIdx+npts]
        fIdx = int(np.floor(freq*numfftpts/inputRate))
        print("processCMData: freq= %f" % (freq))
        
        maxIdx = np.argmax(mag_rgn)
        mic_stim_freq_mag = mag_rgn[maxIdx]
    except Exception as ex:
        traceback.print_exc()
        
    # process the bioamp data
    zero_pad_factor = 2
    numfftpts = ptsPerRep*zero_pad_factor
    winfcn = np.hanning(ptsPerRep)
    win_corr = 2  # magnitude correction for hanning window
    bioamp_fft = np.fft.fft(winfcn*bioamp_data, numfftpts)
    endIdx = np.ceil(numfftpts/2)
    bioamp_fft = bioamp_fft[0:endIdx]
    bioamp_fft_mag = win_corr*2*np.abs(bioamp_fft)*win_corr/ptsPerRep
    
    bioamp_fft_freq = np.linspace(0, inputRate/2, endIdx)
    fIdx = int(np.floor(freq*numfftpts/inputRate))
    print("processCMData:  freq= %f "  % (freq))

    bioamp_stim_freq_mag = np.NAN
        
    try:            
        npts = zero_pad_factor
        mag_rgn = bioamp_fft_mag[fIdx-npts:fIdx+npts]
        fIdx = int(np.floor(freq*numfftpts/inputRate))
        print("processCMData: freq= %f" % (freq))
        
        maxIdx = np.argmax(mag_rgn)
        bioamp_stim_freq_mag = mag_rgn[maxIdx]
        
        noise_rgn = bioamp_fft_mag[fIdx+npts:fIdx+npts+100]
        bioamp_fft_noise_std = np.std(noise_rgn)
        bioamp_fft_noise_mean = np.mean(noise_rgn)
    except Exception as ex:
        traceback.print_exc()        
    
    # process the bioamp
    #Wn = [300/inputRate, 3e3/inputRate]
    #(b, a) = scipy.signal.butter(5, Wn=Wn, btype='bandpass')
    #filt_bioamp = scipy.signal.lfilter(b, a, bioamp_data) 
    
    print("processCMData: mic_stim_freq_mag= %f " % (mic_stim_freq_mag))
    pData = CMPointData()
    pData.mic_data_avg = mic_data
    pData.bioamp_data_avg = bioamp_data
    
    #pData.filtBioamp = filt_bioamp
    pData.t = t
    pData.mic_fft_mag = mic_fft_mag_log
    pData.mic_fft_freq = mic_freq
    pData.mic_stim_freq_mag = mic_stim_freq_mag
    
    pData.bioamp_fft_freq = bioamp_fft_freq
    pData.bioamp_fft_mag = bioamp_fft_mag
    pData.bioamp_stim_freq_mag = bioamp_stim_freq_mag
    
    if CMdataIn is None:
        CMdataIn = CMData(freqArray, ampArray, len(bioamp_data))
    
    CMdataIn.CMResp[freq_idx, amp_idx] = bioamp_stim_freq_mag
    CMdataIn.noise_mean[freq_idx, amp_idx] = bioamp_fft_noise_mean
    CMdataIn.noise_std[freq_idx, amp_idx] = bioamp_fft_noise_std
    CMdataIn.tracings[freq_idx, amp_idx, :] = bioamp_data
    CMdataIn.micTracings[freq_idx, amp_idx, :] = mic_data
    CMdataIn.t = t
    
    return pData, CMdataIn
    
    
# save the processed data as excel spreadsheet
def saveCMDataXLS(CMdata, trialDuration, trialReps, ws, saveOpts):
    try:
        numFreq = len(CMdata.freqArray)
        ws.write(2, 0, "Trial duration")
        ws.write(2, 1, "%0.3g" % trialDuration)
        ws.write(3, 0, "Trial reps")
        ws.write(3, 1, "%d" % trialReps)
       
        # writeExcelFreqAmpHeader(ws, freq, amp, row=0, col=1):
        row = 6
        ws.write(row, 0, 'Response  (uV)')
        CMPCommon.writeExcelFreqAmpHeader(ws, CMdata.freqArray, CMdata.ampArray, row+1, 0)
        data = CMdata.CMResp * 1e9  
        data = np.round(data)/1e3 # convert to uV
        CMPCommon.writeExcel2DData(ws, data, row+2, 1)

        row = row + numFreq + 2     
        ws.write(row, 0, 'Noise Mean (uV)')
        CMPCommon.writeExcelFreqAmpHeader(ws, CMdata.freqArray, CMdata.ampArray, row+1, 0)
        data = CMdata.noise_mean * 1e9  
        data = np.round(data)/1e3 # convert to uV
        CMPCommon.writeExcel2DData(ws, data, row+2, 1)
        
        row = row + numFreq + 2       
        ws.write(row, 0, 'Noise Stdev (uV)')
        CMPCommon.writeExcelFreqAmpHeader(ws, CMdata.freqArray, CMdata.ampArray, row+1, 0)
        data = CMdata.noise_std * 1e9  
        data = np.round(data)/1e3 # round and convert to uV
        CMPCommon.writeExcel2DData(ws, CMdata.noise_mean, row+2, 1)
        
        # save tracings if user has checked box        
        if saveOpts.saveTracings:
            row = row + numFreq + 3
            ws.write(row, 0, 't (ms)')
            col = 2
            for t in CMdata.t:
                t = t*1e6 
                t = np.round(t)/1e3 # convert to ms
                ws.write(row, col, t)
                col += 1
                
            row += 2
            ws.write(row, 0, 'Freq')
            ws.write(row, 1, 'Amp')
            ws.write(row, 2, 'Averaged Bioamp Tracing (uV)')
            freqArray = CMdata.freqArray
            ampArray = CMdata.ampArray
            row += 1
            for f_idx in range(0, len(freqArray)):
                for a_idx in range(0, len(ampArray)):
                    f = freqArray[f_idx]
                    a = ampArray[a_idx]
                    ws.write(row, 0, f)
                    ws.write(row, 1, a)
                    tr = CMdata.tracings[f_idx, a_idx, :]
                    tr = tr*1e9
                    tr = np.round(tr)/1e3  # round to nearest nv and conver tto uV
                    col = 2
                    for pt in tr:
                        ws.write(row, col, pt)
                        col += 1
                    row += 1
                    
            row += 2
            ws.write(row, 0, 'Freq')
            ws.write(row, 1, 'Amp')
            ws.write(row, 2, 'Averaged Mic Tracing (uPa)')
            freqArray = CMdata.freqArray
            ampArray = CMdata.ampArray
            row += 1
            for f_idx in range(0, len(freqArray)):
                for a_idx in range(0, len(ampArray)):
                    f = freqArray[f_idx]
                    a = ampArray[a_idx]
                    ws.write(row, 0, f)
                    ws.write(row, 1, a)
                    tr = CMdata.micTracings[f_idx, a_idx, :]
                    tr = tr*1e9
                    tr = np.round(tr)/1e3  # round to nearest nv and conver tto uV
                    col = 2
                    for pt in tr:
                        ws.write(row, col, pt)
                        col += 1
                    row += 1                    

    except:
        print('CM.saveCMDataXLS: Exception writing data to file')        
        traceback.print_exc()
        
        
# save the processed data of this protocol
def saveCMData(CMdata, trialDuration, trialReps, filepath, saveOpts, timeStr):
    f = open(filepath, 'a')
    
    try:
        f.write('[data]')
        f.write('\nType= CM')
        f.write('\nTime= ' + timeStr)
        f.write('\nNote= ' + saveOpts.note)
        f.write("\nTrial duration= %0.3g" % trialDuration)
        f.write("\nTrial reps=  %0.3g" % trialReps)

        f.write("\nF= ")
        for freq in CMdata.freqArray:
            f.write(" %0.3f" % freq)
        f.write("\nA= ")
        for a in CMdata.ampArray:
            f.write(" %0.1f" % a)
    
        
        f.write('\nResp=')        
        for f_i in range(0, CMdata.numFreq):
            for a_i in range(0, CMdata.numAmp):
                f.write(' %0.5g' % CMdata.CMResp[f_i, a_i])
                if a_i < CMdata.numAmp-1:
                    f.write(',')
            f.write('\n')                
            
        f.write('\nNoise STD=')        
        for f_i in range(0, CMdata.numFreq):
            for a_i in range(0, CMdata.numAmp):
                    f.write(' %0.5g' % CMdata.noise_std[f_i, a_i])
                    if a_i < CMdata.numAmp-1:
                        f.write(',')
            f.write('\n')
            
        f.write('\nNoise Mean=')        
        for f_i in range(0, CMdata.numFreq):
            for a_i in range(0, CMdata.numAmp):
                    f.write(' %0.5g' % CMdata.noise_mean[f_i, a_i])
                    if a_i < CMdata.numAmp-1:
                        f.write(',')
            f.write('\n')
            
        if saveOpts.saveTracings:
            f.write('\nt= ')
            for t in CMdata.t[0:-1]:
                f.write(' %0.5g,' % t)
            f.write(' %0.5g ' % CMdata.t[-1])
                
            f.write('\nF\tA\tBioamp Tracing (uV)')
            freqArray = CMdata.freqArray
            ampArray = CMdata.ampArray
            for f_idx in range(0, len(freqArray)):
                for a_idx in range(0, len(ampArray)):
                    f.write('\n%0.3f\t%0.1f\t' % (freqArray[f_idx], ampArray[f_idx]))
                    tr = CMdata.tracings[f_idx, a_idx, :]
                    for pt in tr[0:-1]:
                        f.write(' %0.5g,' % pt)
                    f.write(' %0.5g' % tr[-1])
                    
            f.write('\nF\tA\tMic Tracing (uPa)')
            freqArray = CMdata.freqArray
            ampArray = CMdata.ampArray
            for f_idx in range(0, len(freqArray)):
                for a_idx in range(0, len(ampArray)):
                    f.write('\n%0.3f\t%0.1f\t' % (freqArray[f_idx], ampArray[f_idx]))
                    tr = CMdata.micTracings[f_idx, a_idx, :]
                    for pt in tr[0:-1]:
                        f.write(' %0.5g,' % pt)
                    f.write(' %0.5g' % tr[-1])                    
                        
        f.write('\n[/data]\n')
    except:
        print('CM.saveCMData: Exception writing data to file')        
        traceback.print_exc()
        
    f.close()
    
def saveCMDataPickle(CMdata, trialDuration, trialReps, fileName, saveOpts, timeStr):
    CMdata.trialDuration = trialDuration
    CMdata.trialReps = trialReps
    CMdata.note = saveOpts.note
    CMdata.timeStamp = timeStr
    tracings = CMdata.tracings
    micTracings = CMdata.micTracings
    if not saveOpts.saveTracings:
        CMdata.tracings = None
        CMdata.micTracings = None
    
    filepath = os.path.join(saveOpts.saveBaseDir, '%s.pickle' % fileName)
    f = open(filepath, 'wb')
    pickle.dump(CMdata, f)
    f.close()

    CMdata.tracings = tracings
    CMdata.micTracings = micTracings
    
   
def saveCMDataFig(CMdata, trialDuration, trialReps, saveDir, plotName, timeStr):
    plt.figure(1, figsize=(14, 11), dpi=80)
    plt.clf()
    
    numFreq = len(CMdata.freqArray)
    numAmp = len(CMdata.ampArray)
    nGraphs = numFreq + 1
    numRows = int(np.ceil(nGraphs ** 0.5))
    numCols = int(np.ceil(nGraphs / numRows))
    numSD = 2.5
    for n in range(0, numFreq):
        CMresp = CMdata.CMResp[n, :]
        noise_mean = CMdata.noise_mean[n, :]
        noise_std = CMdata.noise_std[n, :]
        noise = noise_mean + (numSD * noise_std)
        
        plt.subplot(numRows, numCols, n+1)
        plt.plot(CMdata.ampArray, 1e6*CMresp,'-bo', label='Response (uV)')
        plt.hold('on')
        plt.plot(CMdata.ampArray,  1e6*(noise_mean + 3*noise_std), '-ro', label='Noise')   
        plt.plot(CMdata.ampArray, noise_mean, ':r', label='Noise mean')
        # plt.plot(CMdata.ampArray, noise, '-rs', label='+ %g SD' % numSD)
        
        plt.xlabel('Amplitude (dB)', fontsize=10)
        if n == (numFreq-1):
            plt.legend(loc='upper left', fontsize=10)
        #plt.pcolormesh(t, f, Sxx)
        plt.ylabel('dB SPL', fontsize=10)
        plt.title('%0.2f kHz' % (CMdata.freqArray[n]/1e3), x=0.15, fontsize=10)
        
    plt.show()
    fname = os.path.join(saveDir, plotName)
    plt.savefig(fname)
    
def runCM(appObj, testMode=False):
    print("runCM")
    
    appObj.tabWidget.setCurrentIndex(4)
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
        testDataDir = os.path.join(appObj.basePath, 'exampledata', 'Speaker Calibration')
#        filePath = os.path.join(testDataDir, 'AudioParams.pickle')
#        f = open(filePath, 'rb')
#        audioParams = pickle.load(f)
#        f.close()
    else:
        # freqArray = appObj.getFrequencyArray()
        i1 = appObj.CM_freqLow_comboBox.currentIndex()
        i2 = appObj.CM_freqHigh_comboBox.currentIndex()
        print("runCM: i1= ", i1, "i2= ", i2)

        ampLow = appObj.CMampLow_spinBox.value()
        ampHigh = appObj.CMampHigh_spinBox.value()
        ampDelta = appObj.CMampDelta_spinBox.value()
        
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
    trialDur = appObj.CMstimDuration_dblSpinBox.value() * 1e-3
    stimOffset = appObj.CMstimOffset_dblSpinBox.value() * 1e-3
    nReps = appObj.CMtrialReps_spinBox.value()

    # set input rate to three times the highest output frequency, to allow plus a 
    
    #inputRate = 3*freqArray[-1]
    # inputRate = outputRate / int(np.floor(outputRate / inputRate))  # pick closest input rate that evenly divides output rate
    
    
    try:
        frameNum = 0
        isSaveDirInit = False
        chanNameOut = audioHW.speakerL_daqChan 
        attenLines = audioHW.attenL_daqChan
        
        freq_idx = 0
        CMdata = None
        
        for freq in freqArray:
            spkOut_trial = makeCMOutput(freq, trialDur, stimOffset, audioHW)
            spkOut = np.tile(spkOut_trial, nReps)
            
            npts = len(spkOut_trial)
            tOut = np.linspace(0, npts/outputRate, npts)
            print("runCM npts=%d len(spkOut_trial)= %d len(tOut)= %d" % (npts, len(spkOut_trial), len(tOut)))
            amp_idx = 0
            ptsPerRep = inputRate
            
            for amp in ampArray:
                print("runCM freq=" + repr(freq), " amp= ", + amp, " freq_idx= ", freq_idx, " amp_idx= ", amp_idx)
                vOut, attenLvl = audioHW.getCalibratedOutputVoltageAndAttenLevel(freq, amp, 0)
                
                # attenSig = AudioHardware.makeLM1972AttenSig(0)
                if not testMode:
                    AudioHardware.Attenuator.setLevel(attenLvl, attenLines)
                    # daq.sendDigOutCmd(attenLines, attenSig)
                    # appObj.oct_hw.SetAttenLevel(0, attenLines)
                
                pl = appObj.spCal_output
                pl.clear()
                endIdx = int(5e-3 * outputRate)        # only plot first 5 ms
                #pl.plot(t[0:endIdx], spkOut[0:endIdx], pen='b')
                pl.plot(tOut, spkOut_trial, pen='b')
                
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Time', 's', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Output', 'V', **labelStyle)
                        
                
                numInputSamples = nReps*int(inputRate*len(spkOut_trial)/outputRate)
                
                
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
                
                    daq.waitDoneOutput(stopAndClear=True)
                    daq.stopAnalogInput()
                    daq.clearAnalogInput()
                
                npts = len(mic_data)
                t = np.linspace(0, npts/inputRate, npts)
                pl = appObj.spCal_micInput
                pl.clear()
                pl.plot(t, mic_data, pen='b')
                
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Time', 's', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Response', 'Pa', **labelStyle)
    
    # def processCMData(mic_data, bioamp_data, nReps, freq, amp_idx, inputRate, CMdataIn):            
                CMptData, CMdata = processCMData(mic_data, bioamp_data, nReps, freq, freq_idx, amp_idx, freqArray, ampArray, inputRate, CMdata)

                print("runCM: plotting data")
                plotCMdata(appObj, CMptData, CMdata)
                
    #                if appObj.getSaveState():
    #                    if not isSaveDirInit:
    #                        saveDir = OCTCommon.initSaveDir(saveOpts, 'Speaker Calibration', audioParams=audioParams)
    #                        isSaveDirInit = True
    #    
    #                    if saveOpts.saveRaw:
    #                        OCTCommon.saveRawData(mic_data, saveDir, frameNum, dataType=3)
                    
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
        number = appObj.CMnumber
        name = 'CM'
        d = datetime.datetime.now()
        timeStr = d.strftime('%H_%M_%S')
        excelWS = CMPCommon.initExcelSpreadsheet(workbook, name, number, timeStr, note)
    
        appObj.CMnumber += 1                
        saveOpts.saveTracings = appObj.CM_saveTracings_checkBox.isChecked(  )
        saveDir = appObj.saveDir_lineEdit.text()
        saveCMDataXLS(CMdata, trialDur, nReps, excelWS, saveOpts)
        #saveCMData(CMdata, trialDur, nReps, appObj.saveFileTxt_filepath, saveOpts, timeStr)
        
        plotName = 'CM %d %s %s' % (number, timeStr, saveOpts.note)
        saveCMDataFig(CMdata, trialDur, nReps, saveDir, plotName, timeStr)
        saveCMDataPickle(CMdata, trialDur, nReps, plotName, saveOpts, timeStr)
            
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        QtGui.QMessageBox.critical (appObj, "Error", "Error during collection. Check command line output for details")
        
    8# update the audio hardware speaker calibration                     
    appObj.isCollecting = False
    QtGui.QApplication.processEvents() # check for GUI events, such as button presses
    appObj.finishCollection()

