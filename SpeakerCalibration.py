# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:22:44 2015

@author: OHNS
"""

import numpy as np
from PyQt4 import QtCore, QtGui, uic
import AudioHardware
import pickle
import os
import sys
import traceback

class SpeakerCalData:
    def __init__(self, freqArray):
        self.voltsOut = 0.1
        self.freq = freqArray
        numFreq = freqArray.shape[1]
        print("SpeakerCalData: numFreq= %d" % numFreq)
        self.magResp = np.zeros((2, numFreq))
        self.magResp[:, :] = np.NaN
        self.phaseResp = np.zeros((2, numFreq))
        self.phaseResp[:, :] = np.NaN


class MicData:
    def __init__(self):
        self.raw = None   # raw mic response
        self.t = None   # time
        self.fft_mag = None
        self.fft_phase = None
        self.fft_freq = None
        self.stim_freq_mag = None
        self.stim_freq_phase = None
        self.thd = None
        
def makeSpeakerCalibrationOutput(freq, audioHW, trialDur):
    outV = 100e-3
    outputRate = audioHW.DAQOutputRate
    # trialDur = 50e-3
    
    trialPts = np.ceil(trialDur * outputRate)
    stimEnv = 1e-3
    envPts = np.ceil(stimEnv * outputRate)
    t = np.linspace(0, trialDur, trialPts)
    sig = outV*np.sin(2*np.pi*freq*t)
    envFcn = np.ones((trialPts))
    envFcn[0:envPts] = np.linspace(0, 1, envPts)
    envFcn[trialPts-envPts:] = np.linspace(1, 0, envPts)
    sig = sig*envFcn
    
    return sig
   
def processSpkCalData(mic_data, freq, freq_idx, inputRate, speakerCalIn, spkNum, audioHW):
    # print("SpeakerCalProtocol: processData: mic_data=" + repr(mic_data))
    # ensure data is 1D
    if len(mic_data.shape) > 1:
        mic_data = mic_data[:, 0]
        
    numpts = len(mic_data)
    print("SpeakerCalibration: processData: numpts= %d" % (numpts))

    t = np.linspace(0, numpts/inputRate, numpts)
    zero_pad_factor = 2
    numfftpts = numpts*zero_pad_factor
    winfcn = np.hanning(numpts)
    win_corr = 2
    mic_fft = np.fft.fft(winfcn*mic_data, numfftpts)
    endIdx = np.ceil(numfftpts/2)
    mic_fft = mic_fft[0:endIdx]
    mic_fft_mag = 2*np.abs(mic_fft)
    
    # convert to dB, correctting for RMS and FFT length
    fftrms_corr = 1/(numpts*np.sqrt(2))
    mic_fft_mag = win_corr*fftrms_corr*mic_fft_mag 
    mic_fft_mag_log = 20*np.log10(mic_fft_mag/20e-6 )  # 20e-6 pa
    
    mic_fft_phase = np.angle(mic_fft)
    mic_freq = np.linspace(0, inputRate/2, endIdx)
    fIdx = int(np.floor(freq*numfftpts/inputRate))
    print("SpeakerCalibration: processData: freq= %f fIdx= %d" % (freq, fIdx))

    stim_freq_mag = np.NAN
    stim_freq_phase = np.NAN
        
    try:            
        npts = zero_pad_factor
        mag_rgn = mic_fft_mag_log[fIdx-npts:fIdx+npts]
        phase_rgn = mic_fft_phase[fIdx-npts:fIdx+npts]
        fIdx = int(np.floor(freq*numfftpts/inputRate))
        print("SpeakerCalibration: processData: freq= %f fIdx= %d" % (freq, fIdx))
        
        maxIdx = np.argmax(mag_rgn)
        stim_freq_mag = mag_rgn[maxIdx]
        stim_freq_phase = phase_rgn[maxIdx]
        
        if audioHW.micResponse is not None:
            f = audioHW.micResponse[0, :]
            db = audioHW.micResponse[1, :]
            dbAdj = np.interp([freq], f, db)
            DebugLog.log("SpeakerCalibration: processData: dbAdj= %f" % (dbAdj))
            stim_freq_mag = stim_freq_mag - dbAdj
    except Exception as ex:
        traceback.print_exc()
    
    print("SpeakerCalibration: processData: stim_freq_mag= %f stim_freq_phase= %f" % (stim_freq_mag, stim_freq_phase))
    micData = MicData()
    micData.raw = mic_data
    micData.t = t
    micData.fft_mag = mic_fft_mag_log
    micData.fft_phase = mic_fft_phase
    micData.fft_freq = mic_freq
    micData.stim_freq_mag = stim_freq_mag
    micData.stim_freq_phase = stim_freq_phase
    
    speakerCalIn.magResp[spkNum, freq_idx] = stim_freq_mag
    speakerCalIn.phaseResp[spkNum, freq_idx] = stim_freq_phase
        
    return micData, speakerCalIn
    
    
def processSpkCalDataChirp(mic_data_chirp, mic_data_ref, inputRate, spCal, spkIdx, chirp_f0, chirp_f1, ref_freq):
    # print("SpeakerCalProtocol: processData: mic_data=" + repr(mic_data))
    # ensure data is 1D
    if len(mic_data_chirp.shape) > 1:
        mic_data_chirp = mic_data_chirp[:, 0]
    if len(mic_data_ref.shape) > 1:
        mic_data_ref = mic_data_ref[:, 0]
        
    numpts = len(mic_data_ref)
    print("SpeakerCalibration: processData: numpts= %d" % (numpts))

    t = np.linspace(0, numpts/inputRate, numpts)
    zero_pad_factor = 2
    numfftpts = numpts*zero_pad_factor
    winfcn = np.hanning(numpts)
    mic_fft = np.fft.fft(winfcn*mic_data_ref, numfftpts)
    endIdx = np.ceil(numfftpts/2)
    mic_fft = mic_fft[0:endIdx]
    mic_fft_mag = 2*np.abs(mic_fft)
    
    # convert to dB, correctting for RMS and FFT length
    fftrms_corr = 2/(numpts*np.sqrt(2))
    mic_fft_mag = fftrms_corr*mic_fft_mag 
    mic_fft_mag_log = 20*np.log10(mic_fft_mag/20e-6 )  # 20e-6 pa
    
    mic_fft_phase = np.angle(mic_fft)
    mic_freq = np.linspace(0, inputRate/2, endIdx)
    fIdx = int(np.floor(ref_freq*numfftpts/inputRate))
    print("SpeakerCalibration: processData: freq= %f fIdx= %d" % (freq, fIdx))

    stim_freq_mag = np.NAN
    stim_freq_phase = np.NAN
        
    try:            
        npts = zero_pad_factor
        mag_rgn = mic_fft_mag_log[fIdx-npts:fIdx+npts]
        phase_rgn = mic_fft_phase[fIdx-npts:fIdx+npts]
        fIdx = int(np.floor(freq*numfftpts/inputRate))
        print("SpeakerCalibration: processData: freq= %f fIdx= %d" % (freq, fIdx))
        
        maxIdx = np.argmax(mag_rgn)
        stim_freq_mag = mag_rgn[maxIdx]
        stim_freq_phase = phase_rgn[maxIdx]
    except Exception as ex:
        traceback.print_exc()
    
    numpts = len(mic_data_chirp)
    t = np.linspace(0, numpts/inputRate, numpts)
    numfftpts = numpts*zero_pad_factor
    winfcn = np.hanning(numpts)
    mic_fft = np.fft.fft(winfcn*mic_data_ref, numfftpts)
    endIdx = np.ceil(numfftpts/2)
    mic_fft = mic_fft[0:endIdx]
    mic_fft_mag = 2*np.abs(mic_fft)
    
    # convert to dB, correctting for RMS and FFT length
    fftrms_corr = 2/(numpts*np.sqrt(2))
    mic_fft_mag = fftrms_corr*mic_fft_mag 
    mic_fft_mag_log = 20*np.log10(mic_fft_mag/20e-6 )  # 20e-6 pa
    
    print("SpeakerCalibration: processData: stim_freq_mag= %f stim_freq_phase= %f" % (stim_freq_mag, stim_freq_phase))
    micData = MicData()
    micData.raw = mic_data
    micData.t = t
    micData.fft_mag = mic_fft_mag_log
    micData.fft_phase = mic_fft_phase
    micData.fft_freq = ref_freq
    micData.stim_freq_mag = stim_freq_mag
    micData.stim_freq_phase = stim_freq_phase

    
    speakerCalIn.magResp[spkNum, :] = magResp
        
    return micData, speakerCalIn
    
    
    # save the processed data of this protocol
def saveSpeakerCal(spCalData, saveDir):
    filepath = os.path.join(saveDir, 'speaker_cal_last.pickle')
    f = open(filepath, 'wb')
    pickle.dump(spCalData, f)
    f.close()
   
def loadSpeakerCal(filepath):
    f = open(filepath, 'rb')
    spCal = pickle.load(f)
    
    f.close()
    
    return spCal
    
def runSpeakerCal(appObj, testMode=False):
    print("runSpeakerCal")
    appObj.tabWidget.setCurrentIndex(0)
    appObj.doneFlag = False
    appObj.isCollecting = True
    # trigRate = octfpga.GetTriggerRate()
    audioHW = appObj.audioHW
    outputRate = audioHW.DAQOutputRate
    inputRate = audioHW.DAQInputRate
    
    if testMode:
        testDataDir = os.path.join(appObj.basePath, 'exampledata', 'Speaker Calibration')
#        filePath = os.path.join(testDataDir, 'AudioParams.pickle')
#        f = open(filePath, 'rb')
#        audioParams = pickle.load(f)
#        f.close()
    else:
        freqArray = appObj.getFrequencyArray()
        
    # numSpk = audioParams.getNumSpeakers()
    numSpk = 1
    cIdx = appObj.speaker_comboBox.currentIndex()
    if cIdx > 0:
        numSpk = 2
    
    if not testMode:
        from DAQHardware import DAQHardware
        daq = DAQHardware()

    chanNamesIn= [ audioHW.mic_daqChan]
    micVoltsPerPascal = audioHW.micVoltsPerPascal
    # mode = 'chirp'
    mode = ''
    spCal = None
    # freq_array2 = audioParams.freq[1, :]
    
    try:
        frameNum = 0
        isSaveDirInit = False
        trialDur = appObj.spCal_stimDuration_dblSpinBox.value()*1e-3
        
        freq_array = freqArray
        freq_array2 = freqArray/1.22

        if numSpk == 1:
            freq_array = np.concatenate((freq_array, freq_array2))
            freq_array = np.sort(freq_array)
            freq_array2 = freq_array
        
        spCal = SpeakerCalData(np.vstack((freq_array, freq_array2)))            
        
        for spkNum in range(0, numSpk):
            chanNameOut = audioHW.speakerL_daqChan 
            attenLines = audioHW.attenL_daqChan
            attenLinesOther = audioHW.attenR_daqChan
            spkIdx = 0
                
            if spkNum == 2:
                chanNameOut = audioHW.speakerR_daqChan
                attenLines = audioHW.attenR_daqChan
                attenLinesOther = audioHW.attenL_daqChan
                spkIdx = 1
    
            freq_idx = 0

            attenSig = AudioHardware.makeLM1972AttenSig(0)
            
            if not testMode:
                AudioHardware.Attenuator.setLevel(0, attenLines)
                AudioHardware.Attenuator.setLevel(60, attenLinesOther)
                # daq.sendDigOutCmd(attenLines, attenSig)
                # appObj.oct_hw.SetAttenLevel(0, attenLines)
                
            if mode == 'chirp':
                tChirp = 1    
                f0 = 100
                f1 = 100e3
                k = (f1- f0)/tChirp
                nChirpPts = round(outputRate*tChirp)
                t = np.linspace(0, tChirp, nChirpPts)
                spkOut = np.cos(2*np.pi*(f1*t + (k/2)*t**2))
                
                pl = appObj.spCal_output
                pl.clear()
                endIdx = int(5e-3 * outputRate)        # only plot first 5 ms
                pl.plot(t[0:endIdx], spkOut[0:endIdx], pen='b')
                        
                numInputSamples = int(inputRate*len(spkOut)/outputRate) 
                
                if testMode:
                    # mic_data = OCTCommon.loadRawData(testDataDir, frameNum, dataType=3)                    
                    pass
                else:
                    daq.setupAnalogOutput([chanNameOut], audioHW.daqTrigChanIn, int(outputRate), spkOut)
                    daq.startAnalogOutput()
                    
                    # setup the input task
                    daq.setupAnalogInput(chanNamesIn, audioHW.daqTrigChanIn, int(inputRate), numInputSamples) 
                    daq.startAnalogInput()
                
                    # trigger the acquiisiton by sending ditital pulse
                    daq.sendDigTrig(audioHW.daqTrigChanOut)
                    
                    mic_data = daq.readAnalogInput()
                    mic_data = mic_data[0, :]
                    mic_data_chirp = mic_data/micVoltsPerPascal

                
                if not testMode:
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
                
                # play refernce tone
                refFreq = 4e3
                tRef = 50e-3
                
                nRefPts = round(outputRate*tRef)
                t = np.linspace(0, tRef, nRefPts)
                spkOut = np.cos(2*np.pi*refFreq*t)
                
                # apply envelope
                i1 = round(outputRate*1e-3)
                i2 = nRefPts- i1
                env = np.linspace(0, 1, i1)
                spkOut[0:i1] = spkOut[0:i1]*env
                spkOut[i2:] = spkOut[i2:]*(1-env)
                
                if testMode:
                    # mic_data = OCTCommon.loadRawData(testDataDir, frameNum, dataType=3)                    
                    pass
                else:
                    daq.setupAnalogOutput([chanNameOut], audioHW.daqTrigChanIn, int(outputRate), spkOut)
                    daq.startAnalogOutput()
                    
                    # setup the input task
                    daq.setupAnalogInput(chanNamesIn, audioHW.daqTrigChanIn, int(inputRate), numInputSamples) 
                    daq.startAnalogInput()
                
                    # trigger the acquiisiton by sending ditital pulse
                    daq.sendDigTrig(audioHW.daqTrigChanOut)
                    
                    mic_data = daq.readAnalogInput()
                    mic_data_ref = mic_data/micVoltsPerPascal
                    
                if not testMode:
                    daq.waitDoneOutput(stopAndClear=True)
                    daq.stopAnalogInput()
                    daq.clearAnalogInput()
                
                micData, spCal = processSpkCalDataChirp(mic_data_chirp, mic_data_ref, inputRate, spCal, spkIdx, f0, f1, refFreq)
                    
                pl = appObj.spCal_micFFT
                pl.clear()
                df = micData.fft_freq[1] - micData.fft_freq[0]
                nf = len(micData.fft_freq)
                i1 = int(freq_array[0]*0.9/df)
                i2 = int(freq_array[-1]*1.1/df)
                print("SpeakerCalibration: df= %0.3f i1= %d i2= %d nf= %d" % (df, i1, i2, nf))
                pl.plot(micData.fft_freq[i1:i2], micData.fft_mag[i1:i2], pen='b')
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Magnitude', 'db SPL', **labelStyle)
                
                pl = appObj.spCal_spkResp
                pl.clear()
#                pl.plot(1000*spCal.freq[spkIdx, :], spCal.magResp[spkIdx, :], pen="b", symbol='o')
                pl.plot(freq_array, spCal.magResp[spkIdx, :], pen="b", symbol='o')
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Magnitude', 'db SPL', **labelStyle)
            else:
                for freq in freq_array:
                    print("runSpeakerCal freq=" + repr(freq))
                    spkOut = makeSpeakerCalibrationOutput(freq, audioHW, trialDur)    
                    npts = len(spkOut)
                    t = np.linspace(0, npts/outputRate, npts)
                    
                    pl = appObj.spCal_output
                    pl.clear()
                    endIdx = int(5e-3 * outputRate)        # only plot first 5 ms
                    pl.plot(t[0:endIdx], spkOut[0:endIdx], pen='b')
                            
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
                        
                        mic_data = daq.readAnalogInput()
                        mic_data = mic_data[0, :]
                        mic_data = mic_data/micVoltsPerPascal
    
                    
                    if not testMode:
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
                    
                    micData, spCal = processSpkCalData(mic_data, freq, freq_idx, inputRate, spCal, spkIdx, audioHW)
                    
                    pl = appObj.spCal_micFFT
                    pl.clear()
                    df = micData.fft_freq[1] - micData.fft_freq[0]
                    nf = len(micData.fft_freq)
                    i1 = int(freq_array[0]*0.9/df)
                    i2 = int(freq_array[-1]*1.1/df)
                    print("SpeakerCalibration: df= %0.3f i1= %d i2= %d nf= %d" % (df, i1, i2, nf))
                    pl.plot(micData.fft_freq[i1:i2], micData.fft_mag[i1:i2], pen='b')
                    labelStyle = appObj.xLblStyle
                    pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
                    labelStyle = appObj.yLblStyle
                    pl.setLabel('left', 'Magnitude', 'db SPL', **labelStyle)
                    
                    pl = appObj.spCal_spkResp
                    pl.clear()
    #                pl.plot(1000*spCal.freq[spkIdx, :], spCal.magResp[spkIdx, :], pen="b", symbol='o')
                    pl.plot(freq_array, spCal.magResp[spkIdx, :], pen="b", symbol='o')
                    labelStyle = appObj.xLblStyle
                    pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
                    labelStyle = appObj.yLblStyle
                    pl.setLabel('left', 'Magnitude', 'db SPL', **labelStyle)
                    
                    freq_idx += 1
                    
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

                
            # if done flag, break out of loop
            if appObj.doneFlag:
                break
                
        if not appObj.doneFlag:
            saveDir = appObj.settingsPath
            saveSpeakerCal(spCal, saveDir)
            appObj.audioHW.loadSpeakerCalFromProcData(spCal)
            appObj.spCal = spCal            
            
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        QtGui.QMessageBox.critical (appObj, "Error", "Error during calibration. Check command line output for details")           
        
    8# update the audio hardware speaker calibration                     
    appObj.isCollecting = False
    QtGui.QApplication.processEvents() # check for GUI events, such as button presses
    appObj.finishCollection()


    