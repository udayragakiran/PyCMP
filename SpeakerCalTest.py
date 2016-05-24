# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:22:44 2015

@author: OHNS
"""

from DAQHardware import *
import numpy as np
from PyQt4 import QtCore, QtGui, uic
import AudioHardware
import pickle
import os
import sys
import traceback
import SpeakerCalibration

def makeSpkCalTestOutput(freq, amp, audioHW, stimDur, spkNum):
    #outV = 100e-3
    (outV, attenLvl) = audioHW.getCalibratedOutputVoltageAndAttenLevel(freq, amp, spkNum)

    if outV == 0:
        return None, 0
        
    stimEnv = 1 # 1 ms
    outputRate = audioHW.DAQOutputRate
    trialDur = stimDur
    trialPts = np.ceil(trialDur * outputRate)
    stimEnv = 1e-3*stimEnv
    envPts = np.ceil(stimEnv * outputRate)
    t = np.linspace(0, trialDur, trialPts)
    sig = outV*np.sin(2*np.pi*freq*t)
    envFcn = np.ones((trialPts))
    envFcn[0:envPts] = np.linspace(0, 1, envPts)
    envFcn[trialPts-envPts:] = np.linspace(1, 0, envPts)
    sig = sig*envFcn
    
    return sig, attenLvl
   
def processSpkCalTestData(mic_data, freq, freq_idx, amp_idx, inputRate, audioHW, magRespIn, phaseRespIn, THDIn):
    # print("SpeakerCalProtocol: processData: mic_data=" + repr(mic_data))
    # ensure data is 1D
    if len(mic_data.shape) > 1:
        mic_data = mic_data[:, 0]
        
    numpts = len(mic_data)
    print("SpeakerCalProtocol: processData: numpts= %d" % (numpts))

    t = np.linspace(0, numpts/inputRate, numpts)
    zero_pad_factor = 2
    numfftpts = numpts*zero_pad_factor
    winfcn = np.hanning(numpts)
    mic_fft = np.fft.fft(winfcn*mic_data, numfftpts)
    endIdx = np.ceil(numfftpts/2)
    mic_fft = mic_fft[0:endIdx]
    mic_fft_mag = 2*np.abs(mic_fft)
    
    # convert to dB, correctting for RMS and FFT length
    fftrms_corr = 2/(numpts*np.sqrt(2))
    mic_fft_mag = fftrms_corr*mic_fft_mag  # 20e-6 pa
    mic_fft_mag_log = 20*np.log10(mic_fft_mag/20e-6)  
    
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
        maxIdx = np.argmax(mag_rgn)
        stim_freq_mag = mag_rgn[maxIdx]
        stim_freq_phase = phase_rgn[maxIdx]
        
        mag_rgn = mic_fft_mag[fIdx-npts:fIdx+npts]
        stim_freq_mag_pa = mag_rgn[maxIdx]
        
        if audioHW.micResponse is not None:
            f = audioHW.micResponse[0, :]
            db = audioHW.micResponse[1, :]
            dbAdj = np.interp([freq], f, db)
            DebugLog.log("SpeakerCalTest: processData: dbAdj= %f" % (dbAdj))
            stim_freq_mag = stim_freq_mag - dbAdj
    except Exception as ex:
        #print(ex)
        traceback.print_exc(file=sys.stdout)
    
    thd = 0
    try:        
        nmax = int(np.floor((inputRate/2)/freq))
        npts = zero_pad_factor
#        for n in range(2, nmax+1):
#            fIdx = int(np.floor(freq*n*numfftpts/inputRate))
#            
#            mag_rgn = mic_fft_mag[fIdx-npts:fIdx+npts]
#            maxIdx = np.argmax(mag_rgn)
#            resp = mag_rgn[maxIdx]
#            
#            idx1= fIdx+npts+2
#            idx2 = fIdx+npts+82
#            noise_rgn = mic_fft_mag[fIdx-npts:fIdx+npts]
#            noise_mean = np.mean(noise_rgn)
#            noise_std = np.std(noise_rgn)
#            noise_lvl = noise_mean + 2*noise_std
#            print("n= ", n, " resp= ", resp, " noise_lvl= ", noise_lvl)
#            if resp > noise_lvl:
#                thd += (resp**2)

        #print("sum distortions= ", thd, " stim_freq_mag_pa= ", stim_freq_mag_pa)            
        # thd = (thd ** 0.5)/stim_freq_mag
        fIdx1 = int(np.floor(freq*2*numfftpts/inputRate))
        fIdx2 = int(np.floor((freq/2)*numfftpts/inputRate))
        
        mag_rgn1 = mic_fft_mag_log[fIdx1-npts:fIdx1+npts]
        mag_rgn2 = mic_fft_mag_log[fIdx2-npts:fIdx2+npts]
        
        maxIdx1 = np.argmax(mag_rgn1)
        maxIdx2 = np.argmax(mag_rgn2)
        
        resp1 = mag_rgn1[maxIdx1]
        resp2 = mag_rgn2[maxIdx2]
        #thd = max((resp1, resp2))
        thd = resp2
        print("thd= ", thd)
#        if thd > 0:
#            thd = 20*np.log10(thd)  # convert to dB
#        else:
#            thd = np.nan
#        print("20*log10(thd/20e-6)= ", thd)
    except Exception as ex:
        #print(ex)        
        traceback.print_exc(file=sys.stdout)
        
    print("SpeakerCalibration: processData: stim_freq_mag= %f stim_freq_phase= %f" % (stim_freq_mag, stim_freq_phase))
    micData = SpeakerCalibration.MicData()
    micData.raw = mic_data
    micData.t = t
    micData.fft_mag = mic_fft_mag_log
    micData.fft_phase = mic_fft_phase
    micData.fft_freq = mic_freq
    micData.stim_freq_mag = stim_freq_mag
    micData.stim_freq_phase = stim_freq_phase
    micData.thd = thd
    
    magRespIn[freq_idx, amp_idx] = stim_freq_mag
    phaseRespIn[freq_idx, amp_idx] = stim_freq_phase
    THDIn[freq_idx, amp_idx] = thd
        
    return micData, magRespIn, phaseRespIn, THDIn
    
        
def GetTestData(frameNum):
    mic_data = None
    # TODO load mic_data here
    
    return mic_data
    
def runSpeakerCalTest(appObj):
    print("runSpeakerCal")
    appObj.tabWidget.setCurrentIndex(5)
    appObj.doneFlag = False
    appObj.isCollecting = True
    # trigRate = octfpga.GetTriggerRate()
    audioHW = appObj.audioHW
    outputRate = audioHW.DAQOutputRate
    inputRate = audioHW.DAQInputRate
    
    # audioParams = appObj.getAudioParams()
    testMode = False

    freq_array = appObj.getFrequencyArray()
    
    ampLow = appObj.calTest_ampLow_spinBox.value()
    ampHigh = appObj.calTest_ampHigh_spinBox.value()
    ampDelta = appObj.calTest_ampDelta_spinBox.value()
    
    numSteps = np.floor((ampHigh - ampLow)/ampDelta) + 1
    amp_array = np.linspace(ampLow, ampHigh, numSteps)
       
    stimDur = appObj.calTest_stimDuration_dblSpinBox.value()*1e-3
    spkIdx = appObj.speaker_comboBox.currentIndex()
    
    # numSpk = audioParams.getNumSpeakers()
    if not testMode:
        from DAQHardware import DAQHardware
        daq = DAQHardware()
        
    daq = DAQHardware()
    chanNamesIn= [ audioHW.mic_daqChan]
    micVoltsPerPascal = audioHW.micVoltsPerPascal

    # spCal = SpeakerCalData(audioParams)
    try:
        frameNum = 0
        isSaveDirInit = False
        saveOpts = appObj.getSaveOpts()
        
        if spkIdx == 0:
            chanNameOut = audioHW.speakerL_daqChan
            attenLines = audioHW.attenL_daqChan
        else:
            chanNameOut = audioHW.speakerR_daqChan
            attenLines = audioHW.attenR_daqChan
            
        print("freq_array=" + repr(freq_array))
        numAmp = len(amp_array)
        numFreq = len(freq_array)
        magResp = np.zeros((numFreq, numAmp))
        magResp[:, :] = np.nan
        phaseResp = np.zeros((numFreq, numAmp))
        phaseResp[:, :] = np.nan
        THD = np.zeros((numFreq, numAmp))
        THD[:, :] = np.nan
        
        attenLvlLast = -1
        for freq_idx in range(0, numFreq):
            for amp_idx in range(0, numAmp):
                freq = freq_array[freq_idx]
                amp = amp_array[amp_idx]
                
                spkOut, attenLvl = makeSpkCalTestOutput(freq, amp, audioHW, stimDur, spkIdx)    
                print('runSpeakerCalTest freq= %0.3f kHz amp= %d attenLvl= %d' % (freq, amp, attenLvl))
                
                if spkOut is None:
                    print('runSpeakerCalTest freq= %0.3f kHz cannot output %d dB' % (freq, amp))
                    frameNum = frameNum + 1
                    continue
                
                npts = len(spkOut)
                t = np.linspace(0, npts/outputRate, npts)
                
                pl = appObj.spCalTest_output
                pl.clear()
                endIdx = int(5e-3 * outputRate)        # only plot first 5 ms
                pl.plot(t[0:endIdx], spkOut[0:endIdx], pen='b')
                
#                attenSig = AudioHardware.makeLM1972AttenSig(attenLvl)
                numInputSamples = int(inputRate*len(spkOut)/outputRate) 
                
                if not testMode:
                    if attenLvl != attenLvlLast:
                        # daq.sendDigOutCmd(attenLines, attenSig)
                        # AudioHardware.Attenuator.setLevel(attenLvl, attenLines)
                        if spkIdx == 0:
                            audioHW.setAttenuatorLevel(attenLvl, audioHW.maxAtten, daq)
                        else:
                            audioHW.setAttenuatorLevel(audioHW.maxAtten, attenLvl, daq)
                    attenLvlLast = attenLvl
                    
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
                    
                    daq.waitDoneInput()
                    daq.stopAnalogInput()
                    daq.clearAnalogInput()
                    daq.waitDoneOutput(stopAndClear=True)
                else:
                    mic_data = GetTestData(frameNum)
                
                npts = len(mic_data)
                t = np.linspace(0, npts/inputRate, npts)
                pl = appObj.spCalTest_micInput
                pl.clear()
                pl.plot(t, mic_data, pen='b')
                
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Time', 's', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Response', 'Pa', **labelStyle)
                
                micData, magResp, phaseResp, THD = processSpkCalTestData(mic_data, freq, freq_idx, amp_idx,  inputRate, audioHW, magResp, phaseResp, THD)
                     
                pl = appObj.spCalTest_micFFT
                pl.clear()
                df = micData.fft_freq[1] - micData.fft_freq[0]
                nf = len(micData.fft_freq)
                i1 = int(freq_array[0]*0.9/df)
                i2 = int(freq_array[-1]*1.1/df)
                print("runSpeakerCalTest: df= %0.3f i1= %d i2= %d nf= %d" % (df, i1, i2, nf))
                pl.plot(micData.fft_freq[i1:i2], micData.fft_mag[i1:i2], pen='b')
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Magnitude', 'db SPL', **labelStyle)
                
                pl = appObj.spCalTest_spkResp
                pl.clear()
                
                for a_idx in range(0, numAmp):
                    pen=(a_idx, numAmp)
                    pl.plot(freq_array, magResp[:, a_idx], pen=pen, symbol='o')
                        
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Magnitude', 'db SPL', **labelStyle)
                
                pl = appObj.spCalTest_spkDist
                pl.clear()
                
                for a_idx in range(0, numAmp):
                    pen=(a_idx, numAmp)
                    pl.plot(freq_array, THD[:, a_idx], pen=pen, symbol='o')
                        
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Magnitude', 'db SPL', **labelStyle)
                
                if saveOpts.saveRaw:
                    if not isSaveDirInit:
                        saveDir = CMPCommon.initSaveDir(saveOpts, 'Speaker Cal Test')
                        isSaveDirInit = True
    
                    if saveOpts.saveRaw:
                        CMPCommon.saveRawData(mic_data, saveDir, frameNum, dataType=3)
            
                frameNum = frameNum + 1
                
                QtGui.QApplication.processEvents() # check for GUI events, such as button presses
                
                # if done flag, break out of loop
                if appObj.doneFlag:
                    break
            
            # if done flag, break out of loop
            if appObj.doneFlag:
                break
                

            
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        QtGui.QMessageBox.critical (appObj, "Error", "Error during scan. Check command line output for details")           
        
    8# update the audio hardware speaker calibration                     
    appObj.isCollecting = False
    QtGui.QApplication.processEvents() # check for GUI events, such as button presses
    appObj.finishCollection()


    