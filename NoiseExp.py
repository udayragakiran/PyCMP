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
import time

    
def runNoiseExp(appObj, testMode=False):
    print("runNoiseExp")
    appObj.tabWidget.setCurrentIndex(5)
    appObj.doneFlag = False
    appObj.isCollecting = True
    # trigRate = octfpga.GetTriggerRate()
    audioHW = appObj.audioHW
    outputRate = audioHW.DAQOutputRate
    inputRate = audioHW.DAQInputRate
    
    cutoffLow = 1e3*appObj.noiseExp_filterLow_dblSpinBox.value()
    cutoffHigh = 1e3*appObj.noiseExp_filterHigh_dblSpinBox.value()
    durationMins = appObj.noiseExp_duration_dblSpinBox.value()
    durationSecs = durationMins*60
    
    try:
        if not testMode:
            from DAQHardware import DAQHardware
            daq = DAQHardware()
    
        chanNamesIn= [ audioHW.mic_daqChan]
        micVoltsPerPascal = audioHW.micVoltsPerPascal
        # mode = 'chirp'
        
        # make 1 second of noise         
        nOut = outputRate*1  
        magResp = np.zeros(nOut)
        fIdx1 = int(nOut*cutoffLow/outputRate)
        fIdx2 = int(nOut*cutoffHigh/outputRate)
        magResp[fIdx1:fIdx2] = 1
        phaseResp = 2*np.pi*np.random.rand(nOut) - np.pi
        sig = magResp*np.exp(-1j*phaseResp)
        spkOut = np.real(np.fft.ifft(sig))
        mx = np.max(spkOut)
        mn = np.min(spkOut)
        # normalies signal to be between -1 and 1
        spkOut = 2*(spkOut - mn)/(mx - mn) - 1
        
        maxV = audioHW.speakerOutputRng[1]
        spkOut = maxV*spkOut
        
        pl = appObj.spCalTest_output
        pl.clear()
        #endIdx = int(5e-3 * outputRate)        # only plot first 5 ms
        npts = len(spkOut)
        endIdx = npts
        t = np.linspace(0, npts/outputRate, npts)
        pl.plot(t[0:endIdx], spkOut[0:endIdx], pen='b')
                
        numInputSamples = int(inputRate*0.1)
            
        if testMode:
            # mic_data = OCTCommon.loadRawData(testDataDir, frameNum, dataType=3)                    
            pass
        else:
            chanNameOut = audioHW.speakerL_daqChan 
            attenLines = audioHW.attenL_daqChan
            attenLinesOther = audioHW.attenR_daqChan
                        
            if not testMode:
                AudioHardware.Attenuator.setLevel(0, attenLines)
                AudioHardware.Attenuator.setLevel(60, attenLinesOther)
    
            # setup the output task
            daq.setupAnalogOutput([chanNameOut], audioHW.daqTrigChanIn, int(outputRate), spkOut, isContinuous=True)
            daq.startAnalogOutput()
    
            # trigger the acquiisiton by sending ditital pulse
            daq.sendDigTrig(audioHW.daqTrigChanOut)
        
        tElapsed = 0
        tLast = time.time()
        npts = numInputSamples
        t = np.linspace(0, npts/inputRate, npts)
    
        while tElapsed < durationSecs:
            if not testMode:
                # setup the input task
                daq.setupAnalogInput(chanNamesIn, audioHW.daqTrigChanIn, int(inputRate), numInputSamples) 
                daq.startAnalogInput()
            
                # trigger the acquiisiton by sending ditital pulse
                daq.sendDigTrig(audioHW.daqTrigChanOut)
                
                daq.waitDoneInput()
    
                mic_data = daq.readAnalogInput()
                mic_data = mic_data[0, :]
                
                daq.stopAnalogInput()
                daq.clearAnalogInput()
            else:
                mic_data = np.random.rand(numInputSamples)
                
            mic_data = mic_data/micVoltsPerPascal
            
            pl = appObj.spCalTest_micInput
            pl.clear()
            pl.plot(t, mic_data, pen='b')
            
            labelStyle = appObj.xLblStyle
            pl.setLabel('bottom', 'Time', 's', **labelStyle)
            labelStyle = appObj.yLblStyle
            pl.setLabel('left', 'Response', 'Pa', **labelStyle)

            # calculate RMS
            nMicPts = len(mic_data)
            micRMS = np.mean(mic_data**2)
            micRMS = 20*np.log10(micRMS**0.5/2e-5)
            appObj.spCalTest_rms_label.setText("%0.3f dB" % micRMS)            
            
            nfft = int(2**np.ceil(np.log(nMicPts*5)/np.log(2)))
            print("nfft = ", nfft)
            win_fcn = np.hanning(nMicPts)
            micFFT = 2*np.abs(np.fft.fft(win_fcn*mic_data, nfft))/nMicPts
            micFFT = 2*micFFT[0:len(micFFT) // 2]
            micFFT = 20*np.log10(micFFT/2e-5)
            freq = np.linspace(0, inputRate/2, len(micFFT))
            pl = appObj.spCalTest_micFFT
            pl.clear()
            #df = freq[1] - freq[0]
            #print("NoiseExp: df= %0.3f i1= %d i2= %d nf= %d" % (df, i1, i2, nf))
            pl.plot(freq, micFFT, pen='b')
            labelStyle = appObj.xLblStyle
            pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
            labelStyle = appObj.yLblStyle
            pl.setLabel('left', 'Magnitude', 'db SPL', **labelStyle)
            
            QtGui.QApplication.processEvents() # check for GUI events, such as button presses
            
            # if done flag, break out of loop
            if appObj.doneFlag:
                break      
            
            t1 = time.time()
            tElapsed += (t1 - tLast)
            tLast = t1
    
        
        if not testMode:
            daq.stopAnalogOutput()
            daq.clearAnalogOutput()
            
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        QtGui.QMessageBox.critical (appObj, "Error", "Error during calibration. Check command line output for details")           
        
    8# update the audio hardware speaker calibration                     
    appObj.isCollecting = False
    QtGui.QApplication.processEvents() # check for GUI events, such as button presses
    appObj.finishCollection()


    