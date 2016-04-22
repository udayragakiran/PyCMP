# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:45:12 2016

@author: OHNS
"""

import traceback
import sys
from PyQt4 import QtCore, QtGui, uic
import pyqtgraph as pg
import numpy as np
import scipy.signal

def run(appObj, testMode=False):
    print("ReadMicBioAmp.run")
    appObj.tabWidget.setCurrentIndex(1)
    appObj.doneFlag = False
    appObj.isCollecting = True
    # trigRate = octfpga.GetTriggerRate()
    audioHW = appObj.audioHW
    bioamp = appObj.bioamp
    # outputRate = audioHW.DAQOutputRate
    inputRate = audioHW.DAQInputRate
    
    if not testMode:
        from DAQHardware import DAQHardware
        daq = DAQHardware()

    chanNamesIn= [ audioHW.mic_daqChan, bioamp.daqChan]
    micVoltsPerPascal = audioHW.micVoltsPerPascal
    bioampGain = bioamp.gain
    
    firstIter = True
    while not appObj.doneFlag:
        try:
            if testMode:
                # mic_data = OCTCommon.loadRawData(testDataDir, frameNum, dataType=3)                    
                continue
            else:
                # inputTime = 100e-3
                inputTime = 1e-3*appObj.readMicBioamp_duration_dblSpinBox.value()
                numInputSamples = round(inputRate*inputTime)
                
                # setup the input task
                daq.setupAnalogInput(chanNamesIn, audioHW.daqTrigChanIn, int(inputRate), numInputSamples) 
                daq.startAnalogInput()
            
                # trigger the acquiisiton by sending ditital pulse
                daq.sendDigTrig(audioHW.daqTrigChanOut)
                
                data = daq.readAnalogInput()
                print("data.shape= ", data.shape)
                mic_data = data[0, :]
                bioamp_data = data[1, :]
                #mic_data = data[:, 0]
                #bioamp_data = data[:, 1]
                
                mic_data = mic_data/micVoltsPerPascal
                bioamp_data = bioamp_data/bioampGain
    
                daq.stopAnalogInput()
                daq.clearAnalogInput()
            
            npts = len(mic_data)
            t = np.linspace(0, npts/inputRate, npts)
            
            pl = appObj.inputs_micPlot
            if firstIter:
                pl.clear()
                
                micPI = pl.plot(t, mic_data, pen='b')
                
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Time', 's', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Response', 'Pa', **labelStyle)
            else:
                data = np.vstack((t, mic_data))
                micPI.setData(data.transpose())
            
            pl = appObj.inputs_bioampPlot
            if firstIter:
                pl.clear()
                bioampPI = pl.plot(t, bioamp_data, pen='b')
    
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Time', 's', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Response', 'V', **labelStyle)
            else:
                data = np.vstack((t, bioamp_data))
                bioampPI.setData(data.transpose())
            
            numfftpts = npts*2
            mic_fft = np.fft.fft(mic_data, numfftpts)
            endIdx = np.ceil(numfftpts/2)
            mic_fft = mic_fft[0:endIdx]
            mic_fft_mag = 2*np.abs(mic_fft)
            
            fftrms_corr = 2/(npts*np.sqrt(2))
            mic_fft_mag = fftrms_corr*mic_fft_mag 
            mic_fft_mag_log = 20*np.log10(mic_fft_mag/20e-6 )  # 20e-6 pa
            
            mic_freq = np.linspace(0, inputRate/2, endIdx)
            
            pl = appObj.inputs_micFFTPlot
            if firstIter:
                pl.clear()
                micFFTPI = pl.plot(mic_freq, mic_fft_mag_log, pen='b')

                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Frequency', 'Hz', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Magnitude', 'dB SPL', **labelStyle)
            else:
                data = np.vstack((mic_freq, mic_fft_mag_log))
                micFFTPI.setData(data.transpose())
                
            Wn = [300, 3000]
            Wn = np.array(Wn)/inputRate
            #Wn = [0.001, 0.01]
#            (b, a) = scipy.signal.butter(5, Wn=Wn, btype='bandpass')
            (b, a) = scipy.signal.iirfilter(2, Wn,  btype='bandpass', ftype='bessel')

            #b = scipy.signal.firwin(21, Wn)
            #a = [1.0]
            bioamp_filt = scipy.signal.lfilter(b, a, bioamp_data) 

            print("bioamp_data.shape= ", bioamp_data.shape, " t.shape=", t.shape, " Wn=", Wn)
            print("b= ", b)
            print("a= ", a)

            
            if firstIter:
                pl = appObj.inputs_bioampFilteredPlot
                pl.clear()
                bioampFFTPI = pl.plot(t, bioamp_filt, pen='b')
    
                labelStyle = appObj.xLblStyle
                pl.setLabel('bottom', 'Time', 's', **labelStyle)
                labelStyle = appObj.yLblStyle
                pl.setLabel('left', 'Response', 'V', **labelStyle)
                
            else:
                #data = np.vstack((t, bioamp_filt))
                bioampFFTPI.setData(t, bioamp_filt)
                
            firstIter = False
            
        except Exception as ex:
            traceback.print_exc(file=sys.stdout)
            QtGui.QMessageBox.critical (appObj, "Error", "Error. Check command line output for details")
            appObj.doneFlag = True
            
    
        QtGui.QApplication.processEvents() # check for GUI events, such as button presses
    
    # update the audio hardware speaker calibration                     
    appObj.finishCollection()
