# PyOCT.py
#  main file and  GUI interface

from ctypes import *
from matplotlib import *

import sys
import time
import datetime
import traceback
import os
import copy
import platform  # for differentiating platforms
import shutil # for copytree
import pickle

from PyQt4 import QtCore, QtGui, uic
import pyqtgraph as pg

import AudioHardware 
from OCTProtocolParams import *
from scipy import stats
from scipy import signal

import SpeakerCalibration
import SpeakerCalTest
import ReadMicBioamp
import Bioamp
import CM
import CMPCommon
import DPOAE
import ABR

#form_class = uic.loadUiType(os.path.join("..", "ui", "PyCMP.ui"))[0]                 # Load the UI
form_class = uic.loadUiType("PyCMP.ui")[0]                 # Load the UI
class CMPWindowClass(QtGui.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        try:
            self.setupUi(self)
        except Exception as ex:
            print(format(ex))

        
        baseDir = os.getcwd()
        self.baseDir = baseDir
        self.configPath = os.path.join(self.baseDir, 'config', 'local')
        self._readConfig()

        self.doneFlag = False
        self.isCollecting = False
        self.settingsPath = self.configPath
        
        self.xLblStyle = {'color': '#000', 'font-size': '16pt'}
        self.yLblStyle = {'color': '#000', 'font-size': '16pt'}
        
        self.freqComboBoxes = [self.DPOAE_freqLow_comboBox, self.DPOAE_freqHigh_comboBox, self.ABR_freqLow_comboBox, self.ABR_freqHigh_comboBox, self.CM_freqLow_comboBox, self.CM_freqHigh_comboBox]
        self._bindEventHandlers()
        
        self.protocolButtons = [self.calSpeakers_pushButton, self.calTest_pushButton, self.readMicBioamp_pushButton, self.runCM_pushButton, self.runDPOAE_pushButton, self.runABR_pushButton ]
        freqArray = self.getFrequencyArray()
        self.updateFrequencyInfo(freqArray)
        
        self.excelWB = None
        self.chooseSaveDir()  # prompt user to select a save directory and initialize new workbooko file
        
        self.CMnumber = 1
        self.DPOAEnumber = 1
        self.ABRnumber = 1
        
        self.ABRclick_RMS = 0
        
        self.loadSetups()
        
    def _readConfig(self):
        try:
            filepath = os.path.join(self.configPath, 'AudioHardware.txt')
            audioHW = AudioHardware.readAudioHWConfig(filepath)
        except:
            print("Could not read in audio hardware config")
            audioHW = AudioHardware.AudioHardware()
            
        self.audioHW = audioHW
        
        try:
            filepath = os.path.join(self.configPath, 'Bioamp.txt')
            bioamp = Bioamp.readBioampConfig(filepath)
        except:
            print("Could not read in bioamp config")
            bioamp = Bioamp.Bioamp()
            
        self.bioamp = bioamp
        
        # load in last speeaker calibration
        filename = 'speaker_cal_last.pickle'
        filepath = os.path.join(self.configPath, filename)
        spCal = SpeakerCalibration.loadSpeakerCal(filepath)
        self.audioHW.loadSpeakerCalFromProcData(spCal)
        
        
        micFilePath = os.path.join(self.configPath, 'microphones.txt')
        if os.path.exists(micFilePath):
            (micNameArray, micRespArray) = CMPCommon.readMicResponseFile(micFilePath)   # defined in OCTCommon.py
        else:
            micRespArray = []
            micNameArray = []
        
        # insert defaults
        micRespArray.insert(0, None)
        micNameArray.insert(0, 'Flat')
        
        self.micRespArray = micRespArray
        self.micNameArray = micNameArray
        
        for name in micNameArray:
            self.microphone_comboBox.addItem(name)

        micName = self.audioHW.micName
        print('CMPWindowClass.__init__: micName= %s' % micName)
        try:
            idx = micNameArray.index(micName)
        except ValueError:
            QtGui.QMessageBox.critical (self, "Could not find mic response", "Could not find mic response '%s', defaulting to flat response " % micName)
            idx = 0
        self.microphone_comboBox.setCurrentIndex(idx)
        self.audioHW.micResponse = micRespArray[idx]
        
    def loadSetups(self):
        self.freqQuickSet_comboBox.clear()
        self.freqSets = []
        self.freqSetsDict = {}
        freqSetsDir = os.path.join(self.configPath, 'Frequency Sets')
        
        try:
            fileList = os.listdir(freqSetsDir)
            for fName in fileList:
                filename, file_extension = os.path.splitext(fName)
                if file_extension=='.pickle':
                    filePath = os.path.join(freqSetsDir, fName)
                    try:
                        f = open(filePath, 'rb')
                        fSetup = pickle.load(f)
                        f.close()
                        
                        fParts = re.split('\.', fName)
                        setName = fParts[0]
                        self.freqSets.append(fSetup)
                        self.freqQuickSet_comboBox.addItem(setName)
                        idx = len(self.freqSets) - 1
                        self.freqSetsDict[setName] = idx
                    except Exception as ex:
                        traceback.print_exc(file=sys.stdout)
                        print("loadQuickSets: could not load frequency set '%s'" % setName)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            print("loadQuickSets: could not load frequency sets from directory'%s'" % freqSetsDir)
            
        
        
    def _bindEventHandlers(self):
        self.calSpeakers_pushButton.clicked.connect(self.calibrateSpeakersClicked)
        self.calTest_pushButton.clicked.connect(self.calibrationTestClicked)
        self.readMicBioamp_pushButton.clicked.connect(self.readMicBioampClicked)
        self.runCM_pushButton.clicked.connect(self.runCMClicked)
        self.freqDelta_dblSpinBox.valueChanged.connect(self.freqDeltaChanged)
        self.freqNumSteps_spinBox.valueChanged.connect(self.freqNumStepsChanged)
        self.freqLow_dblSpinBox.valueChanged.connect(self.freqParamsChanged)
        self.freqHigh_dblSpinBox.valueChanged.connect(self.freqParamsChanged)
        self.freqSpacing_comboBox.currentIndexChanged.connect(self.freqParamsChanged)
        
        self.freqQuickSet_comboBox.currentIndexChanged.connect(self.freqQuickSet_Changed)
        self.freqQuickSet_save_pushButton.clicked.connect(self.freqQuickSet_Save)
        
        self.saveDirChange_pushButton.clicked.connect(self.chooseSaveDir)
        self.newFile_pushButton.clicked.connect(self.newSaveFile)
        
        self.runDPOAE_pushButton.clicked.connect(self.runDPOAEclicked)
        self.runABR_pushButton.clicked.connect(self.runABRclicked)
        
        self.ABR_calClick_pushButton.clicked.connect(self.ABR_calClick)
        
        self.connect(self, QtCore.SIGNAL('triggered()'), self.closeEvent)
        
    def calibrateSpeakersClicked(self):
        if not self.isCollecting:
            SpeakerCalibration.runSpeakerCal(self)
        else:
            self.doneFlag = True
            
    def calibrationTestClicked(self):
        if not self.isCollecting:
            SpeakerCalTest.runSpeakerCalTest(self)
        else:
            self.doneFlag = True
    
    def readMicBioampClicked(self):
        if not self.isCollecting:
            ReadMicBioamp.run(self)
        else:
            self.doneFlag = True
        
    def runCMClicked(self):
        if not self.isCollecting:
            CM.runCM(self)
        else:
            self.doneFlag = True
    
    def runDPOAEclicked(self):
        if not self.isCollecting:
            DPOAE.runDPOAE(self)
        else:
            self.doneFlag = True
            
    def runABRclicked(self):
        if not self.isCollecting:
            ABR.runABR(self)
        else:
            self.doneFlag = True
            
    def ABR_calClick(self):
        if not self.isCollecting:
            ABR.calibrateClick(self)
        else:
            self.doneFlag = True
        
    def freqDeltaChanged(self):
        fLow = self.freqLow_dblSpinBox.value()*1000
        fHigh = self.freqHigh_dblSpinBox.value()*1000
        fdelta = self.freqDelta_dblSpinBox.value()*1000
        numSteps = round((fHigh - fLow)/fdelta) + 1
        numSteps = max((numSteps, 1))
        self.freqNumSteps_spinBox.blockSignals(True)
        self.freqNumSteps_spinBox.setValue(numSteps)
        self.freqNumSteps_spinBox.blockSignals(False)
        self.freqParamsChanged()

    def freqNumStepsChanged(self):
        fLow = self.freqLow_dblSpinBox.value()*1000
        fHigh = self.freqHigh_dblSpinBox.value()*1000
        numSteps = self.freqNumSteps_spinBox.value()
        fdelta = round((fHigh - fLow)/(numSteps-1))
        
        self.freqDelta_dblSpinBox.blockSignals(True)
        self.freqDelta_dblSpinBox.setValue(fdelta/1e3)
        self.freqDelta_dblSpinBox.blockSignals(False)
        self.freqParamsChanged()        
        
    def freqParamsChanged(self):
        freqArray = self.getFrequencyArray()
        self.updateFrequencyInfo(freqArray)
        
    def finishCollection(self):
        self.isCollecting = False
        for btn in self.protocolButtons:
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
            
        self.status_label.setText("Idle")
        self.progressBar.setValue(100)
        
    # updates the frequency information in the GUI
    def updateFrequencyInfo(self, freqArray):
        for comboBox in self.freqComboBoxes:
            comboBox.clear()
            for freq in freqArray:
                comboBox.addItem("%.3g" % (freq/1e3))
    
    def getFrequencyArray(self):
        fLow = self.freqLow_dblSpinBox.value()*1000
        fHigh = self.freqHigh_dblSpinBox.value()*1000
        fnumSteps = self.freqNumSteps_spinBox.value()
        fspacing = self.freqSpacing_comboBox.currentIndex()
        if fspacing == 0: # linear
            freqArray = np.linspace(fLow, fHigh, fnumSteps)
        else:
            freqArray = np.logspace(np.log10(fLow), np.log10(fHigh), fnumSteps)
            
        # round to nearest 0.01 Hz
        freqArray = np.round(freqArray * 100) / 100
        
        print("getFrequencyArray: freqArray=", freqArray)
        
        return freqArray
        
    def getFrequencySetup(self):
        fSetup = CMPCommon.blankRecord()
        fSetup.freqLow = self.freqLow_dblSpinBox.value()*1000
        fSetup.freqHigh = self.freqHigh_dblSpinBox.value()*1000
        fSetup.numSteps = self.freqNumSteps_spinBox.value()
        fSetup.spacing = self.freqSpacing_comboBox.currentIndex()
        fSetup.speaker = self.speaker_comboBox.currentIndex()
        
        return fSetup
        
    def loadFreqSet(self, fSetup):
        self.freqLow_dblSpinBox.setValue(fSetup.freqLow/1000)
        self.freqHigh_dblSpinBox.setValue(fSetup.freqHigh/1000)
        self.freqNumSteps_spinBox.setValue(fSetup.numSteps)
        self.freqSpacing_comboBox.setCurrentIndex(fSetup.spacing)
        self.speaker_comboBox.setCurrentIndex(fSetup.speaker)

    def freqQuickSet_Changed(self, idx):
        freqSet = self.freqSets[idx]
        print("freqQuickSet_Changed idx= %d" % (idx))
        self.loadFreqSet(freqSet)
    
    def freqQuickSet_Save(self):
        fSetup = self.getFrequencySetup()
        saveDir = os.path.join(self.configPath, 'Frequency Sets')
        curSetName = self.freqQuickSet_comboBox.currentText() 
        setName, ok = QtGui.QInputDialog.getText(self, 'Quickset Name', 'Name:', text=curSetName)
        
        saveSet = False
        newSet = True
        if ok:
            if self.freqQuickSet_comboBox.findText(setName) >= 0:
                newSet = False
                reply = QtGui.QMessageBox.question(self, 'Set already exsists', "Overwrite existing set %s?" % (setName), QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
                if reply == QtGui.QMessageBox.Yes:
                    saveSet = True
            else:
                saveSet = True
                    
        if saveSet:
            filepath = os.path.join(saveDir, '%s.pickle' % setName)
            f = open(filepath, 'wb')
            pickle.dump(fSetup, f)
            f.close()
            if(newSet):
                self.freqSets.append(fSetup)
                self.freqQuickSet_comboBox.addItem(setName)
                idx = len(self.freqSets) - 1
                self.freqkSetsDict[setName] = idx
                self.freqQuickSet_comboBox.setCurrentIndex(idx)
            else:
                idx = self.freqSetsDict[setName]
                self.freqSets[idx] = fSetup
    
    def chooseSaveDir(self):
        caption = "Choose save directory"
        directory = self.saveDir_lineEdit.text()
        newDir = QtGui.QFileDialog.getExistingDirectory (self, caption, directory)
        self.saveDir_lineEdit.setText(newDir)
        self.newSaveFile()
        
    def newSaveFile(self):
        saveDir = self.saveDir_lineEdit.text()
        if self.excelWB is not None:
            self.excelWB.close()
            
        d = datetime.datetime.now()
        timeStr = d.strftime('%Y-%m-%d %H_%M_%S')
        filename = 'PyCMP ' + timeStr + '.xlsx'
        filepath = os.path.join(saveDir, filename)    
        self.excelWB = CMPCommon.initExcelWorkbook(filepath)
        self.saveFile_lineEdit.setText(filename)
        
        filename = 'PyCMP ' + timeStr + '.txt'
        filepath = os.path.join(saveDir, filename)
        self.saveFileTxt_filepath = filepath
        
    def getSaveOpts(self):
        saveOpts = CMPCommon.SaveOpts()
        saveOpts.saveBaseDir = self.saveDir_lineEdit.text()
        saveOpts.saveRaw = self.saveRaw_checkBox.isChecked()
        saveOpts.note = self.note_lineEdit.text()
        
        return saveOpts
        
        
    def closeEvent(self, event):
        print("CMPWindowClass: closeEvent() called")
        if self.excelWB is not None:
            print("CMPWindowClass: closing workbook ... ")
            self.excelWB.close()
            print("                ... done")
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myWindow = CMPWindowClass(None)
    myWindow.show()
    app.exec_()
