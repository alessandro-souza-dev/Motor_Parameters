#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Moto: Induction motor parameter estimation tool

Main window

Author: Julius Susanto
Last edited: August 2014
"""

import json
import os, sys

try:
    from PyQt5 import QtCore, QtGui as _QtGui, QtWidgets

    class _QtGuiCompat(object):
        def __getattr__(self, name):
            if name == 'qApp':
                return QtWidgets.QApplication.instance()
            if hasattr(_QtGui, name):
                return getattr(_QtGui, name)
            if hasattr(QtWidgets, name):
                return getattr(QtWidgets, name)
            raise AttributeError(name)

    QtGui = _QtGuiCompat()
except ImportError:
    from PyQt4 import QtCore, QtGui
    QtWidgets = QtGui
import numpy as np
import dateutil, pyparsing
import matplotlib.pyplot as plt
import globals
import saveload
from common_calcs import calc_pqt, get_torque, get_torque_sc
from descent import nr_solver, lm_solver, dnr_solver, nr_solver_sc
from genetic import ga_solver
from hybrid import hy_solver
from lab_calcs import (
    estimate_single_cage_parameters,
    load_point_summary,
    rated_torque,
    single_cage_curves,
    single_cage_performance,
    single_cage_summary,
    synchronous_speed,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts):
    return os.path.join(BASE_DIR, *parts)


def dialog_path(result):
    if isinstance(result, tuple):
        return result[0]
    return result

class Window(QtGui.QMainWindow):
    
    def __init__(self):
        super(Window, self).__init__()
        
        globals.init()
        self.initUI()       
        
    def initUI(self):
        
        self.resize(800, 600)
        self.centre()
        
        # Set background colour of main window to white
        palette = QtGui.QPalette()
        background_role = QtGui.QPalette.Window if hasattr(QtGui.QPalette, 'Window') else QtGui.QPalette.Background
        palette.setColor(background_role, QtCore.Qt.white)
        self.setPalette(palette)
        
        self.setWindowTitle('SPE Moto | Induction Motor Parameter Estimation Tool')
        self.setWindowIcon(QtGui.QIcon(resource_path('icons', 'motor.png')))    
        self.lab_input_fields = {}
        self.lab_result_fields = {}
        self.lab_results = None
        self.lab_plot_figure = None
              
        """
        Actions
        """
        exitAction = QtGui.QAction(QtGui.QIcon(resource_path('icons', 'exit.png')), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)
        
        loadAction = QtGui.QAction('&Open File...', self)
        loadAction.setStatusTip('Open file and load motor data')
        loadAction.triggered.connect(self.load_action)
        
        saveAction = QtGui.QAction('&Save As...', self)
        saveAction.setStatusTip('Save motor data')
        saveAction.triggered.connect(self.save_action)
        
        aboutAction = QtGui.QAction('&About Moto', self)
        aboutAction.setStatusTip('About Moto')
        aboutAction.triggered.connect(self.about_dialog)
        
        helpAction = QtGui.QAction('&User Manual', self)
        helpAction.setShortcut('F1')
        helpAction.setStatusTip('Moto user documentation')
        helpAction.triggered.connect(self.user_manual)
        
        """
        Menubar
        """
        menu_bar = self.menuBar() 
        fileMenu = menu_bar.addMenu('&File')
        fileMenu.addAction(loadAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(exitAction)
        helpMenu = menu_bar.addMenu('&Help')
        helpMenu.addAction(helpAction)
        helpMenu.addSeparator()
        helpMenu.addAction(aboutAction)
        
        """
        Main Screen
        """
        
        heading_font = QtGui.QFont()
        heading_font.setPointSize(10)
        heading_font.setBold(True)
        
        ################
        # Motor details
        ################
        
        header1 = QtGui.QLabel('Motor')
        #header1.setMinimumWidth(50)
        header1.setMinimumHeight(30)
        header1.setFont(heading_font)
        
        label1 = QtGui.QLabel('Description')
        #label1.setMinimumWidth(50)
        
        self.le1 = QtGui.QLineEdit()
        #self.le1.setMinimumWidth(150)
        self.le1.setText(str(globals.motor_data["description"]))
               
        label2 = QtGui.QLabel('Synchronous speed')
        #label2.setMinimumWidth(50)
        
        self.le2 = QtGui.QLineEdit()
        #self.le2.setMinimumWidth(50)
        self.le2.setText(str(globals.motor_data["sync_speed"]))
        
        label2a = QtGui.QLabel('rpm')
        #label2a.setMinimumWidth(30)
 
        label3 = QtGui.QLabel('Rated speed')
        #label3.setMinimumWidth(50)
        
        self.le3 = QtGui.QLineEdit()
        #self.le3.setMinimumWidth(50)
        self.le3.setText(str(globals.motor_data["rated_speed"]))
        
        label3a = QtGui.QLabel('rpm')
        #label3a.setMinimumWidth(30)
           
        label4 = QtGui.QLabel('Rated power factor')
        #label4.setMinimumWidth(50)
        
        self.le4 = QtGui.QLineEdit()
        #self.le4.setMinimumWidth(50)
        self.le4.setText(str(globals.motor_data["rated_pf"]))
        
        label4a = QtGui.QLabel('pf')
        #label4a.setMinimumWidth(20)
        
        label5 = QtGui.QLabel('Rated efficiency')
        #label5.setMinimumWidth(50)
        
        self.le5 = QtGui.QLineEdit()
        #self.le5.setMinimumWidth(50)
        self.le5.setText(str(globals.motor_data["rated_eff"]))
        
        label5a = QtGui.QLabel('pu')
        #label5a.setMinimumWidth(20)

        label6 = QtGui.QLabel('Breakdown torque')
        #label6.setMinimumWidth(50)
        
        self.le6 = QtGui.QLineEdit()
        #self.le6.setMinimumWidth(50)
        self.le6.setText(str(globals.motor_data["T_b"]))
        
        label6a = QtGui.QLabel('T/Tn')
        #label6a.setMinimumWidth(40)
        
        label7 = QtGui.QLabel('Locked rotor torque')
        #label7.setMinimumWidth(50)
        
        self.le7 = QtGui.QLineEdit()
        #self.le7.setMinimumWidth(50)
        self.le7.setText(str(globals.motor_data["T_lr"]))
        
        label7a = QtGui.QLabel('T/Tn')
        #label7a.setMinimumWidth(40)
        
        label8 = QtGui.QLabel('Locked rotor current')
        #label8.setMinimumWidth(50)
        
        self.le8 = QtGui.QLineEdit()
        #self.le8.setMinimumWidth(50)
        self.le8.setText(str(globals.motor_data["I_lr"]))
        
        label8a = QtGui.QLabel('pu')
        #label8a.setMinimumWidth(40)
        
        ########
        # Model
        ########
        
        header2 = QtGui.QLabel('Model')
        header2.setMinimumHeight(40)
        header2.setFont(heading_font)
        
        label_model = QtGui.QLabel('Model')
        #label_model.setMinimumWidth(150)
        
        self.combo_model = QtGui.QComboBox()
        self.combo_model.addItem("Single cage")
        # self.combo_model.addItem("Single cage w/o core losses")
        self.combo_model.addItem("Double cage")
        self.combo_model.setCurrentIndex(1)
        
        self.img1 = QtGui.QLabel()
        self.img1.setAlignment(QtCore.Qt.AlignCenter)
        self.img1.setMinimumSize(180, 120)
        self.img1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.motor_pixmap = None
        self.set_motor_image('dbl_cage.png')
        
        #####################
        # Algorithm settings
        #####################
        
        header3 = QtGui.QLabel('Settings')
        header3.setMinimumHeight(40)
        header3.setFont(heading_font)
        
        label9 = QtGui.QLabel('Maximum # iterations')
        
        self.le9 = QtGui.QLineEdit()
        self.le9.setText(str(globals.algo_data["max_iter"]))
        self.le9.setStatusTip('Maximum number of iterations allowed')
        
        label10 = QtGui.QLabel('Convergence criterion')
        
        self.le10 = QtGui.QLineEdit()
        self.le10.setText(str(globals.algo_data["conv_err"]))
        self.le10.setStatusTip('Squared error required to qualify for convergence')

        self.label11 = QtGui.QLabel('Linear constraint k_r')
        
        self.le11 = QtGui.QLineEdit()
        self.le11.setText(str(globals.algo_data["k_r"]))
        self.le11.setStatusTip('Linear constraint for Rs')

        self.label12 = QtGui.QLabel('Linear constraint k_x')
        
        self.le12 = QtGui.QLineEdit()
        self.le12.setText(str(globals.algo_data["k_x"]))
        self.le12.setStatusTip('Linear constraint for Xr2')
        
        # Genetic Algorithm Widgets
        ############################
        
        self.labeln_gen = QtGui.QLabel('Maximum # generations')
        self.labeln_gen.setVisible(0)
        self.labelpop = QtGui.QLabel('Members in population')
        self.labelpop.setVisible(0)
        self.labeln_r = QtGui.QLabel('Members in mating pool')
        self.labeln_r.setVisible(0)
        self.labeln_e = QtGui.QLabel('Elite children')
        self.labeln_e.setVisible(0)
        self.labelc_f = QtGui.QLabel('Crossover fraction')
        self.labelc_f.setVisible(0)
        
        self.len_gen = QtGui.QLineEdit()
        self.len_gen.setText(str(globals.algo_data["n_gen"]))
        self.len_gen.setStatusTip('Maximum number of generations allowed')
        self.len_gen.hide()
        
        self.lepop = QtGui.QLineEdit()
        self.lepop.setText(str(globals.algo_data["pop"]))
        self.lepop.setStatusTip('Number of members in each generation')
        self.lepop.hide()
        
        self.len_r = QtGui.QLineEdit()
        self.len_r.setText(str(globals.algo_data["n_r"]))
        self.len_r.setStatusTip('Number of members in a mating pool')
        self.len_r.hide()
        
        self.len_e = QtGui.QLineEdit()
        self.len_e.setText(str(globals.algo_data["n_e"]))
        self.len_e.setStatusTip('Number of elite children')
        self.len_e.hide()
        
        self.lec_f = QtGui.QLineEdit()
        self.lec_f.setText(str(globals.algo_data["c_f"]))
        self.lec_f.setStatusTip('Proportion of children spawned through crossover')
        self.lec_f.hide()
        
        
        label_algo = QtGui.QLabel('Algorithm')
        #label_algo.setMinimumWidth(150)
        
        self.combo_algo = QtGui.QComboBox()
        self.combo_algo.addItem("Newton-Raphson")
        self.combo_algo.addItem("Levenberg-Marquardt")
        self.combo_algo.addItem("Damped Newton-Raphson")
        self.combo_algo.addItem("Genetic Algorithm")
        self.combo_algo.addItem("Hybrid GA-NR")
        self.combo_algo.addItem("Hybrid GA-LM")
        self.combo_algo.addItem("Hybrid GA-DNR")
        
        calc_button = QtGui.QPushButton("Calculate")
        calc_button.setStatusTip('Estimate equivalent circuit parameters')
        
        self.plot_button = QtGui.QPushButton("Plot")
        self.plot_button.setDisabled(1)
        self.plot_button.setStatusTip('Plot torque-speed and current-speed curves')
        
        ####################
        # Algorithm results
        ####################
        
        header4 = QtGui.QLabel('Results')
        #header4.setMinimumWidth(150)
        header4.setMinimumHeight(40)
        header4.setFont(heading_font)
        
        label13 = QtGui.QLabel('R_s')
        #label13.setFixedWidth(50)
        
        self.leRs = QtGui.QLineEdit()
        self.leRs.setStatusTip('Stator resistance (pu)')
        
        label14 = QtGui.QLabel('X_s')
        #label14.setMinimumWidth(150)
        
        self.leXs = QtGui.QLineEdit()
        self.leXs.setStatusTip('Stator reactance (pu)')
        
        label15 = QtGui.QLabel('X_m')
        #label15.setMinimumWidth(150)
        
        self.leXm = QtGui.QLineEdit()
        self.leXm.setStatusTip('Magnetising resistance (pu)')
        
        label16 = QtGui.QLabel('X_r1')
        #label16.setMinimumWidth(150)
        
        self.leXr1 = QtGui.QLineEdit()
        self.leXr1.setStatusTip('Inner cage rotor reactance (pu)')
        
        label17 = QtGui.QLabel('R_r1')
        #label17.setMinimumWidth(150)
        
        self.leRr1 = QtGui.QLineEdit()
        self.leRr1.setStatusTip('Inner cage rotor resistance (pu)')
        
        self.label18 = QtGui.QLabel('X_r2')
        #label18.setMinimumWidth(150)
        
        self.leXr2 = QtGui.QLineEdit()
        self.leXr2.setStatusTip('Outer cage rotor reactance (pu)')
        
        self.label19 = QtGui.QLabel('R_r2')
        #label19.setMinimumWidth(150)
        
        self.leRr2 = QtGui.QLineEdit()
        self.leRr2.setStatusTip('Outer cage rotor resistance (pu)')
        
        label20 = QtGui.QLabel('R_c')
        #label20.setMinimumWidth(150)
        
        self.leRc = QtGui.QLineEdit()
        self.leRc.setStatusTip('Core loss resistance (pu)')
        
        label21 = QtGui.QLabel('Converged?')
        #label21.setMinimumWidth(150)
        
        self.leConv = QtGui.QLineEdit()
        self.leConv.setStatusTip('Did algorithm converge?')
        
        label22 = QtGui.QLabel('Squared Error')
        #label22.setMinimumWidth(150)
        
        self.leErr = QtGui.QLineEdit()
        self.leErr.setStatusTip('Squared error of estimate')
        
        label23 = QtGui.QLabel('Iterations')
        #label23.setMinimumWidth(150)
        
        self.leIter = QtGui.QLineEdit()
        self.leIter.setStatusTip('Number of iterations / generations')
        
        ##############
        # Grid layout
        ##############
        
        grid = QtGui.QGridLayout()
        
        # Motor details
        i = 0
        grid.addWidget(header1, i, 0)
        grid.addWidget(label1, i+1, 0)
        grid.addWidget(self.le1, i+1, 1, 1, 5)
        grid.addWidget(label2, i+2, 0)
        grid.addWidget(self.le2, i+2, 1)
        grid.addWidget(label2a, i+2, 2)
        grid.addWidget(label3, i+3, 0)
        grid.addWidget(self.le3, i+3, 1)
        grid.addWidget(label3a, i+3, 2)
        grid.addWidget(label4, i+4, 0)
        grid.addWidget(self.le4, i+4, 1)
        grid.addWidget(label4a, i+4, 2)
        grid.addWidget(label5, i+5, 0)
        grid.addWidget(self.le5, i+5, 1)
        grid.addWidget(label5a, i+5, 2)
        grid.addWidget(label6, i+3, 4)
        grid.addWidget(self.le6, i+3, 5)
        grid.addWidget(label6a, i+3, 6)
        grid.addWidget(label7, i+4, 4)
        grid.addWidget(self.le7, i+4, 5)
        grid.addWidget(label7a, i+4, 6)
        grid.addWidget(label8, i+5, 4)
        grid.addWidget(self.le8, i+5, 5)
        grid.addWidget(label8a, i+5, 6)
        
        # Model
        i = 9
        #grid.addWidget(header2, i, 0)
        grid.addWidget(label_model, i+1, 0)
        grid.addWidget(self.combo_model, i+1, 1)
        grid.addWidget(self.img1, i+1, 3, i-7, 6)
        
        # Algorithm settings
        i = 12
        grid.addWidget(header3, i, 0)
        grid.addWidget(label_algo, i+1, 0)
        grid.addWidget(self.combo_algo, i+1, 1)
        grid.addWidget(label9, i+2, 0)
        grid.addWidget(self.le9, i+2, 1)
        grid.addWidget(label10, i+3, 0)
        grid.addWidget(self.le10, i+3, 1)
        grid.addWidget(self.label11, i+2, 3)
        grid.addWidget(self.le11, i+2, 4)
        grid.addWidget(self.label12, i+3, 3)
        grid.addWidget(self.le12, i+3, 4)
        
        # Genetic algorithm parameters
        grid.addWidget(self.labeln_gen, i+2, 3)
        grid.addWidget(self.len_gen, i+2, 4)
        grid.addWidget(self.labelpop, i+3, 3)
        grid.addWidget(self.lepop, i+3, 4)
        grid.addWidget(self.labeln_r, i+4, 3)
        grid.addWidget(self.len_r, i+4, 4)
        grid.addWidget(self.labeln_e, i+2, 5)
        grid.addWidget(self.len_e, i+2, 6)
        grid.addWidget(self.labelc_f, i+3, 5)
        grid.addWidget(self.lec_f, i+3, 6)
        
        grid.addWidget(calc_button, i+4, 5)
        grid.addWidget(self.plot_button, i+4, 6)
        
        # Algorithm results
        i = 17
        grid.addWidget(header4, i, 0)
        grid.addWidget(label13, i+1, 0)
        grid.addWidget(self.leRs, i+1, 1)
        grid.addWidget(label14, i+2, 0)
        grid.addWidget(self.leXs, i+2, 1)
        grid.addWidget(label15, i+3, 0)
        grid.addWidget(self.leXm, i+3, 1)
        grid.addWidget(label20, i+4, 0)
        grid.addWidget(self.leRc, i+4, 1)
        grid.addWidget(label16, i+1, 3)
        grid.addWidget(self.leXr1, i+1, 4)
        grid.addWidget(label17, i+2, 3)
        grid.addWidget(self.leRr1, i+2, 4)
        grid.addWidget(self.label18, i+3, 3)
        grid.addWidget(self.leXr2, i+3, 4)
        grid.addWidget(self.label19, i+4, 3)
        grid.addWidget(self.leRr2, i+4, 4)
        grid.addWidget(label21, i+1, 5)
        grid.addWidget(self.leConv, i+1, 6)
        grid.addWidget(label22, i+2, 5)
        grid.addWidget(self.leErr, i+2, 6)
        grid.addWidget(label23, i+3, 5)
        grid.addWidget(self.leIter, i+3, 6)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(4, 1)
        grid.setColumnStretch(5, 1)
        grid.setColumnMinimumWidth(3, 24)
        
        grid.setAlignment(QtCore.Qt.AlignTop)      

        main_screen = QtGui.QWidget()
        main_screen.setLayout(grid)
        main_screen.setStatusTip('Ready')

        scroll_area = QtGui.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtGui.QFrame.NoFrame)
        scroll_area.setWidget(main_screen)

        self.tabs = QtGui.QTabWidget()
        self.tabs.addTab(scroll_area, 'Parameter Estimation')
        self.tabs.addTab(self.create_lab_tab(), 'Laboratory Tests')

        self.setCentralWidget(self.tabs)
        
        # Event handlers
        calc_button.clicked.connect(self.calculate)
        self.plot_button.clicked.connect(self.plot_curves)
        
        self.le1.editingFinished.connect(self.update_data)
        self.le2.editingFinished.connect(self.update_data)
        self.le3.editingFinished.connect(self.update_data)
        self.le4.editingFinished.connect(self.update_data)
        self.le5.editingFinished.connect(self.update_data)
        self.le6.editingFinished.connect(self.update_data)
        self.le7.editingFinished.connect(self.update_data)
        self.le8.editingFinished.connect(self.update_data)
        self.le9.editingFinished.connect(self.update_data)
        self.le10.editingFinished.connect(self.update_data)
        self.le11.editingFinished.connect(self.update_data)
        self.le12.editingFinished.connect(self.update_data)
        self.len_gen.editingFinished.connect(self.update_data)
        self.lepop.editingFinished.connect(self.update_data)
        self.len_r.editingFinished.connect(self.update_data)
        self.len_e.editingFinished.connect(self.update_data)
        self.lec_f.editingFinished.connect(self.update_data)
        
        ##########################
        #TO DO - connects for combo boxes - combo_model and combo_algo (what signal to use?)
        ##########################
        self.combo_algo.currentIndexChanged.connect(self.update_algo)
        self.combo_model.currentIndexChanged.connect(self.update_model)
        
        self.statusBar().showMessage('Ready')
    
    # Calculate parameter estimates
    def calculate(self):
        self.statusBar().showMessage('Calculating...')
        
        sf = (globals.motor_data["sync_speed"] - globals.motor_data["rated_speed"]) / globals.motor_data["sync_speed"]
        
        if self.combo_model.currentIndex() == 0:
            # Single cage
            p = [sf, globals.motor_data["rated_eff"], globals.motor_data["rated_pf"], globals.motor_data["T_b"]]
            [z, iter, err, conv] = nr_solver_sc(p, 0, globals.algo_data["k_x"], globals.algo_data["k_r"], globals.algo_data["max_iter"], globals.algo_data["conv_err"]) 
            
        else:
            # Double cage
            p = [sf, globals.motor_data["rated_eff"], globals.motor_data["rated_pf"], globals.motor_data["T_b"], globals.motor_data["T_lr"], globals.motor_data["I_lr"] ]            
            
            if self.combo_algo.currentText() == "Newton-Raphson":
                [z, iter, err, conv] = nr_solver(p, 0, globals.algo_data["k_x"], globals.algo_data["k_r"], globals.algo_data["max_iter"], globals.algo_data["conv_err"])           
            
            if self.combo_algo.currentText() == "Levenberg-Marquardt":
                [z, iter, err, conv] = lm_solver(p, 0, globals.algo_data["k_x"], globals.algo_data["k_r"], 1e-7, 5.0, globals.algo_data["max_iter"], globals.algo_data["conv_err"])
                
            if self.combo_algo.currentText() == "Damped Newton-Raphson":
                [z, iter, err, conv] = dnr_solver(p, 0, globals.algo_data["k_x"], globals.algo_data["k_r"], 1e-7, globals.algo_data["max_iter"], globals.algo_data["conv_err"])
                
            if self.combo_algo.currentText() == "Genetic Algorithm":
                [z, iter, err, conv] = ga_solver(self, p, globals.algo_data["pop"], globals.algo_data["n_r"], globals.algo_data["n_e"], globals.algo_data["c_f"], globals.algo_data["n_gen"], globals.algo_data["conv_err"])
                
            if self.combo_algo.currentText() == "Hybrid GA-NR":
                [z, iter, err, conv] = hy_solver(self, "NR", p, globals.algo_data["pop"], globals.algo_data["n_r"], globals.algo_data["n_e"], globals.algo_data["c_f"], globals.algo_data["n_gen"], globals.algo_data["conv_err"])
                
            if self.combo_algo.currentText() == "Hybrid GA-LM":
                [z, iter, err, conv] = hy_solver(self, "LM", p, globals.algo_data["pop"], globals.algo_data["n_r"], globals.algo_data["n_e"], globals.algo_data["c_f"], globals.algo_data["n_gen"], globals.algo_data["conv_err"])
                
            if self.combo_algo.currentText() == "Hybrid GA-DNR":
                [z, iter, err, conv] = hy_solver(self, "DNR", p, globals.algo_data["pop"], globals.algo_data["n_r"], globals.algo_data["n_e"], globals.algo_data["c_f"], globals.algo_data["n_gen"], globals.algo_data["conv_err"])
        
        self.leRs.setText(str(np.round(z[0],5)))
        self.leXs.setText(str(np.round(z[1],5)))
        self.leXm.setText(str(np.round(z[2],5)))
        self.leRr1.setText(str(np.round(z[3],5)))        
         
        if self.combo_model.currentIndex() == 1:
            self.leXr1.setText(str(np.round(z[4],5)))
            self.leRr2.setText(str(np.round(z[5],5)))
            self.leXr2.setText(str(np.round(z[6],5)))
            self.leRc.setText(str(np.round(z[7],5)))
        else:
            self.leRc.setText(str(np.round(z[4],5)))
            self.leXr1.setText(str(np.round(z[5],5)))
        
        if conv == 1:
            self.leConv.setText("Yes")
        else:
            QtGui.QMessageBox.warning(self, 'Warning', "Algorithm did not converge.", QtGui.QMessageBox.Ok)
            self.leConv.setText("No")
            
        self.leErr.setText(str(np.round(err,9)))
        self.leIter.setText(str(iter))
        
        # Only enable the plot button if the squared error is within the bounds of reason
        if err < 1:
            self.plot_button.setEnabled(1)
        else:
            self.plot_button.setDisabled(1)
        
        self.statusBar().showMessage('Ready')
        
    # Plot torque-speed and current-speed curves
    def plot_curves(self):
        sf = (globals.motor_data["sync_speed"] - globals.motor_data["rated_speed"]) / globals.motor_data["sync_speed"]
        if self.combo_model.currentIndex() == 0:
            # Single cage
            x = [float(self.leRs.text()), float(self.leXs.text()) , float(self.leXm.text()), float(self.leRr1.text()), float(self.leRc.text()), float(self.leXr1.text())]
        else:
            # Double cage
            x = [float(self.leRs.text()), float(self.leXs.text()) , float(self.leXm.text()), float(self.leRr1.text()), float(self.leXr1.text()), float(self.leRr2.text()), float(self.leXr2.text()), float(self.leRc.text())]
        
        # Rated per-unit torque
        T_rtd = globals.motor_data["rated_eff"] * globals.motor_data["rated_pf"] / (1 - sf)
        
        Tm = np.zeros(1001)
        Im = np.zeros(1001)
        speed = np.zeros(1001)
        speed[1000] = globals.motor_data["sync_speed"]
        for n in range(0,1000):
            speed[n] = float(n) / 1000 * globals.motor_data["sync_speed"]
            i = 1 - float(n) / 1000
            if self.combo_model.currentIndex() == 0:
                # Single cage
                [Ti, Ii] = get_torque_sc(i,x)
            else:
                # Double cage
                [Ti, Ii] = get_torque(i,x)
            
            Tm[n] = Ti / T_rtd      # Convert torque to T/Tn value
            Im[n] = np.abs(Ii)
        
        # Plot torque-speed and current-speed curves
        if plt.fignum_exists(1):
            # Do nothing
            QtGui.QMessageBox.warning(self, 'Warning', "A plot is already open. Please close to create a new plot.", QtGui.QMessageBox.Ok)
        else:
            plt.figure(1, facecolor='white')
            plt.subplot(211)
            plt.plot(speed, Tm)
            plt.xlim([0, globals.motor_data["sync_speed"]])
            plt.xlabel("Speed (rpm)")
            plt.ylabel("Torque (T/Tn)")
            plt.grid(color = '0.75', linestyle='--', linewidth=1)
            
            plt.subplot(212)
            plt.plot(speed, Im, 'r')
            plt.xlim([0, globals.motor_data["sync_speed"]])
            plt.xlabel("Speed (rpm)")
            plt.ylabel("Current (pu)")
            plt.grid(color = '0.75', linestyle='--', linewidth=1)
            
            plt.show()
    
    # Update global variables on change in data fields
    def update_data(self):
        globals.motor_data["description"] = str(self.le1.text())
        globals.motor_data["sync_speed"] = float(self.le2.text())
        globals.motor_data["rated_speed"] = float(self.le3.text())
        globals.motor_data["rated_pf"] = float(self.le4.text())
        globals.motor_data["rated_eff"] = float(self.le5.text())
        globals.motor_data["T_b"] = float(self.le6.text())
        globals.motor_data["T_lr" ] = float(self.le7.text())
        globals.motor_data["I_lr"] = float(self.le8.text())
        globals.algo_data["max_iter"] = int(self.le9.text())
        globals.algo_data["conv_err"] = float(self.le10.text())
        globals.algo_data["k_r"] = float(self.le11.text())
        globals.algo_data["k_x"] = float(self.le12.text())
        globals.algo_data["n_gen"] = int(self.len_gen.text())
        globals.algo_data["pop"] = int(self.lepop.text())
        globals.algo_data["n_r"] = int(self.len_r.text())
        globals.algo_data["n_e"] = int(self.len_e.text())
        globals.algo_data["c_f"] = float(self.lec_f.text())
    
    # Update data in the main window
    def update_window(self):
        self.le1.setText(str(globals.motor_data["description"]))
        self.le2.setText(str(globals.motor_data["sync_speed"]))
        self.le3.setText(str(globals.motor_data["rated_speed"]))
        self.le4.setText(str(globals.motor_data["rated_pf"]))
        self.le5.setText(str(globals.motor_data["rated_eff"]))
        self.le6.setText(str(globals.motor_data["T_b"]))
        self.le7.setText(str(globals.motor_data["T_lr"]))
        self.le8.setText(str(globals.motor_data["I_lr"]))
        
        self.le9.setText(str(globals.algo_data["max_iter"]))
        self.le10.setText(str(globals.algo_data["conv_err"]))
        self.le11.setText(str(globals.algo_data["k_r"]))
        self.le12.setText(str(globals.algo_data["k_x"]))
        self.len_gen.setText(str(globals.algo_data["n_gen"]))
        self.lepop.setText(str(globals.algo_data["pop"]))
        self.len_r.setText(str(globals.algo_data["n_r"]))
        self.len_e.setText(str(globals.algo_data["n_e"]))
        self.lec_f.setText(str(globals.algo_data["c_f"]))
    
    # Update the screen if the algorithm changes
    def update_algo(self):
        if (self.combo_algo.currentText() == "Genetic Algorithm") or (self.combo_algo.currentText() == "Hybrid GA-LM") or (self.combo_algo.currentText() == "Hybrid GA-NR") or (self.combo_algo.currentText() == "Hybrid GA-DNR"):
                self.label11.setVisible(0)
                self.le11.hide()
                self.label12.setVisible(0)
                self.le12.hide()
                
                self.labeln_gen.setVisible(1)
                self.labelpop.setVisible(1)
                self.labeln_r.setVisible(1)
                self.labeln_e.setVisible(1)
                self.labelc_f.setVisible(1)
                self.len_gen.show()
                self.lepop.show()
                self.len_r.show()
                self.len_e.show()
                self.lec_f.show()
        else:
                self.label11.setVisible(1)
                self.le11.show()
                self.label12.setVisible(1)
                self.le12.show()
                
                self.labeln_gen.setVisible(0)
                self.labelpop.setVisible(0)
                self.labeln_r.setVisible(0)
                self.labeln_e.setVisible(0)
                self.labelc_f.setVisible(0)
                self.len_gen.hide()
                self.lepop.hide()
                self.len_r.hide()
                self.len_e.hide()
                self.lec_f.hide()
    
    # Update if model combo box changed
    def update_model(self):
        if self.combo_model.currentIndex() == 0:
            # Single cage
            self.set_motor_image('single_cage.png')
            self.combo_algo.setCurrentIndex(0)
            self.combo_algo.clear()
            self.combo_algo.addItem("Newton-Raphson")
            self.label18.setVisible(0)
            self.label19.setVisible(0)
            self.leXr2.hide()
            self.leRr2.hide()
        else:
            # Double cage
            self.set_motor_image('dbl_cage.png')
            self.combo_algo.addItem("Levenberg-Marquardt")
            self.combo_algo.addItem("Damped Newton-Raphson")
            self.combo_algo.addItem("Genetic Algorithm")
            self.combo_algo.addItem("Hybrid GA-NR")
            self.combo_algo.addItem("Hybrid GA-LM")
            self.combo_algo.addItem("Hybrid GA-DNR")
            self.label18.setVisible(1)
            self.label19.setVisible(1)
            self.leXr2.show()
            self.leRr2.show()

    def set_motor_image(self, image_name):
        self.motor_pixmap = QtGui.QPixmap(resource_path('images', image_name))
        self.update_motor_image()

    def update_motor_image(self):
        if not self.motor_pixmap or self.motor_pixmap.isNull():
            return

        size = self.img1.size()
        if size.width() <= 1 or size.height() <= 1:
            self.img1.setPixmap(self.motor_pixmap)
            return

        scaled = self.motor_pixmap.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.img1.setPixmap(scaled)

    def resizeEvent(self, event):
        super(Window, self).resizeEvent(event)
        self.update_motor_image()

    def build_lab_form_group(self, title, fields, target_dict, read_only=False, min_label_width=190):
        box = QtGui.QGroupBox(title)
        layout = QtGui.QGridLayout()
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(12)
        layout.setContentsMargins(14, 18, 14, 14)

        box.setStyleSheet(
            "QGroupBox {"
            " background-color: #c7d8ec;"
            " border: 1px solid #b0c2d8;"
            " margin-top: 12px;"
            " font-weight: bold;"
            " font-size: 14px;"
            " color: #111111;"
            "}"
            "QGroupBox::title {"
            " subcontrol-origin: margin;"
            " left: 10px;"
            " padding: 0 4px;"
            "}"
            "QLabel {"
            " font-size: 12px;"
            " color: #111111;"
            " font-weight: bold;"
            "}"
            "QLineEdit {"
            " background: white;"
            " border: 1px solid #8ea6bf;"
            " min-height: 24px;"
            " padding: 2px 6px;"
            "}"
            "QLineEdit[readOnly=\"true\"] {"
            " background: #eef4fb;"
            " color: #22384a;"
            "}"
        )

        for row, (key, label) in enumerate(fields):
            label_widget = QtGui.QLabel(label)
            label_widget.setWordWrap(True)
            label_widget.setMinimumWidth(min_label_width)
            label_widget.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

            field = QtGui.QLineEdit()
            field.setReadOnly(read_only)
            field.setMinimumWidth(120)
            field.setMaximumWidth(150)
            target_dict[key] = field
            layout.addWidget(label_widget, row, 0)
            layout.addWidget(field, row, 1)

        layout.setColumnStretch(0, 1)
        layout.setColumnMinimumWidth(1, 130)

        box.setLayout(layout)
        return box

    def create_lab_tab(self):
        self.lab_model_combo = QtGui.QComboBox()
        self.lab_model_combo.addItem('Single cage')
        self.lab_model_combo.addItem('Double cage')

        self.lab_algo_combo = QtGui.QComboBox()

        control_box = QtGui.QGroupBox('Laboratory Estimation')
        control_layout = QtGui.QGridLayout()
        control_layout.setContentsMargins(14, 18, 14, 14)
        control_layout.setHorizontalSpacing(12)
        control_layout.setVerticalSpacing(10)
        control_box.setStyleSheet(
            "QGroupBox {"
            " background-color: #eef4fb;"
            " border: 1px solid #c7d8ec;"
            " margin-top: 12px;"
            " font-weight: bold;"
            " font-size: 14px;"
            "}"
            "QGroupBox::title {"
            " subcontrol-origin: margin;"
            " left: 10px;"
            " padding: 0 4px;"
            "}"
            "QPushButton {"
            " min-height: 28px;"
            " padding: 3px 10px;"
            "}"
            "QComboBox, QLineEdit {"
            " min-height: 24px;"
            " padding: 2px 6px;"
            "}"
        )
        control_layout.addWidget(QtGui.QLabel('Model'), 0, 0)
        control_layout.addWidget(self.lab_model_combo, 0, 1)
        control_layout.addWidget(QtGui.QLabel('Algorithm'), 0, 2)
        control_layout.addWidget(self.lab_algo_combo, 0, 3)

        self.lab_calc_button = QtGui.QPushButton('Calculate from Tests')
        self.lab_plot_button = QtGui.QPushButton('Plot Laboratory Curves')
        self.lab_save_data_button = QtGui.QPushButton('Save Data')
        self.lab_save_graph_button = QtGui.QPushButton('Save Graph')

        control_layout.addWidget(self.lab_calc_button, 1, 0, 1, 2)
        control_layout.addWidget(self.lab_plot_button, 1, 2)
        control_layout.addWidget(self.lab_save_data_button, 1, 3)
        control_layout.addWidget(self.lab_save_graph_button, 1, 4)
        control_layout.setColumnStretch(5, 1)
        control_box.setLayout(control_layout)

        object_fields = [
            ('manufacturer', 'Fabricante'),
            ('serial_number', 'Nr Serie'),
            ('manufacturing_year', 'Ano Fabricacao'),
            ('rated_power_w', 'Potencia [W]'),
            ('rated_voltage_v', 'Tensao [V]'),
            ('rated_current_a', 'Corrente [A]'),
            ('pole_count', 'Nr Polos'),
            ('frequency_hz', 'Frequencia [Hz]'),
            ('rated_speed_rpm', 'Rotacao [rpm]'),
        ]
        blocked_fields = [
            ('blocked_temp_c', 'Temperatura [C]'),
            ('blocked_resistance_ohm', 'Resistencia [Ohm]'),
            ('blocked_voltage_v', 'Tensao [V]'),
            ('blocked_current_a', 'Corrente [A]'),
            ('blocked_power_w', 'Potencia [W]'),
            ('blocked_frequency_hz', 'Frequencia [Hz]'),
        ]
        no_load_fields = [
            ('no_load_temp_c', 'Temperatura [C]'),
            ('no_load_voltage_v', 'Tensao [V]'),
            ('no_load_current_a', 'Corrente [A]'),
            ('no_load_power_w', 'Potencia [W]'),
            ('fw_loss_w', 'Perdas atrito/vent. [W]'),
        ]
        result_fields = [
            ('Rs', 'Rs [Ohm]'),
            ('Xs', 'Xs [Ohm]'),
            ('Xm', 'Xm [Ohm]'),
            ('Rr1', 'Rr1 [Ohm]'),
            ('Xr1', 'Xr1 [Ohm]'),
            ('Rr2', 'Rr2 [pu/Ohm]'),
            ('Xr2', 'Xr2 [pu/Ohm]'),
            ('Rc', 'Rc [Ohm]'),
        ]
        summary_fields = [
            ('locked_current_a', 'Corrente de Partida [A]'),
            ('locked_torque_nm', 'Conjugado de Partida [Nm]'),
            ('breakdown_torque_nm', 'Conjugado Maximo [Nm]'),
            ('rated_eff_pct', 'Rendimento [%]'),
            ('rated_pf', 'Fator de Potencia'),
            ('converged', 'Convergiu?'),
            ('error', 'Erro'),
        ]

        object_box = self.build_lab_form_group('Dados do Objeto Sob Teste', object_fields, self.lab_input_fields)
        blocked_box = self.build_lab_form_group('Dados da Medicao Com Rotor Bloqueado', blocked_fields, self.lab_input_fields, min_label_width=215)
        no_load_box = self.build_lab_form_group('Dados da Medicao Em Vazio', no_load_fields, self.lab_input_fields, min_label_width=215)
        result_box = self.build_lab_form_group('Parametros Estimados', result_fields, self.lab_result_fields, read_only=True, min_label_width=180)
        summary_box = self.build_lab_form_group('Desempenho Estimado', summary_fields, self.lab_result_fields, read_only=True, min_label_width=180)

        object_box.setMinimumWidth(360)
        blocked_box.setMinimumWidth(430)
        no_load_box.setMinimumWidth(430)

        right_column = QtGui.QWidget()
        right_layout = QtGui.QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(blocked_box)
        right_layout.addWidget(no_load_box)
        right_layout.addStretch(1)
        right_column.setLayout(right_layout)

        results_row = QtGui.QWidget()
        results_layout = QtGui.QHBoxLayout()
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(12)
        results_layout.addWidget(result_box)
        results_layout.addWidget(summary_box)
        results_row.setLayout(results_layout)

        body = QtGui.QWidget()
        body_layout = QtGui.QGridLayout()
        body_layout.setContentsMargins(12, 12, 12, 12)
        body_layout.setHorizontalSpacing(12)
        body_layout.setVerticalSpacing(12)
        body.setStyleSheet("background-color: #f4f7fb;")

        body_layout.addWidget(control_box, 0, 0, 1, 2)
        body_layout.addWidget(object_box, 1, 0)
        body_layout.addWidget(right_column, 1, 1)
        body_layout.addWidget(results_row, 2, 0, 1, 2)
        body_layout.setColumnMinimumWidth(0, 360)
        body_layout.setColumnStretch(0, 0)
        body_layout.setColumnStretch(1, 1)
        body_layout.setAlignment(QtCore.Qt.AlignTop)
        body.setLayout(body_layout)

        scroll_area = QtGui.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtGui.QFrame.NoFrame)
        scroll_area.setWidget(body)

        self.lab_plot_button.setDisabled(1)
        self.lab_save_data_button.setDisabled(1)
        self.lab_save_graph_button.setDisabled(1)

        self.lab_input_fields['frequency_hz'].setText('50')
        self.lab_input_fields['pole_count'].setText('4')

        self.lab_model_combo.currentIndexChanged.connect(self.update_lab_model)
        self.lab_calc_button.clicked.connect(self.calculate_lab)
        self.lab_plot_button.clicked.connect(self.plot_lab_results)
        self.lab_save_data_button.clicked.connect(self.save_lab_data)
        self.lab_save_graph_button.clicked.connect(self.save_lab_graph)
        self.update_lab_model()

        return scroll_area

    def update_lab_model(self):
        self.lab_algo_combo.clear()
        if self.lab_model_combo.currentIndex() == 0:
            self.lab_algo_combo.addItem('Chapman Direct')
            self.lab_algo_combo.setDisabled(1)
        else:
            self.lab_algo_combo.setDisabled(0)
            self.lab_algo_combo.addItem('Newton-Raphson')
            self.lab_algo_combo.addItem('Levenberg-Marquardt')
            self.lab_algo_combo.addItem('Damped Newton-Raphson')
            self.lab_algo_combo.addItem('Genetic Algorithm')
            self.lab_algo_combo.addItem('Hybrid GA-NR')
            self.lab_algo_combo.addItem('Hybrid GA-LM')
            self.lab_algo_combo.addItem('Hybrid GA-DNR')

    def lab_input_data(self):
        text_fields = ['manufacturer', 'serial_number', 'manufacturing_year']
        int_fields = {'pole_count': 'Nr Polos'}
        float_fields = {
            'rated_power_w': 'Potencia [W]',
            'rated_voltage_v': 'Tensao [V]',
            'rated_current_a': 'Corrente [A]',
            'frequency_hz': 'Frequencia [Hz]',
            'rated_speed_rpm': 'Rotacao [rpm]',
            'blocked_temp_c': 'Temperatura rotor bloqueado [C]',
            'blocked_resistance_ohm': 'Resistencia [Ohm]',
            'blocked_voltage_v': 'Tensao rotor bloqueado [V]',
            'blocked_current_a': 'Corrente rotor bloqueado [A]',
            'blocked_power_w': 'Potencia rotor bloqueado [W]',
            'blocked_frequency_hz': 'Frequencia rotor bloqueado [Hz]',
            'no_load_temp_c': 'Temperatura em vazio [C]',
            'no_load_voltage_v': 'Tensao em vazio [V]',
            'no_load_current_a': 'Corrente em vazio [A]',
            'no_load_power_w': 'Potencia em vazio [W]',
            'fw_loss_w': 'Perdas por atrito e ventilacao [W]',
        }

        data = {}
        for key in text_fields:
            data[key] = str(self.lab_input_fields[key].text()).strip()

        for key, label in int_fields.items():
            text = str(self.lab_input_fields[key].text()).strip()
            if not text:
                raise ValueError('%s deve ser informado.' % label)
            data[key] = int(text)

        for key, label in float_fields.items():
            text = str(self.lab_input_fields[key].text()).strip()
            if not text:
                raise ValueError('%s deve ser informado.' % label)
            data[key] = float(text)

        return data

    def solve_lab_double_cage(self, targets):
        p = [targets['sf'], targets['rated_eff'], targets['rated_pf'], targets['T_b'], targets['T_lr'], targets['I_lr']]
        algo_name = self.lab_algo_combo.currentText()

        if algo_name == 'Newton-Raphson':
            return nr_solver(p, 0, globals.algo_data['k_x'], globals.algo_data['k_r'], globals.algo_data['max_iter'], globals.algo_data['conv_err'])
        if algo_name == 'Levenberg-Marquardt':
            return lm_solver(p, 0, globals.algo_data['k_x'], globals.algo_data['k_r'], 1e-7, 5.0, globals.algo_data['max_iter'], globals.algo_data['conv_err'])
        if algo_name == 'Damped Newton-Raphson':
            return dnr_solver(p, 0, globals.algo_data['k_x'], globals.algo_data['k_r'], 1e-7, globals.algo_data['max_iter'], globals.algo_data['conv_err'])
        if algo_name == 'Genetic Algorithm':
            return ga_solver(self, p, globals.algo_data['pop'], globals.algo_data['n_r'], globals.algo_data['n_e'], globals.algo_data['c_f'], globals.algo_data['n_gen'], globals.algo_data['conv_err'])
        if algo_name == 'Hybrid GA-NR':
            return hy_solver(self, 'NR', p, globals.algo_data['pop'], globals.algo_data['n_r'], globals.algo_data['n_e'], globals.algo_data['c_f'], globals.algo_data['n_gen'], globals.algo_data['conv_err'])
        if algo_name == 'Hybrid GA-LM':
            return hy_solver(self, 'LM', p, globals.algo_data['pop'], globals.algo_data['n_r'], globals.algo_data['n_e'], globals.algo_data['c_f'], globals.algo_data['n_gen'], globals.algo_data['conv_err'])
        return hy_solver(self, 'DNR', p, globals.algo_data['pop'], globals.algo_data['n_r'], globals.algo_data['n_e'], globals.algo_data['c_f'], globals.algo_data['n_gen'], globals.algo_data['conv_err'])

    def build_lab_single_results(self, lab_data, params, summary, curves, load_points):
        return {
            'mode': 'single',
            'model_name': 'Single cage',
            'algorithm': 'Chapman Direct',
            'input_data': lab_data,
            'params': {
                'Rs': params['Rs'],
                'Xs': params['Xs'],
                'Xm': params['Xm'],
                'Rr1': params['Rr1'],
                'Xr1': params['Xr1'],
                'Rr2': None,
                'Xr2': None,
                'Rc': params['Rc'],
            },
            'summary': {
                'locked_current_a': summary['locked_rotor']['current_a'],
                'locked_torque_nm': summary['locked_rotor']['torque_nm'],
                'breakdown_torque_nm': summary['breakdown']['torque_nm'],
                'rated_eff_pct': summary['rated']['efficiency'] * 100.0,
                'rated_pf': summary['rated']['power_factor'],
                'converged': 'Direct',
                'error': 0.0,
            },
            'curves': curves,
            'load_points': load_points,
        }

    def build_lab_double_results(self, lab_data, vector, targets, algorithm_name, iterations, error, converged):
        vector = np.abs(vector)
        torque_base = targets['rated_eff'] * targets['rated_pf'] / (1.0 - targets['sf'])
        sync_speed_rpm = synchronous_speed(lab_data['frequency_hz'], lab_data['pole_count'])
        rated_torque_nm = rated_torque(lab_data['rated_power_w'], lab_data['rated_speed_rpm'])

        def performance_at_slip(slip):
            slip = min(max(slip, 1e-4), 1.0)
            torque_pu, current_no_core = get_torque(slip, vector)
            core_current = 1.0 / complex(vector[7], 0.0)
            total_current = abs(current_no_core + core_current)
            pqt = calc_pqt(slip, vector)
            pin_pu = pqt[0] / pqt[5] if pqt[5] > 0 else 0.0
            pf = pin_pu / np.sqrt(pin_pu ** 2 + pqt[1] ** 2) if (pin_pu > 0 or pqt[1] > 0) else 0.0
            return {
                'slip': slip,
                'speed_rpm': sync_speed_rpm * (1.0 - slip),
                'torque_nm': (torque_pu / torque_base) * rated_torque_nm,
                'current_a': total_current * lab_data['rated_current_a'],
                'power_factor': pf,
                'efficiency': pqt[5],
            }

        slips = np.linspace(1.0, 1e-4, 500)
        points = [performance_at_slip(float(slip)) for slip in slips]
        load_points = load_point_summary(lab_data, performance_at_slip)
        breakdown_point = max(points, key=lambda point: point['torque_nm'])
        locked_point = performance_at_slip(1.0)
        rated_point = performance_at_slip(targets['sf'])

        return {
            'mode': 'double',
            'model_name': 'Double cage',
            'algorithm': algorithm_name,
            'input_data': lab_data,
            'params': {
                'Rs': float(vector[0]),
                'Xs': float(vector[1]),
                'Xm': float(vector[2]),
                'Rr1': float(vector[3]),
                'Xr1': float(vector[4]),
                'Rr2': float(vector[5]),
                'Xr2': float(vector[6]),
                'Rc': float(vector[7]),
            },
            'summary': {
                'locked_current_a': locked_point['current_a'],
                'locked_torque_nm': locked_point['torque_nm'],
                'breakdown_torque_nm': breakdown_point['torque_nm'],
                'rated_eff_pct': rated_point['efficiency'] * 100.0,
                'rated_pf': rated_point['power_factor'],
                'converged': 'Yes' if converged == 1 else 'No',
                'error': float(error),
            },
            'curves': {
                'slip': np.array([point['slip'] for point in points]),
                'speed_rpm': np.array([point['speed_rpm'] for point in points]),
                'torque_nm': np.array([point['torque_nm'] for point in points]),
                'current_a': np.array([point['current_a'] for point in points]),
            },
            'load_points': load_points,
            'iterations': iterations,
        }

    def update_lab_result_fields(self):
        if not self.lab_results:
            return

        params = self.lab_results['params']
        summary = self.lab_results['summary']

        for key in ['Rs', 'Xs', 'Xm', 'Rr1', 'Xr1', 'Rr2', 'Xr2', 'Rc']:
            value = params[key]
            self.lab_result_fields[key].setText('' if value is None else str(np.round(value, 6)))

        self.lab_result_fields['locked_current_a'].setText(str(np.round(summary['locked_current_a'], 3)))
        self.lab_result_fields['locked_torque_nm'].setText(str(np.round(summary['locked_torque_nm'], 3)))
        self.lab_result_fields['breakdown_torque_nm'].setText(str(np.round(summary['breakdown_torque_nm'], 3)))
        self.lab_result_fields['rated_eff_pct'].setText(str(np.round(summary['rated_eff_pct'], 3)))
        self.lab_result_fields['rated_pf'].setText(str(np.round(summary['rated_pf'], 4)))
        self.lab_result_fields['converged'].setText(summary['converged'])
        self.lab_result_fields['error'].setText(str(np.round(summary['error'], 8)))

    def calculate_lab(self):
        self.statusBar().showMessage('Calculating laboratory model...')
        try:
            lab_data = self.lab_input_data()
            single_params = estimate_single_cage_parameters(lab_data)
            single_summary = single_cage_summary(lab_data, single_params)
            single_curves = single_cage_curves(lab_data, single_params)
            single_load_points = load_point_summary(
                lab_data,
                lambda slip: single_cage_performance(lab_data, single_params, slip)
            )

            if self.lab_model_combo.currentIndex() == 0:
                self.lab_results = self.build_lab_single_results(lab_data, single_params, single_summary, single_curves, single_load_points)
            else:
                vector, iterations, error, converged = self.solve_lab_double_cage(single_summary['targets'])
                self.lab_results = self.build_lab_double_results(
                    lab_data,
                    vector,
                    single_summary['targets'],
                    self.lab_algo_combo.currentText(),
                    iterations,
                    error,
                    converged,
                )

            self.lab_plot_figure = None
            self.update_lab_result_fields()
            self.lab_plot_button.setEnabled(1)
            self.lab_save_data_button.setEnabled(1)
            self.lab_save_graph_button.setDisabled(1)
            self.statusBar().showMessage('Ready')
        except Exception as exc:
            self.statusBar().showMessage('Ready')
            QtGui.QMessageBox.warning(self, 'Warning', str(exc), QtGui.QMessageBox.Ok)

    def plot_lab_results(self):
        if not self.lab_results:
            QtGui.QMessageBox.warning(self, 'Warning', 'Calculate laboratory parameters first.', QtGui.QMessageBox.Ok)
            return

        if self.lab_plot_figure is not None and plt.fignum_exists(self.lab_plot_figure.number):
            plt.close(self.lab_plot_figure)

        curves = self.lab_results['curves']
        load_points = self.lab_results['load_points']

        self.lab_plot_figure = plt.figure(facecolor='white', figsize=(10, 8))
        self.lab_plot_figure.suptitle('Laboratory Test Results - %s' % self.lab_results['model_name'])

        ax1 = self.lab_plot_figure.add_subplot(221)
        ax1.plot(curves['speed_rpm'], curves['torque_nm'])
        ax1.set_xlabel('Speed (rpm)')
        ax1.set_ylabel('Torque (Nm)')
        ax1.grid(color='0.75', linestyle='--', linewidth=1)

        ax2 = self.lab_plot_figure.add_subplot(222)
        ax2.plot(curves['slip'], curves['torque_nm'], 'r')
        ax2.set_xlabel('Slip (pu)')
        ax2.set_ylabel('Torque (Nm)')
        ax2.grid(color='0.75', linestyle='--', linewidth=1)

        ax3 = self.lab_plot_figure.add_subplot(223)
        ax3.plot(curves['speed_rpm'], curves['current_a'], 'g')
        ax3.set_xlabel('Speed (rpm)')
        ax3.set_ylabel('Current (A)')
        ax3.grid(color='0.75', linestyle='--', linewidth=1)

        ax4 = self.lab_plot_figure.add_subplot(224)
        load_pct = [point['load_fraction'] * 100.0 for point in load_points]
        efficiency = [point['efficiency'] * 100.0 for point in load_points]
        power_factor = [point['power_factor'] for point in load_points]
        ax4.plot(load_pct, efficiency, marker='o', label='Rendimento (%)')
        ax4.plot(load_pct, power_factor, marker='s', label='Fator de Potencia')
        ax4.set_xlabel('Load (%)')
        ax4.set_ylabel('Value')
        ax4.grid(color='0.75', linestyle='--', linewidth=1)
        ax4.legend(loc='best')

        self.lab_plot_figure.tight_layout(rect=[0, 0, 1, 0.96])
        self.lab_save_graph_button.setEnabled(1)
        plt.show()

    def serializable_lab_results(self):
        if not self.lab_results:
            return None

        curves = self.lab_results['curves']
        return {
            'model': self.lab_results['model_name'],
            'algorithm': self.lab_results['algorithm'],
            'input_data': self.lab_results['input_data'],
            'params': self.lab_results['params'],
            'summary': self.lab_results['summary'],
            'load_points': self.lab_results['load_points'],
            'curves': {
                'slip': curves['slip'].tolist(),
                'speed_rpm': curves['speed_rpm'].tolist(),
                'torque_nm': curves['torque_nm'].tolist(),
                'current_a': curves['current_a'].tolist(),
            },
        }

    def save_lab_data(self):
        if not self.lab_results:
            QtGui.QMessageBox.warning(self, 'Warning', 'No laboratory data is available to save.', QtGui.QMessageBox.Ok)
            return

        filename = dialog_path(QtGui.QFileDialog.getSaveFileName(
            self,
            'Save Laboratory Data',
            resource_path('library'),
            'JSON files (*.json);;Text files (*.txt)'
        ))

        if not filename:
            return

        data = self.serializable_lab_results()
        if filename.lower().endswith('.txt'):
            with open(filename, 'w') as handle:
                handle.write('Model;%s\n' % data['model'])
                handle.write('Algorithm;%s\n' % data['algorithm'])
                for key, value in sorted(data['input_data'].items()):
                    handle.write('Input.%s;%s\n' % (key, value))
                for key, value in sorted(data['params'].items()):
                    handle.write('Param.%s;%s\n' % (key, value))
                for key, value in sorted(data['summary'].items()):
                    handle.write('Summary.%s;%s\n' % (key, value))
        else:
            if not filename.lower().endswith('.json'):
                filename = filename + '.json'
            with open(filename, 'w') as handle:
                json.dump(data, handle, indent=2)

    def save_lab_graph(self):
        if self.lab_plot_figure is None or not plt.fignum_exists(self.lab_plot_figure.number):
            QtGui.QMessageBox.warning(self, 'Warning', 'Generate the laboratory plots before saving the graph.', QtGui.QMessageBox.Ok)
            return

        filename = dialog_path(QtGui.QFileDialog.getSaveFileName(
            self,
            'Save Laboratory Graph',
            resource_path('library'),
            'PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)'
        ))

        if not filename:
            return

        self.lab_plot_figure.savefig(filename, bbox_inches='tight')
    
    # Open file and load motor data
    def load_action(self):
        # Open file dialog box
        filename = dialog_path(QtGui.QFileDialog.getOpenFileName(self, "Open Moto File", resource_path('library'), "Moto files (*.mto)"))
        
        if filename:
            saveload.load_file(filename)
            self.update_window()
    
    # Save motor data to file
    def save_action(self):
        # Open save file dialog box
        filename = dialog_path(QtGui.QFileDialog.getSaveFileName(self, "Save Moto File", resource_path('library'), "Moto files (*.mto)"))
        
        if filename:
            saveload.save_file(filename)
    
    # Launch user manual
    def user_manual(self):
        os.startfile(resource_path('docs', 'moto_user_manual.pdf'))
    
    # About dialog box
    def about_dialog(self):
        sigma_logo = resource_path('images', 'Sigma_Power.png').replace('\\', '/')
        QtGui.QMessageBox.about(self, "About Moto",
                """<b>Moto</b> is a parameter estimation tool that can be used to determine the equivalent circuit parameters of induction machines. The tool is intended for use in dynamic time-domain simulations such as stability and motor starting studies.
                   <p>
                   Version: <b>v0.2<b><P>
                   <p>
                   Website: <a href="http://www.sigmapower.com.au/moto.html">www.sigmapower.com.au/moto.html</a>
                   <p> </p>
               <p><img src="%s"></p>
                   <p>&copy; 2014 Sigma Power Engineering Pty Ltd</p>
                   <p>All rights reserved.</p>             
               """ % sigma_logo)
    
    # Centre application window on screen
    def centre(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

def main():
    
    app = QtGui.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()