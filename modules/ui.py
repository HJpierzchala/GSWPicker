import os
from PyQt5 import QtWidgets, QtCore
import subprocess
from PyQt5.QtGui import QDoubleValidator
import shutil
from rich.progress import Progress
from time import sleep
from datetime import datetime
from .config import CFG
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, CFG):
        super().__init__()
        self.setWindowTitle("GSWPicker")
        self.setGeometry(100, 100, 900, 700)
        self.CFG = CFG
        self.initUI()

    def initUI(self):
        # --- Tabs ---
        self.tab_widget = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Processing settings tab
        self.tab1 = QtWidgets.QWidget()
        self.tab1_layout = QtWidgets.QGridLayout(self.tab1)
        self.tab_widget.addTab(self.tab1, "Processing settings")

        # Plotting settings tab
        self.tab2 = QtWidgets.QWidget()
        self.tab2_layout = QtWidgets.QVBoxLayout(self.tab2)
        self.tab_widget.addTab(self.tab2, "Plotting settings")

        # --- Plotting UI ---
        self.domain_label = QtWidgets.QLabel("Pre-inspect:")
        self.domain_selector = QtWidgets.QComboBox()
        self.domain_selector.addItems(["Time domain", "Frequency domain", "Noise analysis"])
        self.domain_label.setEnabled(False)
        self.domain_selector.setEnabled(False)
        self.tab2_layout.addWidget(self.domain_label)
        self.tab2_layout.addWidget(self.domain_selector)

        self.swave_arrivals_checkbox = QtWidgets.QCheckBox("S-wave arrivals")
        self.zoomed_swave_arrivals_checkbox = QtWidgets.QCheckBox("Zoomed in S-wave arrivals")
        self.tab2_layout.addWidget(self.swave_arrivals_checkbox)
        self.tab2_layout.addWidget(self.zoomed_swave_arrivals_checkbox)

        self.bp_section_label = QtWidgets.QLabel("Bandpass parameters:")
        font = self.bp_section_label.font()
        font.setBold(True)
        self.bp_section_label.setFont(font)
        self.tab2_layout.addWidget(self.bp_section_label)

        self.time_cutoff_label = QtWidgets.QLabel("Time cutoff (min):")
        self.time_cutoff_entry = QtWidgets.QLineEdit("2")
        self.time_cutoff_label.setEnabled(False)
        self.time_cutoff_entry.setEnabled(False)
        tc_hbox = QtWidgets.QHBoxLayout()
        tc_hbox.addWidget(self.time_cutoff_label)
        tc_hbox.addWidget(self.time_cutoff_entry)
        self.tab2_layout.addLayout(tc_hbox)

        bp_hbox = QtWidgets.QHBoxLayout()
        self.lowcut_label = QtWidgets.QLabel("Lowcut (Hz):")
        self.lowcut_entry = QtWidgets.QLineEdit("0.1")
        self.highcut_label = QtWidgets.QLabel("Subtraction factor:")
        self.highcut_entry = QtWidgets.QLineEdit("0.1")
        bp_hbox.addWidget(self.lowcut_label)
        bp_hbox.addWidget(self.lowcut_entry)
        bp_hbox.addWidget(self.highcut_label)
        bp_hbox.addWidget(self.highcut_entry)
        self.tab2_layout.addLayout(bp_hbox)
        self.tab2_layout.addStretch()


        self.zoomed_swave_arrivals_checkbox.stateChanged.connect(self.update_zoomed_swave_fields)
        self.domain_selector.currentTextChanged.connect(self.update_domain_selection)
        self.update_zoomed_swave_fields(self.zoomed_swave_arrivals_checkbox.checkState())

        # --- Default values ---
        parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.axis_var = '1D'
        self.mode_var = 'MAD'
        self.wcase_var = 'None'
        self.vadase_path_var = os.path.join(parent, 'vel_data')
        self.date_str_var = datetime.today().strftime('%Y-%m-%d')
        self.compute_noise_var = True
        self.shaking_length_var = self.CFG['UI']['shaking_length_var']
        self.window_size_var = self.CFG['UI']['window_size_var']
        self.result_csv_var = os.path.join(parent, 'results')
        self.result_figures_var = os.path.join(parent, 'results')
        self.storage_option_var = 'Station dependent'

        # --- Processing UI ---
        r = 0
        # Project ID
        self.tab1_layout.addWidget(QtWidgets.QLabel('Project ID:'), r, 0)
        self.eq_lat_entry = QtWidgets.QLineEdit()
        self.tab1_layout.addWidget(self.eq_lat_entry, r, 1)
        r += 1

        # Axis
        self.tab1_layout.addWidget(QtWidgets.QLabel('Axis:'), r, 0)
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(['1D', '2D', '3D'])
        self.axis_combo.setCurrentText(self.axis_var)
        self.tab1_layout.addWidget(self.axis_combo, r, 1)
        r += 1

        # Mode
        self.tab1_layout.addWidget(QtWidgets.QLabel('Mode:'), r, 0)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(['MAD', 'SLOPE', 'W-TEST', 'PREINSPECT'])
        self.mode_combo.setCurrentText(self.mode_var)
        self.mode_combo.currentTextChanged.connect(self.update_mode_dependent_fields)
        self.tab1_layout.addWidget(self.mode_combo, r, 1)
        r += 1

        # Wcase
        self.tab1_layout.addWidget(QtWidgets.QLabel('Wcase:'), r, 0)
        self.wcase_combo = QtWidgets.QComboBox()
        self.wcase_combo.addItems(['sigmoid', 'trapez', 'boxcar', 'method2'])
        self.wcase_combo.setCurrentText(self.wcase_var)
        self.tab1_layout.addWidget(self.wcase_combo, r, 1)
        r += 1

        # Time scale
        self.tab1_layout.addWidget(QtWidgets.QLabel('Time scale:'), r, 0)
        self.time_ref_combo = QtWidgets.QComboBox()
        self.time_ref_combo.addItems(['UTC', 'GSOW', 'RELATIVE O.T'])
        self.time_ref_combo.setCurrentText('GSOW')
        self.tab1_layout.addWidget(self.time_ref_combo, r, 1)
        r += 1

        # Event date
        self.tab1_layout.addWidget(QtWidgets.QLabel('Event date (yyyy-mm-dd):'), r, 0)
        self.date_entry = QtWidgets.QLineEdit(self.date_str_var)
        self.date_entry.setValidator(QRegExpValidator(QRegExp(r'^\d{4}-\d{2}-\d{2}$'), self))
        self.tab1_layout.addWidget(self.date_entry, r, 1)
        r += 1

        # Event time
        self.tab1_layout.addWidget(QtWidgets.QLabel('Event time (HH:MM:SS.sss):'), r, 0)
        self.eq_time_entry = QtWidgets.QLineEdit()
        self.eq_time_entry.setValidator(QRegExpValidator(QRegExp(r'^\d{2}:\d{2}:\d{2}\.\d{3}$'), self))
        self.tab1_layout.addWidget(self.eq_time_entry, r, 1)
        r += 1

        # Vel_data folder
        self.tab1_layout.addWidget(QtWidgets.QLabel('Vel_data folder:'), r, 0)
        self.vadase_path_entry = QtWidgets.QLineEdit(self.vadase_path_var)
        btn1 = QtWidgets.QPushButton('Browse')
        btn1.clicked.connect(self.select_vadase_path)
        self.tab1_layout.addWidget(self.vadase_path_entry, r, 1)
        self.tab1_layout.addWidget(btn1, r, 2)
        r += 1

        # Compute Noise
        self.tab1_layout.addWidget(QtWidgets.QLabel('Compute Noise:'), r, 0)
        self.compute_noise_cb = QtWidgets.QCheckBox()
        self.compute_noise_cb.setChecked(self.compute_noise_var)
        self.tab1_layout.addWidget(self.compute_noise_cb, r, 1)
        r += 1

        # Shaking length
        self.tab1_layout.addWidget(QtWidgets.QLabel('Shaking length (sec):'), r, 0)
        self.shaking_length_entry = QtWidgets.QLineEdit(str(self.shaking_length_var))
        dv = QDoubleValidator(2.0, 999999.0, 2, self)
        dv.setNotation(QDoubleValidator.StandardNotation)
        self.shaking_length_entry.setValidator(dv)
        self.tab1_layout.addWidget(self.shaking_length_entry, r, 1)
        r += 1

        # Window size
        self.tab1_layout.addWidget(QtWidgets.QLabel('Window Size (sec):'), r, 0)
        self.window_size_entry = QtWidgets.QLineEdit(str(self.window_size_var))
        self.window_size_entry.setValidator(QRegExpValidator(QRegExp(r'^[0-9]+([\.,][0-9]*)?$'), self))
        self.tab1_layout.addWidget(self.window_size_entry, r, 1)
        r += 1

        # Output csv
        self.tab1_layout.addWidget(QtWidgets.QLabel('Output csv Folder:'), r, 0)
        self.result_csv_entry = QtWidgets.QLineEdit(self.result_csv_var)
        btn2 = QtWidgets.QPushButton('Browse')
        btn2.clicked.connect(self.select_result_csv)
        self.tab1_layout.addWidget(self.result_csv_entry, r, 1)
        self.tab1_layout.addWidget(btn2, r, 2)
        r += 1

        # Output figures
        self.tab1_layout.addWidget(QtWidgets.QLabel('Output figures Folder:'), r, 0)
        self.result_figures_entry = QtWidgets.QLineEdit(self.result_figures_var)
        btn3 = QtWidgets.QPushButton('Browse')
        btn3.clicked.connect(self.select_result_figures)
        self.tab1_layout.addWidget(self.result_figures_entry, r, 1)
        self.tab1_layout.addWidget(btn3, r, 2)
        r += 1

        # CSV storage
        self.tab1_layout.addWidget(QtWidgets.QLabel('CSV Storage Option:'), r, 0)
        self.storage_option_combo = QtWidgets.QComboBox()
        self.storage_option_combo.addItems(['Station dependent', 'All S-wave arrivals', 'Component wise S-wave arrivals'])
        self.storage_option_combo.setCurrentText(self.storage_option_var)
        self.tab1_layout.addWidget(self.storage_option_combo, r, 1)
        r += 1

        # Specify stations
        self.tab1_layout.addWidget(QtWidgets.QLabel('Specify Station Names:'), r, 0)
        self.specify_entry = QtWidgets.QLineEdit('ALL')
        self.tab1_layout.addWidget(self.specify_entry, r, 1)
        r += 1

        # Run button
        run_btn = QtWidgets.QPushButton('Run Script')
        run_btn.clicked.connect(self.run_script)
        self.tab1_layout.addWidget(run_btn, r, 0, 1, 3)

        # init state
        self.update_mode_dependent_fields(self.mode_combo.currentText())

    # Plotting updates
    def update_domain_selection(self, txt):
        if txt == 'Noise analysis':
            self.axis_combo.setCurrentText('1D')
            self.axis_combo.setEnabled(False)
        else:
            self.axis_combo.setEnabled(True)

    def update_zoomed_swave_fields(self, st):
        en = (st == QtCore.Qt.Checked)
        self.time_cutoff_label.setEnabled(en)
        self.time_cutoff_entry.setEnabled(en)
        self.lowcut_label.setEnabled(en)
        self.lowcut_entry.setEnabled(en)
        self.highcut_label.setEnabled(en)
        self.highcut_entry.setEnabled(en)

    # Processing updates
    def update_mode_dependent_fields(self, mode):
        is_pre = (mode == 'PREINSPECT')
        # Enable or disable processing controls
        for w in [self.compute_noise_cb, self.window_size_entry, self.shaking_length_entry, self.storage_option_combo]:
            w.setEnabled(not is_pre)
        # Wcase only for SLOPE
        okw = (mode == 'SLOPE') and not is_pre
        self.wcase_combo.setEnabled(okw)
        if not okw:
            self.wcase_combo.setCurrentText('None')
        # Plotting controls: domain selector enabled only in PREINSPECT
        self.domain_label.setEnabled(is_pre)
        self.domain_selector.setEnabled(is_pre)
        # Bandpass parameters only when zoomed is checked and not PREINSPECT
        zoomed = self.zoomed_swave_arrivals_checkbox.isChecked()
        en = zoomed and (not is_pre)
        self.time_cutoff_label.setEnabled(en)
        self.time_cutoff_entry.setEnabled(en)
        self.lowcut_label.setEnabled(en)
        self.lowcut_entry.setEnabled(en)
        self.highcut_label.setEnabled(en)
        self.highcut_entry.setEnabled(en)

        self.domain_label.setEnabled(is_pre)
        self.domain_selector.setEnabled(is_pre)
        # Ensure axis selector re‑enabled when leaving PREINSPECT/noise mode
        if not is_pre:
            self.axis_combo.setEnabled(True)
        # Bandpass parameters only when zoomed is checked and not PREINSPECT
        zoomed = self.zoomed_swave_arrivals_checkbox.isChecked()
        en = zoomed and (not is_pre)
        self.time_cutoff_label.setEnabled(en)
        self.time_cutoff_entry.setEnabled(en)
        self.lowcut_label.setEnabled(en)
        self.lowcut_entry.setEnabled(en)
        self.highcut_label.setEnabled(en)
        self.highcut_entry.setEnabled(en)


    # Path selectors
    def select_vadase_path(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select vel_data folder")
        if p:
            self.vadase_path_entry.setText(p)

    def select_result_csv(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select result csv folder")
        if p:
            self.result_csv_entry.setText(p)

    def select_result_figures(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select result figures folder")
        if p:
            self.result_figures_entry.setText(p)

    def run_script(self):
        # pre-run checks
        if not self.date_entry.hasAcceptableInput():
            QtWidgets.QMessageBox.warning(self, "Missing date", "Please specify event date in yyyy-mm-dd format.")
            return
        if self.time_ref_combo.currentText() == 'RELATIVE O.T' and not self.eq_time_entry.hasAcceptableInput():
            QtWidgets.QMessageBox.warning(self, "Missing time", "Please provide event origin time in HH:MM:SS.sss format.")
            return
        AXIS = self.axis_combo.currentText()
        MODE = self.mode_combo.currentText()
        parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script = 'pre.py' if MODE == 'PREINSPECT' else 'main_GUI.py'
        sp = os.path.join(parent, script)
        wcase = self.wcase_combo.currentText() if MODE == 'SLOPE' else 'None'
        vad = self.vadase_path_entry.text()
        dstr = self.date_entry.text()
        noise = str(self.compute_noise_cb.isChecked())
        shake = self.shaking_length_entry.text()
        win = self.window_size_entry.text()
        csvf = self.result_csv_entry.text()
        figf = self.result_figures_entry.text()
        store = self.storage_option_combo.currentText()
        lat = self.eq_lat_entry.text() or 'None'
        etime = self.eq_time_entry.text() or 'None'
        tref_map = {'UTC': 'utc', 'GSOW': 'gsow', 'RELATIVE O.T': 'rel'}
        tref = tref_map[self.time_ref_combo.currentText()]
        stns = self.specify_entry.text().strip()
        inst = 'false' if stns.upper() == 'ALL' else stns
        sw = str(self.swave_arrivals_checkbox.isChecked())
        zsw = str(self.zoomed_swave_arrivals_checkbox.isChecked())
        tco = self.time_cutoff_entry.text() if self.zoomed_swave_arrivals_checkbox.isChecked() else 'None'
        bp = 'True'
        lc = self.lowcut_entry.text() or '0.1'
        sf = self.highcut_entry.text() or '0.1'
        if MODE == 'PREINSPECT':
            dt = self.domain_selector.currentText()
            if dt == 'Frequency domain':
                dom = 'freq'
            elif dt == 'Time domain':
                dom = 'time'
            elif dt == 'Noise analysis':
                dom = 'noise'
            else:
                dom = 'None'
        else:
            dom = 'None'
        cmd = ['python', sp, AXIS, MODE, wcase, vad, dstr, noise, shake, win, inst, csvf, figf, store, lat, etime, sw, zsw, tco, bp, lc, sf, tref, dom]
        try:
            if float(shake) < 2.0:
                raise ValueError
        except:
            QtWidgets.QMessageBox.warning(self, "Invalid shaking", "Shaking length must be >=2.")
            return
        try:
            if float(win.replace(',', '.')) < 0.03:
                raise ValueError
        except:
            QtWidgets.QMessageBox.warning(self, "Invalid window", "Window size must be >=0.03 sec.")
            return
        subprocess.run(cmd)

# Utility functions

def clear_console():
    os.system('cls' if os.name=='nt' else 'clear')


def display_intro(console):
    terminal_width = shutil.get_terminal_size().columns
    title="GSWPicker"
    subtitle="GNSS-based S-wave Detector"
    sep="="*terminal_width
    console.print(f"[bold cyan]{sep}[/]")
    console.print(f"[bold cyan]{title}[/]",justify="center")
    console.print(f"[bold cyan]{subtitle}[/]",justify="center")
    console.print(f"[bold cyan]{sep}[/]")
    console.print(f"[green bold]Program run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    with Progress() as progress:
        task = progress.add_task("[cyan]Initializing program...", total=100)
        for i in range(100):
            sleep(0.02)
            progress.update(task, advance=1)
    console.print("\n[green bold]Initialization complete! Ready to process data.\n")


def display_closing(console):
    terminal_width = shutil.get_terminal_size().columns
    thank_you="Thank you for using GSWPicker Software!"
    message="Questions or feedback: hpierzchala@cbk.waw.pl, a.m.lapadat@tudelft.nl"
    sep="="*terminal_width
    console.print(f"[bold cyan]{sep}[/]")
    console.print(f"[green bold]{thank_you}[/]",justify="center")
    console.print(f"[cyan]{message}[/]",justify="center")
    console.print(f"[bold cyan]{sep}[/]\n")


def write_header(file, t):
    header = f"""
{"="*60}
                     GSWPicker v1.0.0
            GNSS-based S-wave Detection
{"="*60}
Program run: {t.strftime('%Y-%m-%d %H:%M:%S')}
{"-"*60}
"""
    file.write(header)
