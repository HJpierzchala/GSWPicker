from PyQt5 import QtWidgets, QtCore
import sys
from modules.ui import MainWindow
from modules.config import CFG
import os

if __name__ =='__main__':
    cwd = os.getcwd()
    os.makedirs(os.path.join(cwd, 'vel_data'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'results'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'errors'), exist_ok=True)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(CFG=CFG)
    window.show()
    sys.exit(app.exec_())