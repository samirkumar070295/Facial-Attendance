# -*- coding: utf-8 -*-
import sys
from PyQt5.QtCore import QSize
from threading import Thread
from PyQt5.QtGui import QImage, QPalette, QBrush,QMovie, QPainter, QPixmap
from PyQt5.QtWidgets import *
from PyQt5 import QtGui,QtWidgets,QtCore
#from win3 import ChildWindow2
#from rec2 import recognize
import pandas as pd
#from PandasModel import PandasModel
from PyQt5 import QtCore, QtGui, QtWidgets
#from loading import ChildWindow4
import threading
import sklearn.ensemble
import sklearn.neighbors.typedefs
import sklearn.tree
import sklearn.tree._criterion
import sklearn.tree._splitter
import sklearn.tree._tree
import sklearn.neighbors.ball_tree
import sklearn.neighbors.quad_tree
import sklearn.tree._utils
#import sklearn.neighbors.dist_metrics

class ChildWindow1(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        self.title = 'Facial Recognition' 
        super(ChildWindow1, self).__init__(parent)
        self.setGeometry(100, 200, 200, 100)
        self.setFixedSize(500,500)
        self.setWindowTitle(self.title)
        self.movie = QMovie("background2.gif")
        self.movie.frameChanged.connect(self.repaint)
        self.movie.start()
        
        
        
        
        
        
        oImage = QImage("background2.gif")
        sImage = oImage.scaled(QSize(500,500))
        palette = QPalette()                   # resize Image to widgets sizepalette = QPalette()
        palette.setBrush(10, QBrush(sImage))                     # 10 = Windowrole
        self.setPalette(palette)
        self.button1 =QtWidgets.QPushButton('Start Recognition', self)
        self.button1.move(200,250)
        self.button1.clicked.connect(self.loadModel)
        
        #self.button2.clicked.connect(self.reports)
   
    def loadModel(self):
        print("clicked")
        self.recog()
        from realtime_facenet_git import loadModel
        t1=threading.Thread(target=loadModel)
                                    
        t1.start()
        
    
    
    def recog(self):
        #recognize()
        
        self.movie = QMovie("loading.gif")
        self.movie.frameChanged.connect(self.repaint)
        self.movie.start()
        
    
        
     
    def paintEvent(self, event):
        currentFrame = self.movie.currentPixmap()
        frameRect = currentFrame.rect()
        frameRect.moveCenter(self.rect().center())
        if frameRect.intersects(event.rect()):
            painter = QPainter(self)
            painter.drawPixmap(frameRect.left(), frameRect.top(), currentFrame)
            
    
if __name__ == '__main__':

    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ChildWindow1()
    window.setGeometry(500, 100, 800,100)
    window.show()
    app.exec_()
    sys.exit(app.exec_())         
       
        
     
    
        
