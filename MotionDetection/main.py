import sys

import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi


class projectTen(QDialog):
    def __init__(self):
        super(projectTen, self).__init__()
        loadUi('motionwebcam.ui', self)
        self.image = None

        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.motionButton.toggled.connect(self.detect_webcam_motion)
        self.motionButton.setCheckable(True)
        self.motion_Enabled = False
        self.motImgButton.clicked.connect(self.set_motion_image)
        self.motionFrame = None


    def detect_webcam_motion(self,status):
        if status:
            self.motion_Enabled = True
            self.motionButton.setText('Stop Motion')
        else:
            self.motion_Enabled = False
            self.motionButton.setText('Detect Motion')


    def set_motion_image(self):
       gray = cv2.cvtColor(self.image.copy(),cv2.COLOR_BGR2GRAY)
       gray = cv2.GaussianBlur(gray,(21,21),0)
       self.motionFrame = gray
       self.displayImage(self.motionFrame,2)


    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        if (self.motion_Enabled):
            detected_motion=self.detect_motion(self.image)
            self.displayImage(detected_motion,1)
        else:
            self.displayImage(self.image, 1)


    def detect_motion(self,input_img):
        self.text='No motion'
        gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(21,21),0)

        frameDiff =  cv2.absdiff(self.motionFrame,gray)
        thresh = cv2.threshold(frameDiff,40,255,
                               cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh,None,iterations=5)

        im2,cnts,hierarchy = cv2.findContours(thresh.copy(),
                                             cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)

        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []


        height,width,channels = input_img.shape
        min_x,min_y = width,height
        max_x = max_y=0


        for contour,hier in zip(cnts,hierarchy):
            (x,y,w,h) = cv2.boundingRect(contour)
            min_x,max_x = min(x,min_x),max(x+w,max_x)
            min_y,max_y = min(y,min_y),max(y+h,max_y)

        if max_x-min_x>80 and max_y-min_y > 80:
            cv2.rectangle(input_img,(min_x,min_y),(max_x,max_y),
                          (0,255,0,),3)

            self.text='Motion Detected'

        cv2.putText(input_img,'Motion Status: {}'.format(
            self.text),(10,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        return input_img



    def stop_webcam(self):
        self.timer.stop()
        self.capture = cv2.VideoCapture(0)

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)

        outImage = outImage.rgbSwapped();
        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)
        if window == 2:
            self.motionLabel.setPixmap(QPixmap.fromImage(outImage))
            self.motionLabel.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = projectTen()
    window.setWindowTitle('Webcam')
    window.show()
    sys.exit(app.exec_())


