import base64
import hashlib
import io
import json
import sys
import time

import cv2
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox
from mtcnn_ort import MTCNN

import timkeeping
import torch
from torch.nn.functional import interpolate
from PIL import Image
from PIL.ImageQt import ImageQt
import numpy as np
import requests
import datetime
from datetime import datetime
from plus_function import *
import onnxruntime as ort



class MainWindow(QMainWindow):
    def __init__(self,url_checking, url_status, data_path):
        super().__init__()
        self.setWindowTitle("Hệ Thống Chấm Công")
        self.label = self.findChild(QLabel, "in_out")
        self.uic = timkeeping.Ui_TimekeepingSystem()
        self.uic.setupUi(self)
        self.setFixedSize(1042, 653)
        self.status = 0
        self.staff_code = ''
        self.staff_pass = ''


        self.url_checking = url_checking
        self.url_status = url_status
        self.data_path = data_path

        MainWindow = QtWidgets.QMainWindow()
        self.uic.btn_confirm.clicked.connect(lambda: self.confirm())
        self.uic.btn_clear.clicked.connect(lambda : self.clear())
        self.uic.note_reason.clear()

        MainWindow.show()
        self.thread = {}



    def set_staff_infor(self, staff_code, staff_pass):
        self.staff_code = staff_code
        self.staff_pass = staff_pass

    def get_staff_infor(self):
        return self.staff_code, self.staff_pass

    def check_in_out(self, mode):
        link = ''
        char = ''
        hour = datetime.now().strftime("%H")
        # minute = datetime.datetime.now().minute
        go = 0
        if mode == 1:
            if int(hour) <= 12:
                link = self.link_checkin
                char = 'Check-In'
                go = 1
            else:
                link = self.link_checkout
                char = 'Check-Out'
        if mode == 2:
            link = self.link_checkin
            char = 'Check-In'
            go = 1
        if mode == 3:
            link = self.link_checkout
            char = 'Check-Out'
        return link, char, go

    def set_code_name_gui(self, staff_code, staff_name):
        self.uic.staff_code.setText(staff_code)
        self.uic.staff_name.setText(staff_name)

    def get_status(self):
            return self.status

    def set_status(self, value):
        self.status=value

    def clear(self):
        self.uic.staff_code.setText('')
        self.uic.staff_name.setText('')
        self.uic.note_reason.clear()
        self.uic.label_image.clear()
        self.set_status(0)

    def confirm(self):
        try:
            if self.uic.staff_code.text() != ' ':
                staff_code_check, staff_pass_check = self.get_staff_infor()
                staff_note = self.uic.note_reason.toPlainText()
                payload = "{\"checkLogin\": {\"username\":\""+staff_code_check.strip().upper()+"\",\"password\":\""+staff_pass_check.strip()+"\"},\"MaNhanVien\": \""+staff_code_check.strip().upper()+"\",\"LyDo\": \""+staff_note+"\"}\r\n\r\n"
                headers = {
                    'content-type': "application/json",
                    'cache-control': "no-cache",
                    'postman-token': "aafcd5d9-5997-c471-8f48-747dde4494e6"
                }
                # get image from server
                response = requests.request("POST", self.url_checking, data=payload, headers=headers)
                json_data = json.loads(response.text)
                if json_data['Success'] == True:
                    self.uic.staff_code.setText('')
                    self.uic.staff_name.setText('')
                    self.uic.note_reason.clear()
                    self.uic.label_image.clear()
                    QMessageBox.about(self, 'Thông báo', "Chấm công hoàn tất, Mời nhân viên tiếp theo")
                else:
                    QMessageBox.warning(self, "Thông Báo", "Lỗi! vui lòng liên hệ quản trị viên")
                    self.change_mode("2. Đi Vào")
                self.set_status(0)
        except:
            QMessageBox.warning(self, "Thông Báo", "Lỗi! vui lòng liên hệ quản trị viên")
            pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.confirm()
        event.accept()

    def change_mode(self, str_mode):
        #self.set_label_stamp(int(str_mode[0:1]))
        self.uic.in_out.setText(str(str_mode))

    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        self.thread[1].stop()

    def start_capture_video(self):
        self.thread[1] = capture_video(index=1, data_path=self.data_path, url_status=self.url_status)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)

    def show_wedcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.show_video.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self, index, data_path,url_status):
        self.index = index
        print("start threading", self.index)
        super(capture_video, self).__init__()
        self.data_path = data_path
        self.url_status = url_status

    def loaddata(self):
        load_data = torch.load(self.data_path)
        embedding_list = load_data[0]
        name_list = load_data[1]
        return embedding_list, name_list

    def ca_disk(self, img_cropped, embedding_list):
        unsqueeze_crop = torch.unsqueeze(img_cropped, dim=0)#1*3*160*160
        emb = resnet.run(None, {"actual_input_1": np.float32(unsqueeze_crop)})
        emb_torch = torch.from_numpy(emb[0]).to(device)
        dist_list = []
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb_torch, emb_db).item()
            dist_list.append(dist)
        min_dist = min(dist_list)
        min_dist_idx = dist_list.index(min_dist)
        return min_dist_idx, min_dist





    def stringToRGB(self, base64_string):
        imgdata = base64.b64decode(str(base64_string))
        image = Image.open(io.BytesIO(imgdata))
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    def get_face(self, img, box):
        img_crop = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2],:] #160*160*3
        img_crop = cv2.resize(img_crop, (160, 160)) #160*160*3
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR) #160*160*3
        #cv2.imwrite("pic2.jpg", img_crop)
        return img_crop

    def imresample(img, sz):
        im_data = interpolate(img, size=sz, mode="area")
        return im_data

    def crop_resize(self, img, box, image_size):
        if isinstance(img, np.ndarray):
            img = img[box[1]:box[3], box[0]:box[2]]
            out = cv2.resize(
                img,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA
            ).copy()
        elif isinstance(img, torch.Tensor):
            img = img[box[1]:box[3], box[0]:box[2]]
            out = self.imresample(
                img.permute(2, 0, 1).unsqueeze(0).float(),
                (image_size, image_size)
            ).byte().squeeze(0).permute(1, 2, 0)
        else:
            out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
        return out

    def run(self):
        cap = cv2.VideoCapture(0)
        embedding_list, name_list = self.loaddata()
        for name in name_list:
            print("name", name)
            print("++++++++++++++")
        while True:
            ret, cv_img = cap.read()
            if ret:
                try:
                    start_time = time.time()
                    img = Image.fromarray(cv_img)
                    #img_cropped_list, prob_list = mtcnn(img, return_prob=True)
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    results = face_cascade.detectMultiScale(gray, 1.3, 5)
                    #results = detector.detect_faces(cv_img)
                    #print(results)
                    for bounding_box in results:
                        #bounding_box, prob = result['box'], result['confidence']
                       #if prob > 0.9:
                        cv_img = cv2.rectangle(cv_img, (bounding_box[0], bounding_box[1]),(bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),(0, 155, 255), 2)
                        face = torch.from_numpy(self.get_face(cv_img, bounding_box)).permute(2, 1, 0)
                        min_dist_idx, min_dist = self.ca_disk(face, embedding_list)
                        if min_dist < 0.7:
                            staff_infor = name_list[min_dist_idx]

                            staff_code, staff_pass = staff_infor.split('_')
                            main_win.set_staff_infor(staff_code, staff_pass)
                            status = main_win.get_status()
                            if status == 0:
                                payload = "{\r\n\"checkLogin\": {\"username\":\" " + staff_code.strip().upper() + "\",\"password\":\"" + staff_pass + "\"},\r\n\"MaNhanVien\": \"" + staff_code.strip().upper() + "\"\r\n}\r\n"
                                headers = {
                                    'content-type': "application/json",
                                    'cache-control': "no-cache",
                                    'postman-token': "aafcd5d9-5997-c471-8f48-747dde4494e6"
                                }
                                # get status
                                response_status = requests.request("POST", self.url_status, data=payload,headers=headers)
                                json_data_status = json.loads(response_status.text)
                                #json_data_loaded = json_data_infor['Th']
                                if json_data_status["Success"] == True:
                                    main_win.set_code_name_gui(json_data_status['ThongTinNhanVien']['MaNhanVien'],
                                                               json_data_status['ThongTinNhanVien']['TenNhanVien'])
                                    main_win.change_mode(json_data_status['TrangThaiChamCong'])
                                    # convert image from base64 to RGB
                                    img = self.stringToRGB(json_data_status['HinhAnhNhanVien'])
                                    # resize image from (400,300) to (161,211) and show
                                    img = main_win.convert_cv_qt(img)
                                    main_win.uic.label_image.setPixmap(img.scaled(161, 241))
                                    main_win.set_status(1)
                                else:
                                    QMessageBox.about(self, 'Thông báo', "Lỗi!!! Vui lòng liên hệ bộ phận kỹ thuật")
                            if main_win.uic.staff_code.text() != ' ':
                                cv_img = cv2.putText(cv_img, 'Press Enter to complete.....', (20, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                except:
                    #QMessageBox.about(self, 'Thông báo', "Chấm công hoàn tất, mới nhân viên tiếp theo")
                    pass

                self.signal.emit(cv_img)

    def stop(self):
        print("stop threading", self.index)
        self.terminate()

if __name__ == "__main__":

### define status and varible
    staff_code = staff_name = ''

### define host
    url_status = 'http://113.163.69.8:8100/API/NS_NhanSu.svc/ThongTinChamCongNhanVien'
    url_checking = 'http://113.163.69.8:8100/API/NS_NhanSu.svc/ChamCongNhanVien'
    data_path = './Data/data.pt'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, min_face_size=40, device=device)
    detector = MTCNN()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    resnet = ort.InferenceSession("facenet.onnx")
    app = QApplication(sys.argv)
    main_win = MainWindow(url_checking,url_status, data_path)
    main_win.start_capture_video()
    main_win.show()
    sys.exit(app.exec())