import sys
import time

import cv2
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox
from facenet_pytorch import MTCNN, InceptionResnetV1

import timkeeping
from subprocess import call
import torch
from PIL import Image
import numpy as np
import requests
import datetime
from datetime import datetime
import urllib.request
import onnxruntime as ort
from torch.nn import Parameter
from scipy.spatial.distance import cdist


class MainWindow(QMainWindow):
    def __init__(self, host, link_checkin, link_checkout, data_path):
        super().__init__()
        hour = datetime.now().strftime("%H")
        s = "1. Công Vào" if int(hour) <= 12 else "1. Công Ra"
        self.setWindowTitle("Hệ Thống Chấm Công")
        self.label = self.findChild(QLabel, "in_out")

        self.uic = timkeeping.Ui_TimekeepingSystem()
        self.uic.setupUi(self)
        self.setFixedSize(1042, 653)
        self.label_stamp = 1
        self.status = 0
        self.mode = 1

        self.host = host
        self.link_checkin = host + link_checkin
        self.link_checkout = host + link_checkout
        self.data_path = data_path

        MainWindow = QtWidgets.QMainWindow()
        self.uic.btn_checkin.clicked.connect(lambda: self.change_mode("2. Công Vào"))
        self.uic.btn_checkin_2.clicked.connect(lambda: self.change_mode("3. Công Ra"))
        self.uic.btn_confirm.clicked.connect(lambda: self.confirm())
        self.uic.btn_clear.clicked.connect(lambda : self.clear())
        self.uic.in_out.setText(s)
        self.uic.note_reason.clear()

        MainWindow.show()
        self.thread = {}

    def get_label_stamp(self):
        return self.label_stamp

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

    def set_label_stamp(self, label_stamp):
        self.label_stamp = label_stamp

    def set_code_name(self, staff_code, staff_name):
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
            mode = main_win.get_label_stamp()
            link, char, go = self.check_in_out(mode)
            if self.uic.staff_code.text() != ' ':
                staff_code = self.uic.staff_code.text()
                staff_note = self.uic.note_reason.toPlainText()
                data = {'us_name': 'thomas', 'us_pass': '123', 'id_staff': staff_code, 'note': staff_note}
                print("data: ", data)
                print('link', link)
                print('mode: ', mode)
                check_result = requests.post(link, data=data)
                json_results = check_result.json()
                print(json_results)
                if json_results['code'] == str('200') or json_results['code'] == str('205'):
                    self.uic.staff_code.setText('')
                    self.uic.staff_name.setText('')
                    self.uic.note_reason.clear()
                    self.uic.label_image.clear()
                    QMessageBox.about(self, 'Thông báo', "Chấm công hoàn tất, mới nhân viên tiếp theo")
                else:
                    QMessageBox.warning(self, "Thông Báo", "Lỗi! vui lòng liên hệ quản trị viên")
                    self.change_mode("2. Công Vào")
            self.set_status(0)
        except:
            QMessageBox.warning(self, "Thông Báo", "Lỗi! vui lòng liên hệ quản trị viên")
            pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.confirm()
        event.accept()

    def change_mode(self, str_mode):
        self.set_label_stamp(int(str_mode[0:1]))
        self.uic.in_out.setText(str(str_mode))

    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        self.thread[1].stop()

    def start_capture_video(self):
        self.thread[1] = capture_video(index=1, data_path=self.data_path)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)

    def show_wedcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.show_video.setPixmap(qt_img)
        #self.uic.in_out.setPixmap(qt_img)

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
    def __init__(self, index, data_path):
        self.index = index
        print("start threading", self.index)
        super(capture_video, self).__init__()
        self.data_path = data_path

    def loaddata(self):
        load_data = torch.load(self.data_path)
        embedding_list = load_data[0]
        name_list = load_data[1]
        return embedding_list, name_list

    def ca_disk(self, img_cropped, embedding_list):
        unsqueeze_crop = torch.unsqueeze(img_cropped, dim=0)
        emb = resnet.run(None, {"actual_input_1": unsqueeze_crop.cpu().detach().numpy()})
        emb_torch = torch.from_numpy(emb[0]).to(device)
        dist_list = []
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb_torch, emb_db).item()
            dist_list.append(dist)
        min_dist = min(dist_list)
        min_dist_idx = dist_list.index(min_dist)
        return min_dist_idx, min_dist


    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = cap.read()
            if ret:
                try:
                    start_time = time.time()
                    img = Image.fromarray(cv_img)
                    img_cropped_list, prob_list = mtcnn(img, return_prob=True)
                    embedding_list, name_list = self.loaddata()
                    if img_cropped_list is not None:
                        boxes, _ = mtcnn.detect(img)
                        faces_found = boxes.shape[0]
                        if faces_found == 0:
                            cv2.putText(cv_img, "Only One Face!!! Please!!!", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                        else:
                            for i, prob in enumerate(prob_list):
                                if prob > 0.90:
                                    min_dist_idx, min_dist = self.ca_disk(img_cropped_list[i], embedding_list)
                                    staff_infor = name_list[min_dist_idx]
                                    box = boxes[i]
                                    if min_dist < 0.7:
                                        staff_code, staff_name = staff_infor.split('_')
                                        cv_img = cv2.rectangle(cv_img,(int(np.float32(box[0])), int(np.float32(box[1]))),(int(np.float32(box[2])), int(np.float32(box[3]))),(255, 0, 0), 2)
                                        main_win.set_code_name(staff_code, staff_name)
                                        status = main_win.get_status()
                                        if status == 0:
                                            data = {'us_name': 'thomas', 'us_pass': '123', 'staff_code': staff_code}
                                            check_result = requests.post(link_getstaff, data=data)
                                            json_results = check_result.json()
                                            req = urllib.request.urlopen(json_results['staff_image'])
                                            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                                            img = cv2.imdecode(arr, -1)  # 'Load it as it is'
                                            img = main_win.convert_cv_qt(img)
                                            main_win.uic.label_image.setPixmap(img)
                                            main_win.set_status(1)
                                        cv_img = cv2.putText(cv_img, 'Press Enter to complete.....', (20, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                end_time = time.time()
                                fps = 1 / (end_time - start_time)
                                fps=str(fps)
                                cv_img = cv2.putText(cv_img, 'FPS: ' + fps[0:2], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2, cv2.LINE_AA)
                except:
                    QMessageBox.about(self, 'Thông báo', "Chấm công hoàn tất, mới nhân viên tiếp theo")
                    pass

                self.signal.emit(cv_img)

    def stop(self):
        print("stop threading", self.index)
        self.terminate()

if __name__ == "__main__":

### define status and varible
    staff_code = staff_name = ''

### define host
    host = 'http://192.168.48.138/api/'
    link_checkin = 'addtimekeeping'
    link_checkout = 'updatetimekeeping'
    link_getstaff = host + 'getstaff'
    data_path = './Data/data.pt'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, min_face_size=40, device=device)
    #resnet = InceptionResnetV1(pretrained='vggface2').to(device)
    resnet = ort.InferenceSession("facenet.onnx")
    #resnet.eval()

    app = QApplication(sys.argv)
    main_win = MainWindow(host, link_checkin, link_checkout, data_path)
    main_win.start_capture_video()
    main_win.show()
    sys.exit(app.exec())