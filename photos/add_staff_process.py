import glob

import cv2
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QMessageBox
import addstaff
import sys
from facenet_pytorch import InceptionResnetV1
import unidecode
import hashlib
from PIL import  Image

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from datetime import datetime
import os
import shutil
from mtcnn_ort import MTCNN
import onnxruntime as ort
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self, IMG_PATH, Photo_dic, photo_dir, data_path):
        super().__init__()
        self.setWindowTitle("Thêm Nhân Viên - Hệ Thống Chấm Công")
        self.uic = addstaff.Ui_addstaff()
        self.uic.setupUi(self)
        self.setFixedSize(341, 164)

        self.IMG_PATH = IMG_PATH
        self.Photo_dic = Photo_dic
        self.photo_dir = photo_dir
        self.data_path = data_path

        MainWindow = QtWidgets.QMainWindow()
        self.uic.btn_submit.clicked.connect(lambda: self.addstaff())
        MainWindow.show()

    def get_from_form(self):
        return self.uic.staff_code.text(), self.uic.staff_pass.text()

    def load_data(self):
        data = torch.load(self.data_path)
        embedding_list = data[0]
        name_list = data[1]
        return name_list, embedding_list

    def get_face(self, img, box):
        img_crop = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :]  # 160*160*3
        img_crop = cv2.resize(img_crop, (160, 160))  # 160*160*3
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)  # 160*160*3
        # cv2.imwrite("pic2.jpg", img_crop)
        return img_crop

    def encode_image(self, img_cropped):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        unsqueeze_crop = torch.unsqueeze(img_cropped, dim=0)
        print(unsqueeze_crop.shape)
        emb = resnet.run(None, {"actual_input_1": np.float32(unsqueeze_crop)})
        return torch.from_numpy(emb[0]).to(device)


    def inarr(self, string, arr):
        for e in arr:
            if string == e:
                return True
        return False

    def save_data(self, embedding_list, name_list):
        data = [embedding_list, name_list]
        torch.save(data, self.data_path)# saving data.pt file

    def collate_fn(self, x):
        return x[0]

    def remove_accent(self, text):
        return unidecode.unidecode(text)

    def save_images(self, path, image, USR_PATH):
        if os.path.isdir(USR_PATH) == False:
            os.makedirs(USR_PATH)
        cv2.imwrite(path, image)
        cv2.imshow('output', image)
        cv2.waitKey(0)

    def md5_encode(self, string):
        hash = string.strip().encode("utf-8")
        hash = hashlib.md5((hash))
        return hash.hexdigest()

    def capture_image(self, staff_code, staff_pass):
        count = 10
        leap = 1
        name = staff_code +"_"+ self.md5_encode(staff_pass)
        USR_PATH = os.path.join(self.IMG_PATH, name)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("code run on: ", device)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while cap.isOpened() and count:
            ret, frame = cap.read()  # and count:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = face_cascade.detectMultiScale(gray, 1.3, 5)
            #results = detector.detect_faces(frame)
            for bounding_box in results:
                #bounding_box, pro = result['box'], result['confidence']
                #if bounding_box is not None and pro > 0.9 and count:
                face_img = self.get_face(frame, bounding_box)
                path = str(USR_PATH + '/{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-").replace(" ", "-") + str(count)))
                self.save_images(path, face_img, USR_PATH)
                count -= 1
            cv2.imshow('Thu Du Lieu....', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def load_img(self, path):
        list_img = []
        for filename in glob.glob(path + "/*.jpg"):
            img_load = Image.open(filename)
            list_img.append(img_load)
        return list_img

    def addstaff(self):
       staff_code, staff_pass = self.get_from_form()

       if staff_code.strip() != '' and staff_pass.strip() != '':
            name = staff_code + '_' + self.md5_encode(staff_pass)
            name_list, embedding_list = self.load_data()
            USR_PATH = os.path.join(self.IMG_PATH, name)
            if self.inarr(name, name_list) == False:
                self.capture_image(staff_code, staff_pass)
                list_img = self.load_img(USR_PATH)
                for img_one in list_img:
                    #print(type(torch.from_numpy(np.array(img_one)).permute(2, 1, 0)))
                    add_v = self.encode_image(torch.from_numpy(np.array(img_one)).permute(2, 1, 0))
                    embedding_list.append(add_v)
                    name_list.append(name)
                self.save_data(embedding_list, name_list)
                shutil.move(photo_dir + name, self.Photo_dic)
                QMessageBox.about(self, 'Thông báo',"Thêm thông tin nhân viên thành công! reload lại chương trình để kiểm tra!")
                sys.exit()
            else:
                 QMessageBox.about(self, 'Thông báo',"Nhân Viên này đã được thêm vào hệ thống trước đó. Vui lòng kiểm tra lại:::!")

       else:
           QMessageBox.about(self, 'Thông báo', "Vui Lòng Nhập Mã Nhân Viên và Mật Khẩu")



if __name__ == "__main__":

    IMG_PATH = './stamp/'
    Photo_dic = './photos/'
    photo_dir = './stamp/'
    data_path = './Data/data.pt'

    app = QApplication(sys.argv)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    main_win = MainWindow(IMG_PATH, Photo_dic,photo_dir, data_path)
    detector = MTCNN()
    resnet = ort.InferenceSession("facenet.onnx")
    main_win.show()
    sys.exit(app.exec())