import argparse
import platform
from pathlib import Path
import torch
from torch import tensor
from PIL import Image
from PIL.ImageQt import ImageQt

file_filter = "media (*.jpg *.bmp *.dng *.jpeg *.jpg *.mpo *.png *.tif *.tiff *.webp *.pfm *.asf *.avi *.gif *.m4v *.mkv *.mov *.mp4 *.mpeg *.mpg *.ts *.wmv)"
file_img_filter = "image (*.jpg *.bmp *.dng *.jpeg *.jpg *.mpo *.png *.tif *.tiff *.webp)"
file_video_filter = "video (*.gif *.m4v *.mkv *.mov *.mp4 *.mpeg *.mpg *.ts *.wmv)"

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from detect import ROOT
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (Profile, check_img_size, check_imshow, cv2, increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from ui.emoji_ui_0524 import Ui_MainWindow


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # icon
        self.setWindowIcon(QIcon("ui/icon.jpg"))
        # 使用ui文件导入定义界面类
        self.ui = Ui_MainWindow()

        # 权重初始文件名
        self.weight_model = None
        # QTimer 初始化
        self.timer_video = QTimer()

        # 初始化
        self.ui.setupUi(self)
        self.init_slots()
        

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1456, 646)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.media = QtWidgets.QLabel(self.centralwidget)
        self.media.setGeometry(QtCore.QRect(10, 10, 821, 551))
        self.media.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.media.setObjectName("media")
        self.emoji = QtWidgets.QLabel(self.centralwidget)
        self.emoji.setGeometry(QtCore.QRect(840, 10, 601, 551))
        self.emoji.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.emoji.setObjectName("emoji")
        self.increase_conf = QtWidgets.QPushButton(self.centralwidget)
        self.increase_conf.setGeometry(QtCore.QRect(1160, 570, 141, 31))
        self.increase_conf.setObjectName("increase_conf")
        self.decrease_conf = QtWidgets.QPushButton(self.centralwidget)
        self.decrease_conf.setGeometry(QtCore.QRect(1160, 600, 141, 31))
        self.decrease_conf.setObjectName("decrease_conf")
        self.increase_iou = QtWidgets.QPushButton(self.centralwidget)
        self.increase_iou.setGeometry(QtCore.QRect(1300, 570, 141, 31))
        self.increase_iou.setObjectName("increase_iou")
        self.decrease_iou = QtWidgets.QPushButton(self.centralwidget)
        self.decrease_iou.setGeometry(QtCore.QRect(1300, 600, 141, 31))
        self.decrease_iou.setObjectName("decrease_iou")
        self.get_weight = QtWidgets.QPushButton(self.centralwidget)
        self.get_weight.setGeometry(QtCore.QRect(20, 570, 171, 61))
        self.get_weight.setObjectName("get_weight")
        self.video = QtWidgets.QPushButton(self.centralwidget)
        self.video.setGeometry(QtCore.QRect(600, 570, 181, 61))
        self.video.setObjectName("video")
        self.init_model = QtWidgets.QPushButton(self.centralwidget)
        self.init_model.setGeometry(QtCore.QRect(200, 570, 201, 61))
        self.init_model.setObjectName("init_model")
        self.picture = QtWidgets.QPushButton(self.centralwidget)
        self.picture.setGeometry(QtCore.QRect(410, 570, 181, 61))
        self.picture.setObjectName("picture")
        self.camera = QtWidgets.QPushButton(self.centralwidget)
        self.camera.setGeometry(QtCore.QRect(790, 570, 201, 61))
        self.camera.setObjectName("camera")
        self.see_history = QtWidgets.QPushButton(self.centralwidget)
        self.see_history.setGeometry(QtCore.QRect(1010, 570, 141, 61))
        self.see_history.setObjectName("see_history")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "emojiAI"))
        self.media.setText(_translate("MainWindow", "media"))
        self.emoji.setText(_translate("MainWindow", "emoji"))
        self.increase_conf.setText(_translate("MainWindow", "increase confidence"))
        self.decrease_conf.setText(_translate("MainWindow", "decrease confidence"))
        self.increase_iou.setText(_translate("MainWindow", "increase IoU"))
        self.decrease_iou.setText(_translate("MainWindow", "decrease IoU"))
        self.get_weight.setText(_translate("MainWindow", "Selected weight"))
        self.video.setText(_translate("MainWindow", "Detect video"))
        self.init_model.setText(_translate("MainWindow", "weight Model initialization"))
        self.picture.setText(_translate("MainWindow", "Detect pictures"))
        self.camera.setText(_translate("MainWindow", "camera detection"))
        self.see_history.setText(_translate("MainWindow", "history"))
        
    # 绑定信号与槽
    def init_slots(self):
        self.ui.get_weight.clicked.connect(self.select_model)
        self.ui.init_model.clicked.connect(self.model_init_opt)
        self.ui.picture.clicked.connect(self.detect_picture)
        self.ui.video.clicked.connect(self.detect_video)
        self.ui.camera.clicked.connect(self.open_camera)
        self.ui.see_history.clicked.connect(self.open_history_dir)
        self.timer_video.timeout.connect(self.perFrame)
        self.ui.increase_conf.clicked.connect(self.increase_conf_thres)
        self.ui.decrease_conf.clicked.connect(self.decrease_conf_thres)
        self.ui.increase_iou.clicked.connect(self.increase_iou_thres)
        self.ui.decrease_iou.clicked.connect(self.decrease_iou_thres)

    # 打开权重文件
    def select_model(self):
        # 打开文件对话框，让用户选择模型文件
        self.weight_model, _ = QFileDialog.getOpenFileName(self.ui.get_weight, 'Choose weight model', 'weight/', "*.pt;")

        if not self.weight_model:
            # 如果用户取消选择文件或打开失败，则显示警告对话框
            QtWidgets.QMessageBox.warning(self, u"Warning", u"file open failed", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            # 显示所选模型文件的地址
            self.ui.media.setText('The address of the selected model file :' + str(self.weight_model))

    def model_init_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weight/emoji_best_L.pt', help='model path or triton URL')
        parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default="", help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.55, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results',default=True)
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
        
        # opt = parser.parse_args()
        self.opt = parser.parse_args()

        # 如果图片大小不够，扩大
        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand

        # 在终端打印参数
        print_args(vars(self.opt))
        
        # 有选择weight就用选的
        if self.weight_model:
            weights = self.weight_model
        else:
            weights = self.opt.weights

        # GPU或CPU 存储在self
        self.device = select_device(self.opt.device)
        # 加载模型
        self.model = DetectMultiBackend(weights, device=self.device, dnn=self.opt.dnn, data=self.opt.data, fp16=self.opt.half)
        
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.conf_thres = self.opt.conf_thres
        self.iou_thres = self.opt.iou_thres
        
        s = "Now conf_thres:" + str(self.conf_thres) + "  iou_thres:" + str(self.iou_thres)

        # 打印信息
        self.ui.emoji.setText(s)

        # 查看imgsz是否是stride的倍数 不是则默认计算倍数
        self.imgsz = check_img_size(self.opt.imgsz, s=self.stride)  # check image size
        
        print("model initial done")
        print(self.names)
        QtWidgets.QMessageBox.information(self, u"ok", u"Model initialization succeeded")
        
    def emoji_adaptation(self, emoji_ints):
        emoji_arr = ['😊','😭','😛','🤬','😐','✌️','👉','🖐️']
        emoji_str = ''
        for emoji_no in emoji_ints:
            emoji_str = emoji_str + emoji_arr[emoji_no]
        return emoji_str
    

    def detect_picture(self):
        file_path, _  = QFileDialog.getOpenFileName(self.ui.picture, "choose image", "data/test/", file_img_filter)
        
        if not file_path:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"Failed to open image", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            # 在标签中显示选择的图片路径
            self.ui.emoji.setText(f"picture path:{file_path}")
            save_img = not self.opt.nosave  # save inference images
            save_dir = increment_path(Path(self.opt.project) / self.opt.name, exist_ok = self.opt.exist_ok)  # increment run
            (save_dir / 'labels' if self.opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Dataloader
            bs = 1  # batch_size
            dataset = LoadImages(file_path, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.opt.vid_stride)
            # vid_path, vid_writer = [None] * bs, [None] * bs

            # Run inference
            self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            # 其实这里就一张图
            for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                    im = torch.from_numpy(im).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.opt.visualize else False
                    pred = self.model(im, augment=self.opt.augment, visualize=visualize)
                with dt[2]:
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.opt.classes,  self.opt.agnostic_nms, max_det = self.opt.max_det)

                print(pred)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    # 其实这两句已经没什么用了
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    # 判断是否要裁剪
                    imc = im0.copy() if self.opt.save_crop else im0  # for save_crop
                    
                    annotator = Annotator(im0, line_width = self.opt.line_thickness, example=str(self.names))
                    emoji_labels = []  # 创建空的一维字符串数组
                    emoji_ints = [] # 创建空的一维整型数组

                    if len(det):
                        # 坐标映射
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        for c in det[:, 5].unique():
                            # 求和所有同类的框
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            # 拿到标签
                            emoji_labels.append(self.names[int(c)])
                            emoji_ints.append(int(c))
                            
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            # 保存到txt
                            if self.opt.save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            
                            # 保存原图 或者 切割 
                            if save_img or self.opt.save_crop:  # Add bbox to image
                                c = int(cls)  # integer class
                                # 是否隐藏标签和置信度
                                label = None if self.opt.hide_labels else (self.names[c] if self.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                                # 在原图上画框
                                annotator.box_label(xyxy, label, color=colors(c, True))

                            # 是否把截下的目标框保存
                            if self.opt.save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)
                    
                    im0 = annotator.result()

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            # 保存图到路径下
                            cv2.imwrite(save_path, im0)
                        else:
                            print("dataset.mode != 'image'")
                    

                    self.ui.media.setScaledContents(False)  # 不适应窗口
                    
                    # 格式应该是对的 但是闪退
                    # temp = Image.fromarray(im0)
                    # qimage = ImageQt(temp)
                    # pixmap = QtGui.QPixmap.fromImage(qimage)
                    # self.ui.media.setPixmap(pixmap)

                    # temp = cv2.cvtColor(im0, cv2.COLOR_BGR2BGRA)
                    # temp = cv2.resize(temp, (640, 480), interpolation=cv2.INTER_AREA)
                    # QtImg = QtGui.QImage(temp.data, temp.shape[1], temp.shape[0], QtGui.QImage.Format_RGB32)
                    # self.ui.media.setPixmap(QtGui.QPixmap.fromImage(QtImg))

                    # temp = Image.fromarray(im0)
                    # temp.save('JJ.jpg')
                    # qimage = ImageQt(temp)
                    # pixmap = QtGui.QPixmap.fromImage(qimage)
                    # self.ui.media.setPixmap(qimage)

                    # IO读取显得不太行
                    showimg = QPixmap(save_path)
                    showimg = showimg.scaled(self.ui.media.size(), Qt.KeepAspectRatio)
                    self.ui.media.setPixmap(showimg)

                    self.ui.emoji.setFont(QFont("Arial", 68, QFont.Bold))  # 设置字体和大小
                    self.ui.emoji.setWordWrap(True)  # 设置多行文本
                    self.ui.emoji.setAlignment(Qt.AlignCenter)  # 设置居中对齐
                    # 展示emoji
                    if emoji_ints:
                        temp = self.emoji_adaptation(emoji_ints)
                        # temp = "Main emoji: " + str(emoji_ints[0]) + '😀'
                        self.ui.emoji.setText(temp)
                    else:
                        temp = "No emoji" + '🐱‍🐉'
                        self.ui.emoji.setText(temp)

        if self.opt.update:
            # 有选择weight就用选的
            if self.weight_model:
                strip_optimizer(self.weight_model)  # update model (to fix SourceChangeWarning)
            else:
                strip_optimizer(self.opt.weights)  # update model (to fix SourceChangeWarning)
                
    def detect_video(self):
        file_path, _  = QFileDialog.getOpenFileName(self.ui.video, "choose video", "data/test/", file_video_filter)
        
        if not file_path:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"Failed to open video", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            # 在标签中显示选择的图片路径
            self.ui.emoji.setText(f"video path:{file_path}")
            save_img = not self.opt.nosave  # save inference images
            save_dir = increment_path(Path(self.opt.project) / self.opt.name, exist_ok = self.opt.exist_ok)  # increment run
            (save_dir / 'labels' if self.opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Dataloader
            bs = 1  # batch_size
            dataset = LoadImages(file_path, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.opt.vid_stride)
            vid_path, vid_writer = [None] * bs, [None] * bs

            # Run inference
            self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            # 其实这里就一张图
            for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                    im = torch.from_numpy(im).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.opt.visualize else False
                    pred = self.model(im, augment=self.opt.augment, visualize=visualize)
                with dt[2]:
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.opt.classes,  self.opt.agnostic_nms, max_det = self.opt.max_det)

                print(pred)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    # 其实这两句已经没什么用了
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    # 判断是否要裁剪
                    imc = im0.copy() if self.opt.save_crop else im0  # for save_crop
                    
                    annotator = Annotator(im0, line_width = self.opt.line_thickness, example=str(self.names))
                    emoji_labels = []  # 创建空的一维字符串数组
                    emoji_ints = []

                    if len(det):
                        # 坐标映射
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        for c in det[:, 5].unique():
                            # 求和所有同类的框
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            # 拿到标签
                            emoji_labels.append(self.names[int(c)])
                            emoji_ints.append(int(c))
                            
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            # 保存到txt
                            if self.opt.save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            
                            # 保存原图 或者 切割 
                            if save_img or self.opt.save_crop:  # Add bbox to image
                                c = int(cls)  # integer class
                                # 是否隐藏标签和置信度
                                label = None if self.opt.hide_labels else (self.names[c] if self.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                                # 在原图上画框
                                annotator.box_label(xyxy, label, color=colors(c, True))

                            # 是否把截下的目标框保存
                            if self.opt.save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)
                    
                    # showimg = im0

                    im0 = annotator.result()
                    
                    # result = cv2.cvtColor(showimg, cv2.COLOR_BGR2RGB)
                    # showImage = QtGui.QImage(result.data, result.shape[1], result.shape[0], QtGui.QImage.Format_RGB888)
                    # self.ui.emoji.setPixmap(QtGui.QPixmap.fromImage(showImage))

                    # 源码
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                    # self.cap = cv2.VideoCapture(str(p))
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                    # self.timer_video.start(100)
                    # self.timer_video.timeout.connect(self.perFrame)

                    # Save results (image with detections)
                    if save_img:
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
                    
                    # 展示emoji
                    self.ui.emoji.setFont(QFont("Arial", 68, QFont.Bold))  # 设置字体和大小
                    self.ui.emoji.setWordWrap(True)  # 设置多行文本
                    self.ui.emoji.setAlignment(Qt.AlignCenter)  # 设置居中对齐
                    if emoji_ints:
                        temp = self.emoji_adaptation(emoji_ints)
                        self.ui.emoji.setText(temp)
                    else:
                        temp = "No emoji" + '🐱‍🐉'
                        self.ui.emoji.setText(temp)

        if self.opt.update:
            # 有选择weight就用选的
            if self.weight_model:
                strip_optimizer(self.weight_model)  # update model (to fix SourceChangeWarning)
            else:
                strip_optimizer(self.opt.weights)  # update model (to fix SourceChangeWarning)        
                        
    def open_camera(self):
        save_img = not self.opt.nosave  # save inference images
        save_dir = increment_path(Path(self.opt.project) / self.opt.name, exist_ok = self.opt.exist_ok)  # increment run
        (save_dir / 'labels' if self.opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        self.ui.media.setText("Double-click Q on the keyboard to exit shexiangtou")

        # Dataloader
        bs = 1  # batch_size
        self.view_img = self.opt.view_img
        self.view_img = check_imshow(warn=True)
        dataset = LoadStreams("0", img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.opt.vid_stride)
        bs = len(dataset)
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.opt.visualize else False
                pred = self.model(im, augment=self.opt.augment, visualize=visualize)
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.opt.classes,  self.opt.agnostic_nms, max_det = self.opt.max_det)

            print(pred)

            for i, det in enumerate(pred):  # per image
                # 见上文 用于计数
                seen += 1
                # dataset.count 基本也就表示哪一帧了
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # 判断是否要裁剪
                imc = im0.copy() if self.opt.save_crop else im0  # for save_crop
                
                annotator = Annotator(im0, line_width = self.opt.line_thickness, example=str(self.names))
                emoji_labels = []  # 创建空的一维字符串数组
                emoji_ints = []

                if len(det):
                    # 坐标映射
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, 5].unique():
                        # 求和所有同类的框
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        # 拿到标签
                        emoji_labels.append(self.names[int(c)])
                        emoji_ints.append(int(c))
                    for *xyxy, conf, cls in reversed(det):
                        # 保存到txt
                        if self.opt.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            # 保存原图 或者 切割 
                        if save_img or self.opt.save_crop:  # Add bbox to image
                            c = int(cls)  # integer class
                            # 是否隐藏标签和置信度
                            label = None if self.opt.hide_labels else (self.names[c] if self.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                            # 在原图上画框
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        # 是否把截下的目标框保存
                        if self.opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)               
                
                im0 = annotator.result()

                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

                if save_img:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

                # 展示emoji
                self.ui.emoji.setFont(QFont("Arial", 68, QFont.Bold))  # 设置字体和大小
                self.ui.emoji.setWordWrap(True)  # 设置多行文本
                self.ui.emoji.setAlignment(Qt.AlignCenter)  # 设置居中对齐
                if emoji_ints:
                    temp = self.emoji_adaptation(emoji_ints)
                    self.ui.emoji.setText(temp)
                else:
                    temp = "No emoji" + '🐱‍🐉'
                    self.ui.emoji.setText(temp)
                
        if self.opt.update:
            # 有选择weight就用选的
            if self.weight_model:
                strip_optimizer(self.weight_model)  # update model (to fix SourceChangeWarning)
            else:
                strip_optimizer(self.opt.weights)  # update model (to fix SourceChangeWarning)

    def open_history_dir(self):

        # 打开文件对话框并获取选择的路径或文件夹
        file_path, _  = QFileDialog.getOpenFileName(self.ui.see_history, "history", "runs/detect/", file_filter)
        self.ui.emoji.setFont(QFont("Arial", 36, QFont.Bold))  # 设置字体和大小
        self.ui.emoji.setWordWrap(True)  # 设置多行文本
        self.ui.emoji.setAlignment(Qt.AlignLeft)  # 设置居中对齐
        # 在标签中显示选择的图片路径
        self.ui.emoji.setText(f"history path:{file_path}")
        
        # 让该文件可以在self.ui.media上展现出来
        if file_path:
            if file_path.lower().endswith(IMG_FORMATS):
                # 加载图片
                media = QPixmap(file_path)
                media = media.scaled(self.ui.media.size(), Qt.KeepAspectRatio)
                self.ui.media.setPixmap(media)
            elif file_path.lower().endswith(VID_FORMATS):
                self.cap = cv2.VideoCapture(file_path)
                self.timer_video.start(100)
                self.timer_video.timeout.connect(self.perFrame)
                self.ui.emoji.setText("Replay now")
            else:
                self.ui.media.clear()
                self.ui.emoji.setText("Unsupported media format")
        else:
            self.ui.media.clear()
            self.ui.emoji.setText("No media selected")

    def perFrame(self):
        ret, image = self.cap.read()
        if ret:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            elif len(image.shape) == 1:
                vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Indexed8)
            else:
                vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)

            self.ui.media.setPixmap(QPixmap(vedio_img))
            self.ui.media.setScaledContents(True)  # 自适应窗口
        else:
            self.cap.release()
            self.timer_video.stop()

    def increase_conf_thres(self):
        temp = self.conf_thres + tensor(0.05) 
        if(temp < 1.0):
            self.conf_thres = temp
            s = "Now conf_thres:" + str(self.conf_thres) + "  iou_thres:" + str(self.iou_thres)
        else:
            s = "warn: don't let confidence higher than 1.0"

        # 打印信息
        self.ui.emoji.setText(s)
    
    def decrease_conf_thres(self):
        temp = self.conf_thres - tensor(0.05) 
        if(temp > 0):
            self.conf_thres = temp
            s = "Now conf_thres:" + str(self.conf_thres) + "  iou_thres:" + str(self.iou_thres)
        else:
            s = "warn: don't let confidence lower than 0"
            
        # 打印信息
        self.ui.emoji.setText(s)

    def increase_iou_thres(self):
        temp = float(self.iou_thres) + tensor(0.05) 
        if(temp < 1.0):
            self.iou_thres = temp
            s = "Now conf_thres:" + str(self.conf_thres) + "  iou_thres:" + str(self.iou_thres)
        else:
            s = "warn: don't let iou higher than 1.0"

        # 打印信息
        self.ui.emoji.setText(s)
    
    def decrease_iou_thres(self):
        temp = float(self.iou_thres) - tensor(0.05) 
        if(temp > 0):
            self.iou_thres = temp
            s = "Now conf_thres:" + str(self.conf_thres) + "  iou_thres:" + str(self.iou_thres)
        else:
            s = "warn: don't let iou lower than 0"
            
        # 打印信息
        self.ui.emoji.setText(s)

if __name__ == '__main__':
    app = QApplication([])
    mainw = MainWindow()
    mainw.show()
    app.exec_()
