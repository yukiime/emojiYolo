# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import torch

# 创建ROOT路径 此时ROOT路径约等于yolov5文件夹的绝对路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/emoji.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.55,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    # 确定是否要保存img
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 确定文件后缀是否是 IMG_FORMATS + VID_FORMATS
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 确定文件开头是否表示网络流
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 判断是否是数值，即摄像头 / 传入的是否是.txt / 是否是网络流且不是文件       
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    # 屏幕录制
    screenshot = source.lower().startswith('screen')
    # 判断下载
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # 先建立一个保存文件夹 / increment_path:增量目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # save_txt=T 则detect文件夹下新建一个label文件夹
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    # GPU或CPU
    device = select_device(device)
    # 加载模型 weights即传来的weights
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # 下采样stride是model的stride属性
    # names = model.module.names if hasattr(model, 'module') else model.names  
    # names 是 get class names
    stride, names, pt = model.stride, model.names, model.pt
    # 查看imgsz是否是stride的倍数 不是则默认计算倍数
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    # 如果是摄像头
    if webcam:
        # True 表示当前环境支持图像显示
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    # 如果是屏幕
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    # 如果是媒体
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # bs不变，说明每次传入一张图片
    # path 和 writer 的列表长度都为bs
    # vid_path是一个列表 用于存储视频文件的路径
    # vid_writer是一个列表 用于存储视频写入器（VideoWriter）对象
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # warmup:热身 随便建一张图片给GPU跑一下
    # 在模型上执行一次推断以预热模型 
    # 确保模型已加载到GPU并进行了一次前向计算
    # 这有助于提高后续推断的速度
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # 循环的时候会先去执行dataset的__iter__方法 接着执行__next__方法
    # path, im, im0s, vid_cap, s 分别赋值了__next__方法的返回值
    for path, im, im0s, vid_cap, s in dataset:

        # 计时器上下文管理器 
        # 用于测量代码块的执行时间
        # dt是用来存储时间的 看上文dt = (Profile(), Profile(), Profile())
        with dt[0]:
            # 将NumPy数组 im 转换为PyTorch张量，并将其移动到模型所在的设备
            im = torch.from_numpy(im).to(model.device)
            # 判断有没有用到半精度(float32 转换为 float16) 没有仍是float32
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 归一化
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # im有(h,w,c) 高宽通道数 三个维度
            # 少了batch这一维度
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # 这里开始推断
        # Inference
        with dt[1]:
            # visualize是run函数传进来的 默认是FALSE 
            # True的话会保存特征图
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # augment是run函数传进来的 
            # True则数据增强(如翻转、缩放等) 但是会降低运行速度 
            # pred是检测出的框 
            # 关于 pred 维度的解释 
            # pred.shape=(batch_size/1个batch的所有图数, num_boxes/一个图多少个检测框, 5+num_class)
	        # h,w为传入网络图片的长和宽,注意dataset在检测时使用了矩形推理,所以这里h不一定等于w
            # pred[..., 0:4]为预测框坐标=预测框坐标为xywh(中心点+宽长)格式
	        # pred[..., 4]为objectness置信度
	        # pred[..., 5:-1]为分类结果
            # 推测这里是调用了forward方法
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            # conf_thres是传进来的置信度阈值 
            # iou同理 
            # 根据上述两个值过滤
            # max_det是最大检测出的目标数 超过自动舍弃/过滤
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # 返回值查看函数细节 Returns:
            # (batch_size/1个batch的所有图数,一个图多少个检测框,5+1)
            # list of detections, on (n,6) tensor per image [xyxy, conf, cls] 
            # n个检测框
            # xyxy 左上角x值 左上角y值 右下角x值 右下角y值 
            # cls是过滤出的一个class

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # 遍历1个batch中的per image
        # det就是per image的所有框
        for i, det in enumerate(pred):  # per image
            # 见上文 用于计数
            seen += 1
            if webcam:  # batch_size >= 1
                # dataset.count 基本也就表示哪一帧了
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                # frame 根据dataset中有无 无赋值0
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            
            # im.shape[2:] 图的像素大小
            s += '%gx%g ' % im.shape[2:]  # print string

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # 判断是否要裁剪
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            # 画图 见函数详解
            # line_thickness是run传进来的默认是3 
            # names 见上文 names = model.names 是 class names
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # 有几个框
            if len(det):
                # 坐标映射
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 在终端打印信息
                # Print results
                # 遍历所有框
                for c in det[:, 5].unique():
                    # 求和所有同类的框
                    n = (det[:, 5] == c).sum()  # detections per class
                    # 比如5个框: 4个person 1个bus
                    # s的追加 4 person, 1 bus
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # 保存到txt
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    # 保存原图 或者 切割 
                    # 或者 view_img表示摄像头输出
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # 是否隐藏标签和置信度
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # 在原图上画框
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    # 是否把截下的目标框保存
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            # 返回这个画好框的图
            im0 = annotator.result()

            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    # 保存图到路径下
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
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
        
        # 本来是为了保存 现在是按两次q
        key = cv2.waitKey(10)
        if key == 27:
            return im0
        
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        
        # return im0;

    # Print results
    # seen是计数器 多少张图
    # dt 耗时
    # 求出平均时间
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # 终端打印
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    # 额外输出保存路径信息
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    
    # 更新模型 
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

#  parser.add_argument('--source', type=str, default="screen 0 200 280 1700 950", help='file/dir/URL/glob/screen/0(webcam)')
#  parser.add_argument('--source', type=str, default=ROOT / 'data/test', help='file/dir/URL/glob/screen/0(webcam)')
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weight/emoji_best_S.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/test/1001.mp4', help='file/dir/URL/glob/screen/0(webcam)')
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
    opt = parser.parse_args()
    # 如果图片大小不够，扩大
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 在终端打印参数
    print_args(vars(opt))
    return opt

def main(opt):
    # 检查requirements.txt要求的环境
    check_requirements(exclude=('tensorboard', 'thop'))
    # **vars(opt)将字典解压为关键字参数。它将字典中的每个键值对作为单独的关键字参数传递给函数。
    run(**vars(opt))
    # run(weights=ROOT / 'weight/emoji_best_S.pt', source=0)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
