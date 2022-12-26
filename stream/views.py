from django.http import StreamingHttpResponse
from django.shortcuts import render

import torch

from deep_sort.deep_sort import DeepSort
import cv2
from pathlib import Path
import os

import sys
sys.path.insert(0, './yolov5')
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count = 0  # 인원수 카운트
counting_obj = set()  # 한번 카운트 하면 또 안하게
data = set()


def index(request):
    return render(request, 'index.html')


def stream():
    yolo_model = 'yolov5s.pt'
    deep_sort_model = 'osnet_x0_25'
    source = 'videos/in.mp4'
    image_size = [640, 640]
    conf_thres = 0.3
    iou_thres = 0.5
    device = ''
    classes = None
    agnostic_nms = False
    augment = False
    deepsort_config = 'deep_sort/configs/deep_sort.yaml'
    half = False
    visualize = False
    max_det = 1000
    dnn = False

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(deepsort_config)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    image_size = check_img_size(image_size, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    dataset = LoadImages(source, img_size=image_size, stride=stride, auto=pt and not jit)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *image_size).to(device).type_as(next(model.model.parameters())))  # warmup

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1], im0.shape[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]  # 박스 꼭짓점 들 인듯?
                        id = output[4]  # 아마도 객체 아이디
                        cls = output[5]  # class 이다. 욜로 프리트레인모델 기준 0은 person
                        # count
                        count_obj(bboxes, w, h, id, int(cls))
                        c = int(cls)  # integer class
                        if c == 0:
                            label = f'{id} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            #
            # if show_vid:
            global count
            color = (0, 255, 0)
            # draw vertical line
            start_point = (w - 400, 0)
            end_point = (w - 400, h - 300)
            cv2.line(im0, start_point, end_point, color, thickness=2)

            start_point = (w - 800, 0)
            end_point = (w - 800, h - 300)
            cv2.line(im0, start_point, end_point, color, thickness=2)

            start_point = (w - 900, 0)
            end_point = (w - 900, h - 300)
            cv2.line(im0, start_point, end_point, color, thickness=2)

            # 혹시 가로선 필요하면 여기 그어요
            start_point = (w - 800, h - 300)
            end_point = (w - 400, h - 300)
            cv2.line(im0, start_point, end_point, color, thickness=2)

        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n'


# obj counting
# 1. 식별할 클래스(카운팅할 클래스)를 제한하는게 ㅇㅅㅇ..
# issue : 모델이 data set에 넣었는데 나와서 id가 바뀌는 이슈가 있음 ( learning 후에 확인해 봐야 할 듯? ) ->
# 좀 심각한 문제 범위를 늘려볼까 ( 해결은 가능 h선 하나 만들어야 할 듯)
def count_obj(box, w, h, id, cls):
    global count, data, count_obj
    center_coordinates = (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2))  # ( w , h )
    # 알고리즘 개선 필요
    # if (w-500) < int(box[0]+(box[2]-box[0])/2) < (w-400)  and cls == 0 and id not in data and id not in counting_obj: # 첫번째에서 인식
    #     data.add(id)
    # if int(box[0]+(box[2]-box[0])/2) < (w-900) and cls == 0 and id in data and id not in counting_obj : # 한번 확인이 된 아이디이면 카운팅
    #     count += 1
    #     counting_obj.add(id)
    #     data.remove(id)

    # and int(box[1]+(box[3]-box[1])/2) <
    if (w - 800) < int(box[0] + (box[2] - box[0]) / 2) < (w - 400) and int(box[1] + (box[3] - box[1]) / 2) < (
            h - 300) and cls == 0 and id not in data and id not in counting_obj:  # 첫번째에서 인식
        data.add(id)
    if int(box[0] + (box[2] - box[0]) / 2) < (
            w - 900) and cls == 0 and id in data and id not in counting_obj:  # 한번 확인이 된 아이디이면 카운팅
        count += 1
        counting_obj.add(id)
        data.remove(id)


def video_feed(request):
    return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')
