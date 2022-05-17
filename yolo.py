# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import argparse
import datetime, threading, time, cv2, torch, torch.backends.cudnn as cudnn
from ApiOperator import ApiOperator
from models.common import DetectMultiBackend
from utils.datasets import LoadStreams
from utils.general import (LOGGER, check_img_size, check_imshow, check_requirements,
                           non_max_suppression, scale_coords)
from utils.plots import Annotator, cropDetected
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def DetectorActivate():
    # System Configuration
    weights = "yolov5n.pt"
    imageSize = (480, 640)  # inference size (height, width)
    conf_threshold = 0.5  # confidence threshold
    iou_threshold = 0.5  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = "cpu"  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    update = False  # update all models
    line_thickness = 3  # bounding box thickness (pixels)
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imageSize = check_img_size(imageSize, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams('0', img_size=imageSize, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size
    global outputFrame, lock
    lock = threading.Lock()

    # Run inference
    model.warmup(imgsz=(1, 3, *imageSize), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    firstOn = True
    firstOff = True

    while True:
        # Keep Looping after login Successfully
        login = apiOperator.loginCamera()
        while login:

            # After 30 seconds update camera's setting
            if time.perf_counter() - apiOperator.loginTimer >= 30:
                print("Camera 30seconds check up...")
                login = apiOperator.loginCamera()

            # If operator order it to offline, IoT will not run the detector and tells the operator it is offline.

            # If not online / 1
            if apiOperator.camera["operationStatus"] != 1:
                # Telling User the Camera is Stopping Once
                if firstOff:
                    apiOperator.updateOperatingStatus(False)
                    print("- Operation Status Offline Detected. Stopping Operation -")
                    firstOff = False
                    firstOn = True

            else:
                # When operator orders it to online, the IoT report to operator it will be online then start the Human
                # Detection.

                # Telling User the Camera is Working Once
                if firstOn:
                    apiOperator.updateOperatingStatus(True)
                    print("- Operation Status Online Detected. Starting Operation -")
                    firstOff = True
                    firstOn = False

                for path, im, im0s, vid_cap, s in dataset:

                    # After 30 seconds update camera's setting
                    if time.perf_counter() - apiOperator.loginTimer >= 10:
                        print("Camera 30seconds check up...")
                        login = apiOperator.loginCamera()
                        break

                    t1 = time_sync()
                    im = torch.from_numpy(im).to(device)
                    im = im.half() if half else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    t2 = time_sync()
                    dt[0] += t2 - t1

                    # Inference
                    pred = model(im, augment=augment, visualize=visualize)
                    t3 = time_sync()
                    dt[1] += t3 - t2

                    fps = 1 / (t3 - t2)
                    fps = int(fps)

                    # NMS
                    pred = non_max_suppression(pred, conf_threshold, iou_threshold, classes, agnostic_nms,
                                               max_det=max_det)
                    dt[2] += time_sync() - t3

                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        seen += 1
                        p, imageOriginal, frame = path[i], im0s[i].copy(), dataset.count

                        s += f'{i}: '

                        # For Verify if Same Person. Frame without Mark, without Text.
                        imageCroppedForSave = imageOriginal.copy()

                        s += '%gx%g ' % im.shape[2:]  # print string

                        # Show window of Detection
                        annotator = Annotator(imageOriginal, line_width=line_thickness)

                        # Create Marks by User
                        if len(apiOperator.cameraCordsMarks) != 0:
                            annotator.mergeMarks(apiOperator.cameraEnterMarks, apiOperator.cameraExitMarks)

                        # For Verify if entered or exited. Frame with Mark, without Text.
                        imageCroppedForVerify = imageOriginal.copy()

                        # Code adapted from phoenixNAP, 2019.
                        cv2.putText(imageOriginal, "DateTime: " + str(
                            datetime.datetime.now().strftime("Y%Y/M%m/D%d %I:%M:%S %p")) + " | FPS: " + str(fps),
                                    (10, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        # End of code adapted.

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], imageOriginal.shape).round()

                            # Print Number Human Detected Results
                            humanDetected = len(det[:, -1])
                            if humanDetected == 0:
                                s += "No people Detected"
                            else:
                                s += f"{humanDetected} Person{'s' * (humanDetected > 1)}, "

                            # Write results, xyxy = Cropped Image for detected object/person
                            for *xyxy, conf, cls in reversed(det):

                                # Add Box Label to Detected Humans
                                if view_img:
                                    annotator.box_label(xyxy)

                                cropDetected(xyxy, im=imageCroppedForVerify, im2=imageCroppedForSave)

                        # Print time (inference-only)
                        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s) | FPS: {fps}')

                        if view_img:
                            # Stream results
                            imageOriginal = annotator.result()
                            cv2.imshow(str(p), imageOriginal)
                            cv2.waitKey(1)  # 1 millisecond

                        # Upload image if stream is on
                        if apiOperator.camera["streamStatus"] == 1:
                            apiOperator.postStreamInput(imageOriginal)


if __name__ == "__main__":
    apiOperator = ApiOperator()
    check_requirements(exclude=('tensorboard', 'thop'))
    DetectorActivate()
