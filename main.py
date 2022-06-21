import time

import numpy as np
import pyautogui
#from cv2 import cv2

import cv2

import pyautogui
import pywinauto
from mss import mss


prev_frame_time = 0
new_frame_time = 0
start_time = time.time()
font = cv2.FONT_HERSHEY_DUPLEX
mon = {'top': 300, 'left': 740, 'width': 480, 'height': 400}
CONFIG_FILE = 'yolov4-tiny-custom.cfg'
WEIGHT_FILE = 'yolov4-tiny-custom_final.weights'
cv2.getBuildInformation()
net = cv2.dnn.readNet(CONFIG_FILE, WEIGHT_FILE) #
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
oln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = ["head"]
size_scale = 2
#pyautogui.moveTo(100, 150)
a = []
with mss() as sct:
    while True:
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        img1 = sct.grab(mon)
        img_np = np.array(img1)
        img=cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        cv2.putText(img, fps, (7, 70), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        height, width,channels= img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (640, 640), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(oln)
        class_ids = []
        boxes = []
        confidences = []

        for output in outs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.8:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

                    class_ids.append(classID)
        indixes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        for i in range(len(boxes)):
            if i in indixes:
                x, y, w, h = boxes[i]
                cx = int((x + w))
                cy = int((y + h))
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                hh=int(h/2)
                ww=int(w/2)
                pyautogui.PAUSE = 0
                #a.append(x+y)
                #while xx==x and yy==y:
                #pywinauto.mouse.move(coords=(x+740+ww,y+300+hh) )
                pyautogui.moveTo(x+740+ww, y+300+hh)
                #pyautogui.click(clicks=2, interval=0.25)
                #pyautogui.press("shiftleft")
                #time.sleep(0.15)
                #print(x+740,y+300)

        cv2.imshow("frame", img)
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()