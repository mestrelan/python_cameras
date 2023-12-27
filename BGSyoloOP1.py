from ultralytics import YOLO
import cv2
import math 

import sys
import os
import io
import time
import numpy as np

from flask import Response, Flask, render_template
import threading
import argparse 
import datetime, time
import imutils

#Para salvar a saida em um arquivo txt e no console
class OutputTee(io.TextIOBase):
    def __init__(self, filename):
        self.stdout = sys.stdout
        self.file = open(filename, 'a')
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.stdout.flush()
        self.file.flush()

def get_traceback():
    traceback_output = io.StringIO()
    traceback.print_exc(file=traceback_output)
    return traceback_output.getvalue()

def salvar_saidas():
    if not os.path.exists("saidas"):
        os.mkdir("saidas")

    num_saida = 1
    while os.path.exists(f"saidas/saida{num_saida}.txt"):
        num_saida += 1

    nome_arquivo_saida = f"saidas/saida{num_saida}.txt"

    return OutputTee(nome_arquivo_saida)
#até aqui

#Antigo
#cap = cv2.VideoCapture('rtsp://admin:abcd1234@192.168.0.64:554')
#Tiago
#cap = cv2.VideoCapture('rtsp://admin:abcd1234@172.20.0.168:554')

#Mateus
#Video_Source = 'rtsp://admin:abcd1234@172.20.0.168:554'

Video_Source = "S002C002P003R002A043_rgb.avi"

BGS_TYPES = ["FMT", "GMG", "MOG", "MOG2", "KNN", "CNT"]
BGS_TYPE = BGS_TYPES[4]

#Stream http
outputFrame = None
lock = threading.Lock()
app = Flask(__name__)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
 
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
 
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
 
            # ensure the frame was successfully encoded
            if not flag:
                continue
 
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')



def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == "opening":
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == "closing":
        kernel = np.ones((3, 3), np.uint8)

    return kernel

def getFilter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=10)

    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)

    if filter == 'dilation':
        return cv2.dilate(img, getKernel("dilation"), iterations=2)

    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)
        dilation = cv2.dilate(opening, getKernel("dilation"), iterations=2)

        return dilation

def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN(history=5000)
    if BGS_TYPE == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15*3, useHistory=True, maxPixelStability=15*60*3, isParallel=True)
    print("Detector invÃ¡lido")
    sys.exit(1)


video_out = False
if video_out:
    VIDEO_OUT = 'salvo.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (640, 480), False)

cap = cv2.VideoCapture(Video_Source)
cap.set(3, 640)
cap.set(4, 480)


##Yolo#
# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

##OpenPose#
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

threshold=0.2
image_width=960
image_height=540
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")


def stream(frame_count):
    global outputFrame, lock
    #Antigo
    #cap = cv2.VideoCapture('rtsp://admin:abcd1234@192.168.0.64:554')
    #Tiago
    #cap = cv2.VideoCapture('rtsp://admin:abcd1234@172.20.0.168:554')
    #Mateus
    cap = cv2.VideoCapture(Video_Source)
    time.sleep(2.0)
    bg_subtractor = getBGSubtractor(BGS_TYPE)
    video_out = False
    if video_out:
        VIDEO_OUT = 'salvo.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (960, 540), False)

    contador_frame_real=0
    while(True):
        inicio_temptotal=time.time()
        inicio_preproc=time.time()
        contador_frame_real+=1

        #success, img = cap.read()
        ret, frame = cap.read()
        frame=cv2.resize(frame,(960,540))

        bg_mask = bg_subtractor.apply(frame)
        #bg_mask = cv2.medianBlur(bg_mask, 5)
        #bg_mask = getFilter(bg_mask, 'dilation')
        bg_mask = getFilter(bg_mask, 'closing')
        #bg_mask = getFilter(bg_mask, 'opening')
        th, bg_mask = cv2.threshold(bg_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


        ##### Final
        frame = cv2.bitwise_and(frame, frame, mask=bg_mask)
        fim_preproc = time.time()
        
        
        ####### YOLO ############
        inicio_yolo=time.time()
        results = model(frame, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
        fim_yolo=time.time()
        ####### YOLO ############
        
        ####### OpenPose ############
        inicio_OP=time.time()
        photo_height=frame.shape[0]
        photo_width=frame.shape[1]
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (image_width, image_height), (127.5, 127.5, 127.5), swapRB=True, crop=False))

        out = net.forward()
        out = out[:, :19, :, :] 

        assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
                # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

                # Originally, we try to find all the local maximums. To simplify a sample
                # we just find a global one. However only a single pose at the same time
                # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (photo_width * point[0]) / out.shape[3]
            y = (photo_height * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > threshold else None)


        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        t, _ = net.getPerfProfile()

        fim_OP=time.time()
        ####### OpenPose ############

        inicio_posproc=time.time()
        cv2.imshow('Capturing',frame)


        with lock:
            outputFrame = frame.copy()

            if (contador_frame_real%15==0):
                contador_write=str(contador_frame_real)
                #write_path = cwd +'\\saida\\'+ contador_write +'.jpg'
                write_path = 'frames/'+contador_write +'.jpg'
                #print(write_path)
                cv2.imwrite(write_path, outputFrame)
                #frame_show = cv2.resize(frame, (320, 240))
                #cv2_imshow(frame_show)

        try:
            writer.write(frame)
        except:
            pass
        
        fim_posproc=time.time()
        fim_temptotal = time.time()
        temptotal = fim_temptotal - inicio_temptotal
        temp_preproc = fim_preproc - inicio_preproc
        tempyolo = fim_yolo - inicio_yolo
        tempOP = fim_OP - inicio_OP
        temp_posproc = fim_posproc - inicio_posproc
        print("###### Tempos ######")
        print("tempo Total:")
        print(temptotal)
        print("Tempo de pré-processamento:")
        print(temp_preproc)
        print("Tempo de processamento do yolo:")
        print(tempyolo)
        print("Tempo de processamento do OpenPose:")
        print(tempOP)
        print("Tempo de pos processamento:")
        print(temp_posproc)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # release the video stream pointer
    cap.release()
    cv2.destroyAllWindows()
    return 0


#if __name__ == '__main__':
#   sys.exit(main(sys.argv))

# check to see if this is the main thread of execution
if __name__ == '__main__':
    
    # Redireciona a saída para o arquivo
    saida_arquivo = salvar_saidas()
    # Todas as saídas a partir daqui serão redirecionadas para o arquivo e o console

    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    #colocar o ip do pc, usar o comando ifconfig ou ipconfig para saber o ip
    ap.add_argument("-i", "--ip", type=str, required=False, default='192.168.47.147',
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=False, default=8000, 
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    t = threading.Thread(target=stream, args=(args["frame_count"],))
    t.daemon = True
    t.start()
 
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
 

