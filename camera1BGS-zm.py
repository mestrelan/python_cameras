import cv2
import sys
import numpy as np

from flask import Response, Flask, render_template
import threading
import argparse 
import datetime, time
import imutils


#Antigo
#cap = cv2.VideoCapture('rtsp://admin:abcd1234@192.168.0.64:554')
#Tiago
#cap = cv2.VideoCapture('rtsp://admin:abcd1234@172.20.0.168:554')

#Mateus
Video_Source = 'rtsp://admin:abcd1234@172.20.0.168:554'

#Video_Source = "S002C002P003R002A043_rgb.avi"


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
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)

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
		writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (640, 480), False)

	while(True):
		ret, frame = cap.read()
		frame=cv2.resize(frame,(960,540))
		
		bg_mask = bg_subtractor.apply(frame)
		bg_mask = cv2.medianBlur(bg_mask, 5)
		bg_mask = getFilter(bg_mask, 'dilation')
		bg_mask = getFilter(bg_mask, 'closing')
		bg_mask = getFilter(bg_mask, 'opening')
		th, bg_mask = cv2.threshold(bg_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


		##### Final
		frame = cv2.bitwise_and(frame, frame, mask=bg_mask)

		cv2.imshow('Capturing',frame)


		with lock:
			outputFrame = frame.copy()



		try:
			writer.write(frame)
		except:
			pass

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	#cap.release()
	#cv2.destroyAllWindows()
	return 0


#if __name__ == '__main__':
#	sys.exit(main(sys.argv))

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=False, default='172.20.255.3',
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
 
# release the video stream pointer
cap.release()
cv2.destroyAllWindows()