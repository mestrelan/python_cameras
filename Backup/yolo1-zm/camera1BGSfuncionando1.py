import cv2
import sys
import numpy as np

#Antigo
#cap = cv2.VideoCapture('rtsp://admin:abcd1234@192.168.0.64:554')
#Tiago
#cap = cv2.VideoCapture('rtsp://admin:abcd1234@172.20.0.168:554')

#Mateus
#Video_Source = 'rtsp://admin:abcd1234@172.20.0.168:554'

Video_Source = "S002C002P003R002A043_rgb.avi"

BGS_TYPES = ["FMT", "GMG", "MOG", "MOG2", "KNN", "CNT"]
BGS_TYPE = BGS_TYPES[4]

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

def main(args):
	#Antigo
	#cap = cv2.VideoCapture('rtsp://admin:abcd1234@192.168.0.64:554')
	#Tiago
	#cap = cv2.VideoCapture('rtsp://admin:abcd1234@172.20.0.168:554')
	#Mateus
	cap = cv2.VideoCapture(Video_Source)
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

		try:
			writer.write(frame)
		except:
			pass

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))