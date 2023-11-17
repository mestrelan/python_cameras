import cv2
import sys

def main(args):
	#Antigo
	#cap = cv2.VideoCapture('rtsp://admin:abcd1234@192.168.0.64:554')
	#Tiago
	#cap = cv2.VideoCapture('rtsp://admin:abcd1234@172.20.0.168:554')
	#Mateus
	cap = cv2.VideoCapture('rtsp://admin:abcd1234@172.20.0.168:554')
	while(True):
		ret, frame = cap.read()
		frame=cv2.resize(frame,(960,540))
		cv2.imshow('Capturing',frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))