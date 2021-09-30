import cv2




def show_cam(mirror=True):
	cam = cv2.VideoCapture(0)
	prev_image = None
	width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
	while True:
		ret_val, img  = cam.read()
		img = cv2.flip(img, 1)

		cv2.imshow('my webcam', img)
		if cv2.waitKey(1) == 27:
			break

	cv2.destroyAllWindows()

show_cam()
