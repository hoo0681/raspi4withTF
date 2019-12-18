import cv2
cap=cv2.VideoCapture(-1)
while(cap.isOpened()):
	ret,frame=cap.read()
	if(ret):
		frame=cv2.flip(frame,0)
		#gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		#frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
		cv2.imshow('result',frame)
		k=cv2.waitKey(1)&0xFF
		if(k==27):
			break;
cap.release()
cv2.destroyAllWindows()
