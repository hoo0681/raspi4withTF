import cv2
import numpy as np
cap=cv2.VideoCapture(-1)
while(cap.isOpened()):
	ret,frame=cap.read()
	frame=cv2.flip(frame,0)
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	red_mask_2=cv2.inRange(hsv,(170,100,0),(180,255,255))
	mask=red_mask_2
	res=cv2.bitwise_and(frame,frame,mask=mask)
	if(ret):
		cv2.imshow('image',frame)
		cv2.imshow('mask',mask)
		cv2.imshow('result',res)
		k=cv2.waitKey(1)&0xFF
		if(k==27):
			break;
cap.release()
cv2.destroyAllWindows()
