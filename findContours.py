import cv2
import numpy as np

cap=cv2.VideoCapture(-1)
while(cap.isOpened()):
    ret,frame=cap.read()
    frame= cv2.resize(frame,(205*2,154*2))
    frame=cv2.flip(frame,0)
    RGB_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    ####RGB파일로 변환####
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    red_mask_2=cv2.inRange(hsv,(170,190,80),(180,255,255))
    ####빨간색필터거치기##
    bin_mask=red_mask_2/255
    bin_mask=bin_mask.astype('uint8')#contour를 찾기위해서는 소스이미지가 단일 채널의 8비트 이여야한다!!!!
    ####이진화############
    RGB_frame_copy=RGB_frame.copy()
    contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)##컨투어 찾기
    cv2.drawContours(RGB_frame_copy, contours, -1, (0,255,0), 3)
    mm=0
    maxarea=0
    for i,cnt in zip(range(0,len(contours)), contours):
        if maxarea<cv2.contourArea(cnt):
            maxarea=cv2.contourArea(cnt)
            mm=i
    ####가장큰컨투어찾기##
    c0=contours[mm]
    x0, y0 = zip(*np.squeeze(c0))
    x, y, w, h = cv2.boundingRect(c0)
    ####컨투어박스치기####
    result_image=frame[y:y+h,x:x+w,:]
    BGR_frame=cv2.cvtColor(RGB_frame_copy,cv2.COLOR_RGB2BGR)
    if(ret):
        cv2.imshow('image',frame)
        cv2.imshow('contours',BGR_frame)
        cv2.imshow('result',result_image)
        k=cv2.waitKey(1)&0xFF
        if(k==27):
            break
cap.release()
cv2.destroyAllWindows()
####그리기############