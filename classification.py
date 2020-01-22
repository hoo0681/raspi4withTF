import tflite_runtime.interpreter as tflite
import numpy as np
import cv2 

def inv_contour(image,mask,x,y,w,h):
    inv_mask=mask[y:y+h,x:x+w]^0xFF
    inv_mask=inv_mask/255
    inv_mask=inv_mask.astype('uint8')
    contours, hierarchy = cv2.findContours(inv_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mm=0
    maxarea=0
    for i,cnt in zip(range(0,len(contours)), contours):
        if maxarea<cv2.contourArea(cnt):
            maxarea=cv2.contourArea(cnt)
            mm=i
    c0=contours[mm]
    X_,Y_,W_,H_=cv2.boundingRect(c0)
    return image[y+Y_:y+Y_+H_,x+X_:x+X_+W_,:]

def load_interperter(model_path):
    interpreter_=tflite.Interpreter(model_path=model_path)
    interpreter_.allocate_tensors()
    input_ =interpreter_.tensor(interpreter_.get_input_details()[0]['index'])
    output = interpreter_.tensor(interpreter_.get_output_details()[0]["index"])
    return {'model':interpreter_,'input':input_,'output':output}
def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


dic=load_interperter("/home/pi/raspi4withTF/tflitemodel.tflite")
#labels=load_labels("/home/pi/Downloads/labels_mobilenet_quant_v1_224.txt")
labels={0:'dog',1:'cat'}
cap=cv2.VideoCapture(-1)
while(cap.isOpened()):
    ret,frame=cap.read()
    frame= cv2.resize(frame,(205*2,154*2))
    frame=cv2.flip(frame,0)
    RGB_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    ####RGB파일로 변환####
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    red_mask_2=cv2.inRange(hsv,(170,100,80),(180,255,255))
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
    target_image=0
    target_image=inv_contour(frame,red_mask_2,x,y,w,h)
    ####강아지만 자르기###
    test_data=cv2.resize(target_image,(150,150))
    dic['input']()[0][:,:]=test_data
    dic['model'].invoke()
    ans=labels[np.argmax(dic['output']()[0])]
    ####추론#############
    result_image=frame[y:y+h,x:x+w,:]
    BGR_frame=cv2.cvtColor(RGB_frame_copy,cv2.COLOR_RGB2BGR)
    cv2.putText(frame,ans,(0,100), cv2.FONT_ITALIC, 1, (0, 0, 0))
    if(ret):
        cv2.imshow('image',frame)
        cv2.imshow('contours',BGR_frame)
        cv2.imshow('cutimage',result_image)
        cv2.imshow('result',target_image)
        k=cv2.waitKey(1)&0xFF
        if(k==27):
            break
cap.release()
cv2.destroyAllWindows()
####그리기############