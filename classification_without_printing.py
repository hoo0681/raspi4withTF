import tflite_runtime.interpreter as tflite
import numpy as np
import cv2 

def inv_contour(image,mask,x,y,w,h):
    inv_mask=mask[y:y+h,x:x+w]^0xFF
    inv_mask=inv_mask/255
    inv_mask=inv_mask.astype('uint8')
    contours, hierarchy = cv2.findContours(inv_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c0=max(contours, key = cv2.contourArea)
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
labels={1:'dog',0:'cat'}
cap=cv2.VideoCapture(-1)
while(cap.isOpened()):
    ret,frame=cap.read()
    cv2.imshow('cam',frame)
    key=cv2.waitKey(1)
    if key==ord('s') :
        ret,frame=cap.read()
        if(ret):
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
            contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)##컨투어 찾기
            c0=max(contours, key = cv2.contourArea)
            ####가장큰컨투어찾기##
            x0, y0 = zip(*np.squeeze(c0))
            x, y, w, h = cv2.boundingRect(c0)
            result_image= cv2.rectangle(frame, (x, y), (x+w, y+h),(0,0,255), 7)
            cv2.imshow('result_image',result_image)
            ####컨투어박스치기####
            target_image=0
            target_image=inv_contour(RGB_frame,red_mask_2,x,y,w,h)
            
            ####강아지만 자르기###
            test_data=cv2.resize(target_image,(150,150))
            dic['input']()[0][:,:]=test_data
            dic['model'].invoke()
            ans=labels[np.argmax(dic['output']()[0])]
            print(ans)
    ####추론#############
    elif(key==27):
        break
cap.release()
cv2.destroyAllWindows()
####그리기############