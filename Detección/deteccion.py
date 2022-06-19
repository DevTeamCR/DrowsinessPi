from matplotlib import image
from tflite_runtime.interpreter import Interpreter
import tflite_runtime as tflite
from PIL import Image
import numpy as np
import time
import cv2
import imutils
import RPi.GPIO as GPIO

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(18,GPIO.OUT)

ruta_modelo = "./model_mobilenet.tflite"
left_eye_cascade = cv2.CascadeClassifier('./detectores/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('./detectores/haarcascade_righteye_2splits.xml')

def cargarModelo(ruta):
    interpreter = Interpreter(ruta)
    print("Modelo cargado!")
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    return interpreter
    
def clasificacion(interpreter, img):
    img = cv2.resize(img, (165,165))
    img.shape
    output_details = interpreter.get_output_details()

    input_tensor = np.array(np.expand_dims(img,0), dtype=np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediccion = np.squeeze(output_data)
    return  (0 if (prediccion < 0.5)  else 1)


def detectar():
    cap = cv2.VideoCapture(-1)
    interpreter = cargarModelo(ruta_modelo)
    pred_izq = ''
    pred_der = ''
    cont_somnolencia, cont_ojos_cerrados = 0, 0
    

    print('Detectando...')
    while True:
        ret, img = cap.read()

        #ZOOM
        scale=18
        height, width, channels = img.shape

        #prepare the crop
        centerX,centerY=int(height/2),int(width/2)
        radiusX,radiusY= int(scale*height/100),int(scale*width/100)

        minX,maxX=centerX-radiusX,centerX+radiusX
        minY,maxY=centerY-radiusY,centerY+radiusY

        cropped = img[minX:maxX, minY:maxY]
        resized_cropped = cv2.resize(cropped, (width, height)) 

        img = imutils.resize(resized_cropped, width=500)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        min_size = (10,10)
        left_eye = left_eye_cascade.detectMultiScale(gray,1.3,7, cv2.CASCADE_SCALE_IMAGE,minSize=min_size)
        right_eye = right_eye_cascade.detectMultiScale(gray,1.3,7, cv2.CASCADE_SCALE_IMAGE, minSize=min_size)

        for (ex,ey,ew,eh) in left_eye:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            roi_izq = img[ey:ey+eh,ex:ex+ew]
            pred_izq = clasificacion(interpreter,roi_izq)



        for (edx,edy,edw,edh) in right_eye:
            cv2.rectangle(img,(edx,edy),(edx+edw,edy+edh),(0,255,0),2)
            roi_der = img[edy:edy+edh,edx:edx+edw]
            pred_der = clasificacion(interpreter,roi_der)

        print('#################################')
        print('Ojo izquiedo: ',pred_izq)
        print('Ojo derecho: ',pred_der)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if(pred_izq == 1 and pred_der == 1):
            cont_ojos_cerrados += 1
            if(cont_ojos_cerrados == 3):
                cont_somnolencia += 1
        else:
            cont_ojos_cerrados = 0

        #Verificar alarma
        if(cont_ojos_cerrados >= 3):
            cv2.putText(img, 
                'ALERTA!', 
                (250, 350), 
                font, 2, 
                (0, 0, 255), 
                3, 
                cv2.LINE_4)
            GPIO.output(18,True)
        else: 
            GPIO.output(18,False)

        ##Imprime contador de somnolencia            
        cv2.putText(img, 
                    'Contador de somnolencia: '+ str(cont_somnolencia), 
                    (0,40), 
                    font, 1, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
            
        cv2.imshow('img',img)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
        

detectar()
