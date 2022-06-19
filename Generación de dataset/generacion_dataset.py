import cv2
import imutils

detector_ojo_izquierdo = cv2.CascadeClassifier('/home/pi/TESINA/Generacion dataset/cascadas_haar/haarcascade_lefteye_2splits.xml')
detector_ojo_derecho = cv2.CascadeClassifier('/home/pi/TESINA/Generacion dataset/cascadas_haar/haarcascade_righteye_2splits.xml')
camara = cv2.VideoCapture(-1)
numeroMuestra = 1
totalMuestras = 500
while True:
    ret,img = camara.read()

    #ZOOM
    scale=18
    height, width, channels = img.shape

    centerX,centerY=int(height/2),int(width/2)
    radiusX,radiusY= int(scale*height/100),int(scale*width/100)

    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY

    cropped = img[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height))
    img = imutils.resize(resized_cropped, width=500)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ojos = detector_ojo_izquierdo.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in ojos:
        numeroMuestra += 1
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imwrite("/home/pi/TESINA/Generacion dataset/dataset/ojos_abiertos/ojo_abierto"+str(numeroMuestra)+".jpg",gray[y:y+h,x:x+w])
    cv2.imshow('Ojos',img)
    cv2.waitKey(100)
    if numeroMuestra == totalMuestras:
        camara.release()
        cv2.destroyAllWindows()
        break
