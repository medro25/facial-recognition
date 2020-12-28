import cv2
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class simpleVideo(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    
    def get_frame(self):
        _, im = self.video.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #print(gray)
        faces = facec.detectMultiScale(gray,1.3, 5)

        for (x, y, w, h) in faces:
         #   fc = gray_fr[y:y+h, x:x+w]

            #roi = cv2.resize(fc, (48, 48))
            #pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            #cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)

        _, jpeg = cv2.imencode('.jpg', im)
        #self.video.release()
        return jpeg.tobytes()
