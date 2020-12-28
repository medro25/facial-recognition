import cv2, sys, numpy, os
  # 0 for web camera live stream
#  for cctv camera'rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
#  example of cctv or rtsp: 'rtsp://mamun:123456@101.134.16.117:554/user=mamun_password=123456_channel=1_stream=0.sdp'
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
cascPath = 'haarcascade_frontalface_dataset.xml'  # dataset

(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(width, height) = (130, 100)
    # Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

    # OpenCV trains a model from the images
    # NOTE FOR OpenCV2: remove '.face'
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)

faceCascade = cv2.CascadeClassifier(haar_file)
video_capture = cv2.VideoCapture(0)
    
class faceVideo(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__ (self):
        self.video.release()
    def camera_stream(self):
        # Part 1: Create fisherRecognizer
        # Create a list of images and a list of corresponding names
        (_, im) = self.video.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            # Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 3)

            if prediction[1]<85:
                cv2.putText(im,'%s' % (names[prediction[0]]),(x, y-5), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 0),2)
            else:
                cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 0))

         # Capture frame-by-frame
        #ret, frame = video_capture.read()

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #faces = faceCascade.detectMultiScale(
         #   gray,
          #  scaleFactor=1.1,
           # minNeighbors=5,
            #minSize=(30, 30),
            #flags=cv2.CASCADE_SCALE_IMAGE
        #)

        # Draw a rectangle around the faces
        #for (x, y, w, h) in faces:
         #   cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame in browser
        _, jpeg = cv2.imencode('.jpg', im)
        return jpeg.tobytes()
