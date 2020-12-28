from flask import Flask, render_template, Response, request, url_for
from flask import *
from flask_mail import *
from flask_mail import Mail,Message
from email import *
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import os
import pyscreenshot
import random
import string
from camera import VideoCamera
from simplecamera import simpleVideo
#from face import faceVideo
#from face import camera_stream
from model import listN, listA, listH, listSurprise,listSad,listD,listF
import matplotlib.pyplot as plt
import numpy as np
import cv2, sys, numpy, os


app = Flask(__name__)


@app.route('/get_screenshot', methods=['POST'])
def get_screenshot():
    if request.method == 'POST':
        username = request.form.get('username')
        #print(username)
    stdm=''    
    with open('stdEmail.txt') as fp:
            for mail in fp:
                mail = mail.strip().split('/n')
                stdm=mail[0]
    stdm=str(stdm)
    #print(stdm)
    pw=''
    with open('password.txt') as fp2:
        for passw in fp2:
            passw = passw.strip().split('/n')
            pw=passw[0]
        
    app.config['MAIL_SERVER'] = 'mail.gmail.com'
    app.config['MAIL_PORT'] = 465
    app.config['MAIL_USE_SSL']= True

    mail_settings = {
        "MAIL_SERVER": 'smtp.gmail.com',
        "MAIL_USE_TLS": False,
        "MAIL_USE_SSL": True,
        "MAIL_PORT": 465,
        "MAIL_USERNAME": stdm,
        "MAIL_PASSWORD": pw
                
    }

    app.config.update(mail_settings)
    mail= Mail(app)
    im = pyscreenshot.grab(bbox=(540,90,1230,580))
    random_id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])
    file_name = '{}.png'.format(random_id)
    im.save(file_name)
    #print(file_name)
    file_name=file_name.replace('.png','')
    #print(file_name)
    msg = Message('Emotion Results',sender=stdm,recipients=[username])
    with app.open_resource(file_name+".png") as fp:
        msg.attach(file_name+".png", file_name+"/png", fp.read())
    mail.send(msg)
    return render_template('home.html')


def gen_frame2(face):
    #from face import camera_stream
    """Video streaming generator function."""
    while True:
        frame4 = face.camera_stream()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame4 + b'\r\n') # concate frame one by one and show result


@app.route('/video_feed4')
def video_feed4():
    from face import faceVideo
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame2(faceVideo()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen(simplecamera):
    #cv2.destroyAllWindows
    while True:
        frame2 = simplecamera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')

@app.route('/video_feed3')
def video_feed3():
    return Response(gen(simpleVideo()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/exec4')
def index2():
    """Video streaming home page."""
    return render_template('face.html')

def gen_frame2(face):
    #from face import camera_stream
    """Video streaming generator function."""
    while True:
        frame1 = face.camera_stream()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n') # concate frame one by one and show result


@app.route('/video_feed2')
def video_feed2():
    from face import faceVideo
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame2(faceVideo()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/exec3')
def index():
    return render_template('index.html')

def gen(camera):
    #cv2.destroyAllWindows
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def first():
    #cv2.destroyAllWindows
    return render_template('front.html')
@app.route('/front.html', methods=['GET','POST'])
def ff():
    cv2.destroyAllWindows()
    return render_template('front.html')


@app.route('/signup', methods=['post', 'get'])
def signup():
    #cv2.destroyAllWindows
    message = ''
    if request.method == 'POST':
        username = request.form.get('username')  # access the data inside
        #print(username)
        cv2.destroyAllWindows
        cv2.VideoCapture(0).release()
        haar_file = 'haarcascade_frontalface_default.xml'
        datasets = 'datasets'  #All the faces data will be present this folder
        sub_data = username     #These are sub data sets of folder, for my faces I've used my name
        f = open("name.txt","a+")
        f.write(username)
        f.write("\n")
        f.close()
        path = os.path.join(datasets, sub_data)
        #print(path)
        if not os.path.isdir(path):
            os.mkdir(path)
        (width, height) = (130, 100)    # defining the size of image

        face_cascade = cv2.CascadeClassifier(haar_file)
        webcam = cv2.VideoCapture(0) #'0' is use for my webcam, if you've any other camera attached use '1' like this

        # The program loops until it has 30 images of the face.
        count = 1
        while count < 30: 
            (_, im) = webcam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite('%s/%s.png' % (path,count), face_resize)
            count += 1
                
            #cv2.imshow('OpenCV', im)
            key = cv2.waitKey(10)
            if key == 27:
                break
            #return render_template('front.html')
        if len(os.listdir(path))==0:
            webcam.release()
            return render_template('signuperror.html')
        if username == str(username):
            webcam.release()
            cv2.destroyAllWindows()
            message = "Your name is saved"
            return render_template('front.html')
        else:
            message = "Sorry try again"
    return render_template('signup.html')

@app.route('/login', methods=['post', 'get'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password=request.form.get('password')
        #print(email)
        #print(username)
        fp2=open("password.txt","w")
        fp2.write(password)
        fp2.write("\n")
        fp2.close()
        fp=open("StdEmail.txt","w")
        fp.write(email)
        fp.write("\n")
        fp.close()
        with open('name.txt') as f1:
            for line in f1:
                line = line.strip().split('/n')
                if username in line:
                    size=4
                    fa=''
                    fn=''
                    haar_file = 'haarcascade_frontalface_default.xml'
                    datasets = 'datasets'
                    # Part 1: Create fisherRecognizer
                    # Create a list of images and a list of corresponding names
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
                    if (len(images) == 0 and len(lables) == 0):
                        return render_template('loginerror.html')
                    model = cv2.face.LBPHFaceRecognizer_create()
                    model.train(images, lables)

                    # Part 2: Use fisherRecognizer on camera stream
                    face_cascade = cv2.CascadeClassifier(haar_file)
                    webcam = cv2.VideoCapture(0)
                    while True:
                        (_, im) = webcam.read()
                        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        for (x,y,w,h) in faces:
                            cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
                            face = gray[y:y + h, x:x + w]
                            face_resize = cv2.resize(face, (width, height))
                            # Try to recognize the face
                            prediction = model.predict(face_resize)
                            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 3)

                            if prediction[1]<65:
                                cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 0))
                                fa='face found'
                                cv2.destroyAllWindows
                                return render_template('home.html')
                                
                            else:
                                cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 0))
                                fn='face not found'
                                
                        cv2.imshow('OpenCV', im)
                        
                        if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
                            break
                        
                        key = cv2.waitKey(10)
                        if key == 27:
                            break

    return render_template('login.html')

@app.route('/exec6')
def home3():
    cv2.destroyAllWindows
    cv2.VideoCapture(0).release()
    return render_template('home.html')
@app.route('/exec5')
def home2():
    emotions=['Angry','Sad','Happy','Neutral','Surprise','Fear']
    pos=np.arange(len(emotions))
    hp_index=[len(listA),len(listSad),len(listH),len(listN),len(listSurprise),len(listF)]
    bar_labels=emotions
    bar_values=hp_index
    return render_template('bar_chart.html', title='Emotion Results', max=200, labels=bar_labels, values=bar_values)

if __name__ == '__main__':
    app.run()
