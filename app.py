from flask import Flask,render_template,Response, request
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
from flask_mail import Mail, Message
#from app import USER_NAME, RES_MAIL,D_COUNT


'''mixer.init()
sound = mixer.Sound('warning-alarm.wav')

face = cv2.CascadeClassifier('HAARCASCADEfiles\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('HAARCASCADEfiles\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('HAARCASCADEfiles\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

model = load_model('model_eye.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]'''

app=Flask(__name__)

USER_NAME = ""
RES_MAIL = ""
D_COUNT = 0
FLAG = 0

face = cv2.CascadeClassifier('HAARCASCADEfiles\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('HAARCASCADEfiles\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('HAARCASCADEfiles\haarcascade_righteye_2splits.xml')

camera=cv2.VideoCapture(0)

mixer.init()
sound = mixer.Sound('warning-alarm.wav')


def send_mail():
    if USER_NAME == "":
        return '', 204
    with app.app_context():
        app.config['MAIL_SERVER']='smtp.gmail.com'
        app.config['MAIL_PORT'] = 465
        app.config['MAIL_USERNAME'] = 'droive101@gmail.com'
        app.config['MAIL_PASSWORD'] = 'akrhswme'
        app.config['MAIL_USE_TLS'] = False
        app.config['MAIL_USE_SSL'] = True
        mail = Mail(app)
        msg = Message('Alert!!!', sender ='foiger.dreams@gmail.com', recipients = [RES_MAIL])
        msg.subject = "Saftey Alert!!"
        msg.body = ('This is a warning from Droive! \n' + USER_NAME + " is feeling drowsy while driving.")
        mail.send(msg)
        print("execute")
        global D_COUNT
        D_COUNT = 0
    return '', 204

def generate_frames():
    '''face = cv2.CascadeClassifier('HAARCASCADEfiles\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('HAARCASCADEfiles\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('HAARCASCADEfiles\haarcascade_righteye_2splits.xml')'''

    lbl=['Close','Open']

    model = load_model('model_eye.h5')
    path = os.getcwd()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count=0
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]
    global D_COUNT
    
    while True:
            
        ## read the camera frame
        ret,frame=camera.read()
        height,width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(48,48))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(48,48,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            #rpred = model.predict_classes(r_eye)
            predict_x=model.predict(r_eye)
            rpred=np.argmax(predict_x,axis=1)
            if(rpred[0]==1):
                lbl='Open'
            if(rpred[0]==0):
                lbl='Closed'
            break

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye,(48,48))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(48,48,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            #lpred = model.predict_classes(l_eye)
            predict_x=model.predict(l_eye) 
            lpred=np.argmax(predict_x,axis=1)
            if(lpred[0]==1):
                lbl='Open'
            if(lpred[0]==0):
                lbl='Closed'
            break
              
        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Closed: You are feeling drowsy!! ALERT!!",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
        else:
            score=score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            
        
        if(score<0):
            score=0
                
        if(score>10):            
            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            D_COUNT += 1
            try:
                sound.play()
            except:  # isplaying = False
                pass
            if(thicc<14):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
            
        global FLAG
        if (D_COUNT) > 2 and FLAG == 0:
            send_mail()
            FLAG = 1
        
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
        if not ret:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Aboutus')
def Aboutus():
    return render_template('Aboutus.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/Droive')
def Droive():
    return render_template('Droive.html')

@app.route('/', methods =["GET", "POST"])
def get_info():
    if request.method == "POST":
       global USER_NAME
       USER_NAME = request.form.get("name")
       global RES_MAIL
       RES_MAIL = RES_MAIL + str(request.form.get("r_email"))
       return '', 204


if __name__=="__main__":
    app.run()
