import cv2
import sys
import logging as log
import datetime as dt
from skimage import io
import dlib
import time



log.basicConfig(filename='webcam.log',level=log.INFO)

predictor_path = '/Users/vovanmozg/Downloads/bigdata/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    dets = detector(frame, 1)

    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30)
    #     # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    # )

    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(frame, d)


        output = frame[d.top():d.bottom(), d.left():d.right()] # Crop from x, y, w, h -> 100, 200, 300, 400
        fname = 'data/' + dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.jpg'
        print(fname)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        io.imsave(fname, output)

   
    #if anterior != len(faces):
    #    anterior = len(faces)
    #    log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #time.sleep(1) # delays for 5 seconds

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
