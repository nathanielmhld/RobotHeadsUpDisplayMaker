import cv2
import sys
import numpy as np
import logging as log
import datetime as dt
from time import sleep
import pprint
import random
import math
import argparse
import io


import moviepy.editor as mp
    
FRAMERATE = 23.989140
DIALOGUEDELAY = .3
INPUT = 'short.mp4'
OUTPUT = 'output.mp4'
#facial recognition

# [START speech_transcribe_async_word_time_offsets_gcs]
def transcribe_file(speech_file):
    """Transcribe the given audio file asynchronously and output the word time
    offsets."""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='en-US',
        enable_word_time_offsets=True)

    operation = client.long_running_recognize(config, audio)

    print('Waiting for operation to complete...')
    result = operation.result(timeout=90)
    retval = [{}]

    for result in result.results:
        alternative = result.alternatives[0]
        print(u'Transcript: {}'.format(alternative.transcript))
        print(u'Confidence: {}'.format(alternative.confidence))
        
        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            if start_time.seconds + start_time.nanos * 1e-9 in retval[-1]:
                retval[-1][start_time.seconds + start_time.nanos * 1e-9] = retval[-1][start_time.seconds + start_time.nanos * 1e-9] + " " + word
            else:
                retval[-1][start_time.seconds + start_time.nanos * 1e-9] = word

        retval.append({})
# [END speech_transcribe_async_word_time_offsets_gcs]
    return retval


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

mp.VideoFileClip(INPUT).audio.write_audiofile('sound.wav', ffmpeg_params=["-ac", "1"])
dialogue = transcribe_file('sound.wav')

video_capture = cv2.VideoCapture(INPUT) #cv2.VideoCapture(0)

width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('temp.mp4',fourcc, FRAMERATE, (int(width),int(height)))



anterior = 0
i=0
a=0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass


    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #translayer = np.zeros((len(frame), len(frame[0]), 4))
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        print("ouch")
        break

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    #angrilyscrollingText
    a -= 25
    if a < -33315*30:
       a = 0 
    filepath = 'haarcascade_frontalface_default.xml'  
    with open(filepath) as fp:  
        line = fp.readline()
        cnt = 1
        while line:
            if a + cnt*30 + 300 > - 600 and a + cnt*30 + 300 < 0:
                font                   = cv2.FONT_HERSHEY_DUPLEX
                bottomLeftCornerOfText = (int(width) - 900,int(height)+ a + cnt*30  + 300)
                fontScale              = 1
                fontColor              = (0,255,0)
                lineType               = 2

                cv2.putText(frame, line.strip(), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
            line = fp.readline()
            cnt += 1
    #medbotlittle
    filepath = 'medbotlittle.txt'  
    with open(filepath) as fp:  
        line = fp.readline()
        lines = []
        while line:
            time, text = line.split(',')
            if float(time)*FRAMERATE <= i:
                lines.append(text)
            line = fp.readline()
        q = 0
        for line in lines:
            if 600 - (len(lines) - q)*30 > 0:
                font                   = cv2.FONT_HERSHEY_DUPLEX
                bottomLeftCornerOfText = (int(width) - 900, 600 - (len(lines) - q)*30)
                fontScale              = 1
                fontColor              = (0,255,0)
                lineType               = 2

                cv2.putText(frame, line.strip(), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
                q += 1
    #medbotbig
    filepath = 'medbotbig.txt'  
    with open(filepath) as fp:  
        line = fp.readline()
        lines = []
        while line:
            time, text = line.split(',')
            if float(time)*FRAMERATE <= i and float(time)*FRAMERATE > i - 1.5*FRAMERATE:
                lines.append(text)
            line = fp.readline()
        q = 0
        for line in lines:
            if 600 - (len(lines) - q)*30 > 0:
                font                   = cv2.FONT_HERSHEY_DUPLEX
                bottomLeftCornerOfText = (int(width/4), int(height/4))
                fontScale              = 5
                fontColor              = (0,255,0)
                lineType               = 2

                cv2.putText(frame, line.strip(), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
                q += 1

    #dialogue added on the HUD
    lines = []
    for line in dialogue:
        lines += [""]
        for time, word in line.items():
            if (float(time) + DIALOGUEDELAY)*FRAMERATE <= i:
                lines[-1] =  lines[-1] + " " + word

    q = 0
    for line in lines:
        if 600 - (len(lines) - q)*30 > 0:
            font                   = cv2.FONT_HERSHEY_DUPLEX
            bottomLeftCornerOfText = (int(width) - 900, 1200 - (len(lines) - q)*30)
            fontScale              = 1
            fontColor              = (0,255,0)
            lineType               = 2

            cv2.putText(frame, line.strip(), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
            q += 1

            

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0, 255), 5)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    
    # Display the resulting frame
    #cv2.imshow('Video', frame)
    #cv2.imwrite('transparent'+str(i)+'.png',frame)
    out.write(frame)
    if i % 30 == 0:
        print(i/FRAMERATE)
    i += 1



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # Display the resulting frame
        #cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
out.release()
clipto = mp.VideoFileClip('temp.mp4').set_audio(mp.VideoFileClip(INPUT).audio)
clipto.write_videofile(OUTPUT)
print("PROCESS COMPLETE")


#cv2.destroyAllWindows()
