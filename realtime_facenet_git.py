from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import datetime
import openpyxl
import xlsxwriter
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
import threading

#################list containing the data to be written in csv
names=['','Ankit','Garima','Nikita','Ritesh','Rupa','Samir']
inTime=[0,0,0,0,0,0,0]
remark=[0,0,0,0,0,0,0]
outTime=[0,0,0,0,0,0,0]



##############definition for appending to the list
def records(name):
    id=names.index(name)
    now = datetime.datetime.now()
        
    if(inTime[id]==0):
        inTime[id]=f'{now.hour}'+":"+f'{now.minute}'+":"+f'{now.second}'
        if(now.hour<9):
            remark[id]="On-Time"
        else:
            remark[id]="Half-Day"
    else:
        outTime[id]=f'{now.hour}'+":"+f'{now.minute}'+":"+f'{now.second}'
        
        
############final defition to append to the csv
def final():
       
       now=datetime.datetime.now()
       today=f'{now.day}'+"-"+f'{now.month}'+"-"+f'{now.year}'
       workbook = xlsxwriter.Workbook("attendance"+today+".xlsx")
       worksheet = workbook.add_worksheet()
       data = (
           ['Date:',today,'',''],
           ['Names','Time Of Entry','Remark','Exit Time'],
           
           )
       row = 0
       col = 0

# Iterate over the data and write it out row by row. 
       for x,y,i,j  in (data):
           worksheet.write(row, col,     x)
           worksheet.write(row, col + 1, y)
           worksheet.write(row, col + 2, i)
           worksheet.write(row, col + 3, j)
           row += 1
   
       for x in range(1,6):
           worksheet.write(row, col,     names[x])
           if(inTime[x]==0):
             worksheet.write(row, col + 1, "Absent")
           else:
             worksheet.write(row, col + 1, inTime[x])

           worksheet.write(row, col + 2, remark[x])
           worksheet.write(row, col + 3, outTime[x])
           row += 1
       workbook.close()   
       
       
       
def loadModel():
       
        
########################Loading model and prediction

    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, '')
    
            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 1
            batch_size = 1000
            image_size = 182
            input_image_size = 160
    
            HumanNames = ['Ankit','Garima','Nikita','Ritesh','Rupa','Samir']    #train human name
    
            print('Loading feature extraction model')
            modeldir = '20180402-114759\\'
            facenet.load_model(modeldir)
    
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
    
            classifier_filename = 'my_classifier.pkl'
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                print('load classifier file-> %s' % classifier_filename_exp)
    
            video_capture = cv2.VideoCapture(0)
            c = 0
    
            # #video writer
            # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            # out = cv2.VideoWriter('3F_0726.avi', fourcc, fps=30, frameSize=(640,480))
    
            print('Start Recognition!')
            prevTime = 0
            while True:
                ret, frame = video_capture.read()
    
                frame = cv2.resize(frame, (0,1), fx=1.2, fy=1.2)    #resize frame (optional)
    
                curTime = time.time()    # calc fps
                timeF = frame_interval
    
                if (c % timeF == 0):
                    find_results = []
    
                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('Detected_FaceNum: %d' % nrof_faces)
    
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]
    
                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces,4), dtype=np.int32)
    
                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))
    
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
    
                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('face is inner of range!')
                                continue
    
                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
    
                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            # print('result: ', best_class_indices[0])
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    
                                    cv2.putText(frame, result_names,(text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                                    t=threading.Thread(target=records,args=(result_names,))
                                    
                                    t.start()
                    else:
                        print('Unable to align')
    
                sec = curTime - prevTime
                prevTime = curTime
                fps = 1 / (sec)
                str = 'FPS: %2.3f' % fps
                text_fps_x = len(frame[0]) - 150
                text_fps_y = 20
                cv2.putText(frame, str, (text_fps_x, text_fps_y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
                # c+=1
                cv2.imshow('Video(Press q to QUIT)', frame)
                
                
                
     ##################final appending to the csv when its end time. .................
                now=  datetime.datetime.now()                  
                if(now.hour==18 and now.minute==0 and now.second==0):
        
                            print("appending")
                            final()
                            
                            
                            
    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
            video_capture.release()
            # #video writer
            # out.release()
            cv2.destroyAllWindows()
            
            
            


