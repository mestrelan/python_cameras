#Muda o lugar de referencia e agora pode importar os arquivos .py dessa pasta no caminho
import sys
sys.path.insert(0,r'C:\Users\Administrador.WIN-NG4VKK809R4\Music\Allan\CVSC')
sys.path.insert(0,r'C:\Users\Administrador.WIN-NG4VKK809R4\Music\Allan\CVSC\mrfd')

print(sys.path)

import cv2
import tensorflow
import keras

import os
import io
import numpy as np
import glob
import shutil
import datetime

import time
import traceback

from person_tracking import tracking_frames
from data_utils import *
import argparse
import config
from trainer.fusiondiffroigan import Params,Fusion_Diff_ROI_3DCAE_GAN3D
from models import diff_ROI_C3D_AE_no_pool

from models import diff_ROI_C3D_AE_no_pool,ROI_C3D_AE_no_pool,Fusion_C3D_no_pool
from trainer.fusiondiffroigan import Params,Fusion_Diff_ROI_3DCAE_GAN3D
from trainer.util import agg_window,create_windowed_arr,get_output,gather_auc_avg_per_tol,join_mean_std,create_diff_mask

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import tqdm

import sklearn.metrics
import math

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import confusion_matrix

import pandas as pd

import itertools

#Para salvar a saida em um arquivo txt e no console
class OutputTee(io.TextIOBase):
    def __init__(self, filename):
        self.stdout = sys.stdout
        self.file = open(filename, 'a')
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.stdout.flush()
        self.file.flush()

def get_traceback():
    traceback_output = io.StringIO()
    traceback.print_exc(file=traceback_output)
    return traceback_output.getvalue()

def salvar_saidas():
    if not os.path.exists("saidas"):
        os.mkdir("saidas")

    num_saida = 1
    while os.path.exists(f"saidas/saida{num_saida}.txt"):
        num_saida += 1

    nome_arquivo_saida = f"saidas/saida{num_saida}.txt"

    return OutputTee(nome_arquivo_saida)
#até aqui

def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == "opening":
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == "closing":
        kernel = np.ones((3, 3), np.uint8)

    return kernel

def getFilter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)

    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)

    if filter == 'dilation':
        return cv2.dilate(img, getKernel("dilation"), iterations=2)

    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)
        dilation = cv2.dilate(opening, getKernel("dilation"), iterations=2)

        return dilation

def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN(history=5000)
    if BGS_TYPE == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15*3, useHistory=True, maxPixelStability=15*60*3, isParallel=True)
    print("Detector invÃ¡lido")
    sys.exit(1)

def FMT(VIDEO_SOURCE,tipo="RGB"):
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    print('contagem de todos os frames do video')
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #print('gera 25 numeros aleatorios')
    #print(np.random.uniform(size=25))

    print('seleciona 25 frames aleatorios do video')
    framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
    #print(framesIds)
    
    #armazena os 25 frames em um array
    frames = []
    for fid in framesIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        hasFrame, frame = cap.read()
        frames.append(frame)

    print('numero de frames')
    print(np.asarray(frames).shape)
    #print('exemplos')
    #print(frames[0])
    #print(frames[1])
    
    #calculo da mediana de cada uma das imagens considerando todos os pixels (por causa do axis=0, vai por linha)
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    #print('imprime o primeiro frame de exemplo')
    #print(frame[0])
    #print('imprime a imagem de plano de fundo em forma de matriz')
    #print(medianFrame)
    #print('imprime a imagem de plano de fundo')
    #cv2_imshow(medianFrame)

    #cv2.imwrite('model_median_frame.jpg', medianFrame)

    #print('converte para cinza')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
    #cv2_imshow(grayMedianFrame)

    #RGB
    if tipo=="RGB":
        grayMedianFrame = grayMedianFrame[:,300:1750]
    #IR
    if tipo=="IR":
        grayMedianFrame = grayMedianFrame[10:410,50:450]
    
    grayMedianFrame = cv2.resize(grayMedianFrame, (640, 480))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return grayMedianFrame

def main(contador_frame_real,contador_frame,VIDEO_SOURCE,video_path_temp,tipo="RGB",filtro="0",video_out=True):
    #FMT
    grayMedianFrame=FMT(VIDEO_SOURCE,tipo)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    bg_subtractor = getBGSubtractor(BGS_TYPE)
    
    if video_out:
        VIDEO_OUT = os.path.join(video_path_temp,"video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (640, 480), False)
    
    timestamps=[]
    # count the number of frames
    #frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #fps = cap.get(cv2.CAP_PROP_FPS)
    #print(frames / fps)
    #seconds = round(frames / fps)
    #print(f"duration in seconds: {seconds}")
    
    #video_time = datetime.timedelta(seconds=seconds)
    #print(f"video time: {video_time}")
        
    while (cap.isOpened):

        ok, frame = cap.read()
        if not ok:
            print("Erro")
            break
        
        #em milisegundos
        #timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        #em segundos
        #timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)*10**-3)
        
        contador_frame_real+=1
        i=contador_frame_real
        #contador_frame+=1
        numero_de_copias1=2 #6
        numero_de_copias2=2  #3
        
        #IR
        if tipo=="IR":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[10:410,50:450]
        #RGB
        if tipo=="RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[:,300:1750]
        
        #frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
        frame = cv2.resize(frame, (640, 480))

        
        #result = cv2.bitwise_and(frame, frame, mask=bg_mask)
        #cv2.imshow('Frame', frame)
        #cv2.imshow('Mask', result)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2_imshow(frame)



        ########## FMT

        #faz a diferença do frame com o fundo
        bg_mask = cv2.absdiff(frame, grayMedianFrame)

        #redimenciona
        #dframe = cv2.resize(dframe, (0, 0), fx=0.40, fy=0.40)

        #threashold para ajustar e tirar o ruido 
        #th, dframe = cv2.threshold(dframe, 70, 255, cv2.THRESH_BINARY)
        ##th, dframe = cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #print(th)
        #######

        #####Filtro
        #bg_mask = bg_subtractor.apply(frame)
        #th, bg_mask = cv2.threshold(bg_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #bg_mask = getFilter(bg_mask, 'combine')
        #bg_mask = cv2.medianBlur(bg_mask, 5)
        ######

        filtro = str(filtro)
        if filtro=="0":
            pass
        else:
            for letra in filtro:
                if letra =="T":
                    th, bg_mask = cv2.threshold(bg_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    #th, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                if letra=="O":
                    bg_mask = getFilter(bg_mask, 'opening')
                    #frame = getFilter(frame, 'opening')
                if letra=="C":
                    bg_mask = getFilter(bg_mask, 'closing')
                    #frame = getFilter(frame, 'closing')
                if letra=="D":
                    bg_mask = getFilter(bg_mask, 'dilation')
                    #frame = getFilter(frame, 'dilation')
                if letra=="B":
                    bg_mask = cv2.medianBlur(bg_mask, 5)
                    #frame = cv2.medianBlur(frame, 5)

        ##### Final
        frame = cv2.bitwise_and(frame, frame, mask=bg_mask)
        #######

        #Depois fazer outro sistema fazer essa parte final, mas usando o dframe (so a mascara)
        
        #Faz o video

        

        try:
            writer.write(frame)
        except:
            pass

#        i = str(i).zfill(4)
#        #salva o frame no meu google drive
#        cv2.imwrite('/content/gdrive/MyDrive/Colab Notebooks/Allan Fall/Fall37/FALL_37-'+ i +'.jpg', frame)
#        i = int(i)

        if contador_frame_real < 12:
          #i = str(i).zfill(4)
              for contador_de_copias in range(0, numero_de_copias1):
                    contador_frame = contador_frame + 1
                    contador_frame = str(contador_frame).zfill(4)
                    ###print(contador_frame)
                    #salva o frame no meu google drive
                    cv2.imwrite(video_path_temp +'/FALL_37-'+ contador_frame +'.jpg', frame)
                    #cv2.imwrite('/content/FALL_37-'+ i +'.jpg', frame)
                    contador_frame = int(contador_frame)
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)*10**-3)
        elif (contador_frame_real >=12 and contador_frame_real<50):
              for contador_de_copias in range(0, numero_de_copias2): 
                    contador_frame = contador_frame + 1
                    contador_frame = str(contador_frame).zfill(4)
                    ###print(contador_frame)
                    #salva o frame no meu google drive
                    cv2.imwrite(video_path_temp+'/FALL_37-'+ contador_frame +'.jpg', frame)
                    #cv2.imwrite('/content/FALL_37-'+ i +'.jpg', frame)
                    contador_frame = int(contador_frame)
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)*10**-3)
        elif contador_frame_real >50:
              for contador_de_copias in range(0, numero_de_copias1):
                    contador_frame = contador_frame + 1
                    contador_frame = str(contador_frame).zfill(4)
                    ###print(contador_frame)
                    #salva o frame no meu google drive
                    cv2.imwrite(video_path_temp+'/FALL_37-'+ contador_frame +'.jpg', frame)
                    #cv2.imwrite('/content/FALL_37-'+ i +'.jpg', frame)
                    contador_frame = int(contador_frame)
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)*10**-3)
        

        #if (contador_frame_real%2==0):
            #frame_show = cv2.resize(frame, (320, 240))
          #cv2_imshow(frame_show)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    try:
        writer.release()
    except:
        pass
    cap.release()
    return timestamps


def criando_novo_teste(tipo="RGB",filtro="0"):
    pasta_pra_rodar = "/home/monitora/Documents/Motion and Region A A L F D  T I/dataset/Thermal/frame/Fall"
    
    if tipo == "RGB":
        source_folder = "/home/monitora/Documents/Motion and Region A A L F D  T I/Fall/RGB"
    if tipo == "IR":
        source_folder = "/home/monitora/Documents/Motion and Region A A L F D  T I/Fall"
    
    dst_folder_rodados =  os.path.join( source_folder,'Rodados')
    dst_folder_resultados = os.path.join( source_folder,'Resultados')
    dst_folder_fail = os.path.join( source_folder,'Fail')
    Resultados_dos_CVSC=os.path.join(source_folder,"Resultados_dos_CVSC")
    #print(os.listdir(source_folder))
    
    rodar=True
    for pasta in os.listdir(Resultados_dos_CVSC):
        if pasta=="CVSC FMT "+str(filtro):
            rodar=False
    if rodar==True:     
        for file in os.listdir(source_folder):
            if os.path.isfile(os.path.join(source_folder,file)):
                video_da_vez = file
                VIDEO_SOURCE = os.path.join(source_folder,video_da_vez)
                video_path_temp=os.path.join(source_folder,"Fall37")

                try:
                    if os.path.exists(video_path_temp):
                        shutil.rmtree(video_path_temp)
                        os.makedirs(video_path_temp)
                    if not os.path.exists(video_path_temp):
                        os.makedirs(video_path_temp)
                except OSError:
                    print ('Error: Creating directory of data')


                timestamps=main(0,0,VIDEO_SOURCE,video_path_temp,tipo,filtro)

                #pasta_pra_rodar = "/home/monitora/Documents/Motion and Region A A L F D  T I/dataset/Thermal/frame/Fall" 
                if os.path.exists(os.path.join(pasta_pra_rodar,"Fall37")):
                    shutil.rmtree(os.path.join(pasta_pra_rodar,"Fall37"))
                    shutil.move(video_path_temp, pasta_pra_rodar)

                if not os.path.exists(os.path.join(pasta_pra_rodar,"Fall37")):
                    shutil.move(video_path_temp, pasta_pra_rodar)

                nome_do_video = video_da_vez[:-4]

                return timestamps, nome_do_video,source_folder, pasta_pra_rodar
            
    nome_do_video=0
    timestamps=0  
    return timestamps, nome_do_video,source_folder, pasta_pra_rodar
        
def gray_color_image(gray):
    gray_scaled=np.expand_dims(cv2.normalize(gray,None,0,1,cv2.NORM_MINMAX),axis=-1)
    gray_scaled=gray_scaled*255
    gray_scaled=gray_scaled.astype(np.uint8)
    org_color=np.concatenate([gray_scaled,gray_scaled,gray_scaled],axis=-1)
    return org_color
def roi_gray_color_image(roi_gray,box_fr):
    height,width=roi_gray.shape[0],roi_gray.shape[1]
    color_img=np.zeros((height,width,3),dtype='uint8')
    left, top, right, bottom=int(box[1]*width),int(box[0]*height),int(box[3]*width),int(box[2]*height)
    color_img[top:bottom,left:right,:]=gray_color_image(roi_gray[top:bottom,left:right,:])
    return color_img
    

def get_cross_window_frames(recons_seq,height,width,channels,win_length):
    '''
        Take mean of the reconstructed frames present in different windows corresponding to the actual frame timestamp
    '''
    seq_num=recons_seq.shape[0]+win_length-1
    sum_frames=np.zeros((seq_num,height,width,channels),dtype='float')
    count_frames=np.zeros((seq_num))
    for i in range(recons_seq.shape[0]):
        sum_frames[i:i+win_length,:]+=recons_seq[i,:]
        count_frames[i:i+win_length]+=1
    return sum_frames/count_frames[:, np.newaxis, np.newaxis, np.newaxis]

'''
This function is the extension of function animate_fall_detect_Spresen() from https://github.com/JJN123/Fall-Detection/blob/master/util.py

'''
def animate_fall_detect_animation(actual_frames, recons,recons_timestamp, scores,score_type='RE_mean',threshold = 0,to_save = './test.mp4'):
    '''
    Create animation from actual frames, reconstructed frames and frame level anomaly score with timestamps
    '''
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2,2,height_ratios = [2,1])
    
    ht, wd = 64,64

    eps = .0001
    #setup figure
    #fig = plt.figure()
    fig, ((ax1,ax3)) = plt.subplots(1,2,figsize = (6,6))

    ax1.axis('off')
    ax3.axis('off')
    #ax1=fig.add_subplot(2,2,1)

    ax1=fig.add_subplot(gs[0,0])
    ax1.set_title("Original")
    ax1.set_xticks([])
    ax1.set_yticks([])


    #ax2=fig.add_subplot(gs[-1,0])
    ax2=fig.add_subplot(gs[1,:])

    #ax2.set_yticks([])
    #ax2.set_xticks([])
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Frame')
    ax2.set_xlim([1, len(actual_frames)])
    if threshold != 0:
        ax2.axhline(y= threshold, color='r', linestyle='dashed', label = 'RRE')
        ax2.legend()

    #ax3=fig.add_subplot(2,2,2)
    ax3=fig.add_subplot(gs[0,1])
    ax3.set_title("Reconstruction")
    ax3.set_xticks([])
    ax3.set_yticks([])

    #dictionary to frame number to indices
    indices=list(range(len(recons_timestamp)))
    track_indices=dict(zip(recons_timestamp,indices))
    #set up list of images for animation
    ims=[]
    track_ind=None
    for time in tqdm(range(len(actual_frames))):
        #plot images
        im1 = ax1.imshow(actual_frames[time])
        figure= recons[time]
        im2 = ax3.imshow(figure, cmap = 'gray', aspect = 'equal')
        
        if time+1 in track_indices:
            track_ind=track_indices[time+1]
            
        if track_ind is not None:
            scores_curr = scores[:track_ind+1]
            track_num=recons_timestamp[:track_ind+1]
            
            plot, = ax2.plot(track_num, scores_curr,'b.',linestyle='--', marker='.',label=score_type)
        else:
            plot, = ax2.plot([],'b.')
#             plot_r, = ax2.plot([],'b.')
            
            
    

        ims.append([im1, plot, im2]) #list of ims

    #run animation
    ani = animation.ArtistAnimation(fig,ims, interval= 30, repeat=False)
    
    ani.save(to_save)

    ani.event_source.stop()
    del ani
    plt.close()
#     plt.show()


def cvsc(animation_out=True):
    ####Timer
    inicio=time.time()
    #user input
    detection_threshold=0.3
    folder_path=r'C:\Users\Administrador.WIN-NG4VKK809R4\Music\Allan\CVSC\dataset\Thermal\frame\Fall\Fall37'
    
    #parameters
    WIDTH,HEIGHT=config.WIDTH,config.HEIGHT#original data specs
    win_length=config.WIN_LENGTH
    LOAD_DATA_SHAPE=config.LOAD_DATA_SHAPE
    width, height,channels = LOAD_DATA_SHAPE[0],LOAD_DATA_SHAPE[1],LOAD_DATA_SHAPE[2]
    break_win=config.SPLIT_GAP
    stride=config.STRIDE
    
    input_folder=folder_path
    #reading and sorting image paths
    frames_path = glob.glob(input_folder+'/*.jpg') + glob.glob(input_folder+'/*.png')
    frames_path,numbers = sort_frames(frames_path,'Thermal')

    #person tracking
    boxes,track_numbers=tracking_frames(detection_threshold,frames_path,numbers,otsu_box=True)

    ####Timer
    fim=time.time()
    ROI_M_time = fim - inicio
    inicio=time.time()

    # box_path='box.npy'
    # numbers_path='numbers.npy'
    # if os.path.exists(box_path) and  os.path.exists(numbers_path):
    #     boxes=np.load(box_path)
    #     track_numbers=np.load(numbers_path)
    # else:
    #     np.save(box_path,boxes)
    #     np.save(numbers_path,track_numbers)
    
    #preprocess boxes- remove -ve coordinates
    boxes_proc=np.array([improve_box_cord(box,WIDTH,HEIGHT,offset=10) for box in boxes])
    #creating dictionary with key:frame_num value:box with coordinates are scaled in range 0 to 1
    boxes_fr=boxes_proc.copy()
    boxes_fr=boxes_fr.astype('float64')
    boxes_fr[:,0]=boxes_fr[:,0]/(1.0*HEIGHT)
    boxes_fr[:,2]=boxes_fr[:,2]/(1.0*HEIGHT)
    boxes_fr[:,1]=boxes_fr[:,1]/(1.0*WIDTH)
    boxes_fr[:,3]=boxes_fr[:,3]/(1.0*WIDTH)
    num_box_dict=dict(zip(track_numbers,boxes_fr))

    print("Thermal preprocessing....")
    video={}

    #preprocessing all frames
    video["ALL_FRAME"],_,_=preprocess_frames(frames_path,numbers,process_list=['Processed'],ht=height,wd=width,channels=channels,ROI_array=None)
    
    tracked_frames_path=[]

    for num in track_numbers:
        tracked_frames_path.append(frames_path[num-1])
    #preprocessing tracked frames
    #Data as numpy array and list of sorted frame numbers
    data,frame_numbers,frames_path=preprocess_frames(tracked_frames_path,track_numbers,process_list=['Processed','ROI_frame'],ht=height,wd=width,channels=channels,ROI_array=boxes_proc)
    #creating sub vidoes
    data_list,frame_numbers_list=split_data_tracks(data,frame_numbers,gap=break_win,win_length=win_length)
    #Split frames path
    frames_path_list,_=split_data_tracks(frames_path,frame_numbers,gap=break_win,win_length=win_length)
    video['ROI_FRAME']=data_list
    video['NUMBER']=frame_numbers_list
    video['PATH']=frames_path_list
    print("\nCreating MASK data...........\n")
    video['MASK']=create_ROI_mask(ROI_boxes=boxes_proc,ROI_numbers=track_numbers,img_shape=(config.HEIGHT,config.WIDTH,1),load_shape=config.LOAD_DATA_SHAPE,win_length=config.WIN_LENGTH,split_gap=config.SPLIT_GAP)
    #optical flow computation

    ####Timer
    fim=time.time()
    OFC_time = fim - inicio
    inicio=time.time()

    #image sample
    sub_video_num=0
    index=10
    frame_num=video['NUMBER'][sub_video_num][index]
    
    org_frame=video["ALL_FRAME"][frame_num-1]#1 numbering
    roi_frame=video['ROI_FRAME'][sub_video_num][index]
    mask=video['MASK'][sub_video_num][index]
    box=num_box_dict[frame_num]
    
    left, top, right, bottom=int(box[1]*width),int(box[0]*height),int(box[3]*width),int(box[2]*height)
    #plt.imshow(cv2.rectangle(gray_color_image(org_frame), (left, top), (right, bottom), (0, 0, 255),1))
    #plt.imshow(mask[:,:,0],cmap='gray')
    #plt.imshow(roi_gray_color_image(roi_frame,box))
    
    #parameters
    dset = config.track_root_folder
    d_type='ROI_Fusion'
    thermal_channels=1
    flow_channels=3
    regularizer_list = ['BN']
    epochs_trained=299
    lambdas=[1.0,1.0,1.0]#T_S,T_T,F
    thermal_3dcae_path=r'C:\Users\Administrador.WIN-NG4VKK809R4\Music\Allan\CVSC\mrfd\Thermal_track\ROI_Fusion\ROI_C3DAE-no_pool-BN_diff_ROI_C3DAE_no_pool-BN_Fusion_C3D-no_pool-BN\lambda_TS1.0_TT1.0_F1.0\models\GAN_T_R_weights_epoch-299.h5'

    param=Params(width=width, height=height,win_length=win_length,thermal_channels=thermal_channels,flow_channels=flow_channels \
             ,dset=dset,d_type=d_type,regularizer_list=regularizer_list,break_win=break_win)
    param.thermal_lambda_S=lambdas[0]
    param.thermal_lambda_T=lambdas[1]
    param.flow_lambda=lambdas[2]

    #trainer
    GAN3D=Fusion_Diff_ROI_3DCAE_GAN3D(train_par=param,stride=stride)
    #thermal reconstructor model 
    #initialization
    TR, TR_name, _ = diff_ROI_C3D_AE_no_pool(img_width=param.width, img_height=param.height, win_length=param.win_length, regularizer_list=param.regularizer_list,channels=param.thermal_channels,lambda_S=param.thermal_lambda_S,lambda_T=param.thermal_lambda_T,d_type='thermal')

    #Loading weights
    if os.path.isfile(thermal_3dcae_path):
        TR.load_weights(thermal_3dcae_path)
        GAN3D.T_R=TR
        print("Model weights loaded successfully........")
    else:
        print("Saved model weights not found......")
        
    ##### Sliding window
    vid_thermal_list=video['ROI_FRAME']
    vid_thermal_mask_list=video['MASK']
    frame_numbers_cat=np.concatenate(video['NUMBER'])
    
    #creating windows of thermal frames for each subvideo separately
    thermal_data_list = [vid.reshape(len(vid), param.width,param.height, param.thermal_channels) for vid in vid_thermal_list]
    thermal_data_windowed_list = [create_windowed_arr(test_data, stride, param.win_length) for test_data in thermal_data_list]#create_windowe

    # creating windows of mask data
    thermal_mask_list = [vid.reshape(len(vid), param.width,param.height, param.thermal_channels) for vid in vid_thermal_mask_list]

    thermal_mask_windowed_list = [create_windowed_arr(test_data, stride, param.win_length).astype('int8') for test_data in thermal_mask_list]
    # creating windows of mask of difference frames
    diff_mask_windowed_list=[create_diff_mask(mask_windows) for mask_windows in thermal_mask_windowed_list]
    
    num_sub_videos=len(thermal_data_windowed_list)
    
    #Model prediction, frame level anomaly scores and thermal reconstruction
    
    #frame based anomaly scores
    x_std_RE=[]
    x_mean_RE=[]
    mean_frames=[]
    for index in range(num_sub_videos):
        test_data_masked_windowed=thermal_data_windowed_list[index]
        test_mask_windowed=thermal_mask_windowed_list[index]
        test_diff_mask_windowed=diff_mask_windowed_list[index]

        RE_dict, recons_seq = GAN3D.get_T_S_RE_all_agg(thermal_data=test_data_masked_windowed,thermal_masks=test_mask_windowed,diff_masks=test_diff_mask_windowed) #Return dict with value for each score style
        x_std_RE.append(RE_dict['x_std'])
        x_mean_RE.append(RE_dict['x_mean'])
        mean_recons_seq=get_cross_window_frames(recons_seq,param.height,param.width, param.thermal_channels,param.win_length)
        mean_frames.append(mean_recons_seq)
    

    x_std_RE=np.concatenate(x_std_RE)
    x_mean_RE=np.concatenate(x_mean_RE)
    mean_frames=np.concatenate(mean_frames)
    print(mean_frames.shape)
    print(len(frame_numbers_cat))
    
    #try:
    #    plt.imshow(mean_frames[50,:,:,0],cmap='gray')
    #except:
    #    pass
    
    
    '''#para nao mais mostrar os graficos dos videos
    #Anomaly score plot
    plt.plot(frame_numbers_cat,x_std_RE, label='RE_std',linestyle='--', marker='.')
    plt.plot(frame_numbers_cat,x_mean_RE, label='RE_mean',linestyle='--', marker='.')
    # plt.xticks([i+1 for i in range(max(frame_numbers))])
    plt.xlim(1,max(frame_numbers_cat))
    # plt.ylim(0,1)
    plt.legend()
    # plt.axvspan(start,end, alpha = 0.5)
    plt.show()
    '''
    ############  animation  #####################
    #dictionary tracked frames number to index
    indices=list(range(len(frame_numbers_cat)))
    track_indices=dict(zip(frame_numbers_cat,indices))
    
    #convert gray to rgb, add boxes to track frames
    actual_frames=video["ALL_FRAME"]
    org_color_images=[]
    recon_color_images=[]
    #lan
    frame_num = []
    lan_box = []
    
    
    for i in range(len(actual_frames)):
        frame_num.append(i+1)
        if i+1 in track_indices:
            box=num_box_dict[i+1]
            left, top, right, bottom=int(box[1]*width),int(box[0]*height),int(box[3]*width),int(box[2]*height)
            lan_box.append([left, top, right, bottom])
            #add box in org frame
            clr_img=cv2.rectangle(gray_color_image(actual_frames[i]), (left, top), (right, bottom), (0, 0, 255),1)
            org_color_images.append(clr_img)
            #recons image -> color img
            recon_im=gray_color_image(mean_frames[track_indices[i+1]])
            clr_img=cv2.rectangle(recon_im, (left, top), (right, bottom), (0, 0, 255),1)
            recon_color_images.append(clr_img)
        else:
            lan_box.append([])
            org_color_images.append(gray_color_image(actual_frames[i]))
            recon_color_images.append(np.zeros((height,width,3),dtype='uint8'))
            
    print(len(org_color_images))
    print(len(recon_color_images))
    
    #exemplo de um frame
    #try:
    #    index=500
    #    plt.imshow(org_color_images[index])
    #    plt.show()
    #    plt.imshow(recon_color_images[index])
    #    plt.show()
    #except:
    #    pass
    
    if animation_out:
        demo_samples_path=folder_path
        #os.makedirs(demo_samples_path,exist_ok=True)
    
        #user_input
        score_type='mean'
        video_name='animation'
        save_path=demo_samples_path+'/'+video_name+'_'+score_type+'.mp4'
    
        #animate_fall_detect_animation(org_color_images,recon_color_images,frame_numbers_cat, scores=x_mean_RE,score_type='RE_'+score_type,to_save = save_path)
        
    #new
    scores=[]
    falls=[]
    
    cont_list_menor=0
    for num in frame_num:
        if num in frame_numbers_cat:
            scores.append(x_mean_RE[cont_list_menor])
            cont_list_menor+=1
        else:
            scores.append(None)
            
    for score in scores:
        try:
            if score >= 0.012:
                fall=1
            else:
                fall=0
        except:
            #None: sem score
            fall=0
        falls.append(fall)
    
    print("LISTAS"
          +"\n falls: "+str(len(falls))
         +" frame_num: "+str(len(frame_num))
         +" lan_box: "+str(len(lan_box))
         +" scores: "+str(len(scores)))
    #print(falls)
    #print(frame_num)
    #print(lan_box)
    #print(scores)

    ####Timer
    fim=time.time()
    GAN_time = fim - inicio
    timers_CVSC=[ROI_M_time,OFC_time,GAN_time]

    return falls,frame_num,lan_box,scores,timers_CVSC

    

def matrix_metrix(real_values,pred_values,beta):
    CM = sklearn.metrics.confusion_matrix(real_values,pred_values)
    TN = CM[0][0]
    FN = CM[1][0] 
    TP = CM[1][1]
    FP = CM[0][1]
    Population = TN+FN+TP+FP
    Prevalence = round( (TP+FN) / Population,2)
    Accuracy   = round( (TP+TN) / Population,4)
    Precision  = round( TP / (TP+FP),4 )
    NPV        = round( TN / (TN+FN),4 )
    FDR        = round( FP / (TP+FP),4 )
    FOR        = round( FN / (TN+FN),4 ) 
    check_Pos  = Precision + FDR
    check_Neg  = NPV + FOR
    Recall     = round( TP / (TP+FN),4 )
    FPR        = round( FP / (TN+FP),4 )
    FNR        = round( FN / (TP+FN),4 )
    TNR        = round( TN / (TN+FP),4 ) 
    check_Pos2 = Recall + FNR
    check_Neg2 = FPR + TNR
    LRPos      = round( Recall/FPR,4 ) 
    LRNeg      = round( FNR / TNR ,4 )
    try:
        DOR      = round( LRPos/LRNeg,4)
    except:
        DOR      = None
    F1         = round ( 2 * ((Precision*Recall)/(Precision+Recall)),4)
    FBeta      = round ( (1+beta**2)*((Precision*Recall)/((beta**2 * Precision)+ Recall)) ,4)
    MCC        = round ( ((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))  ,4)
    BM         = Recall+TNR-1
    MK         = Precision+NPV-1
    FM         = round ( math.sqrt(Precision*Recall)  ,4)
    try:
        PT         = round ( math.sqrt(FPR) / (math.sqrt(Recall)+math.sqrt(FPR))  ,4)
    except:
        PT         = None
    TS         = round ( TP / (TP+FN+FP),4 )
    BA         = round( (Recall +TNR) / 2 ,4 )
    Hamming    = round( sklearn.metrics.hamming_loss(real_values,pred_values) ,4)
    Jaccard    = round( sklearn.metrics.jaccard_score(real_values,pred_values) ,4)
    AUC        = round(roc_auc_score(real_values,pred_values),4)

  
    mat_met = pd.DataFrame({
        'Metric':['Population','TP','TN','FP','FN','Prevalence','Accuracy','Precision','NPV','FDR','FOR','check_Pos','check_Neg','Recall','FPR','FNR','TNR','check_Pos2','check_Neg2','LR+','LR-','DOR','F1','FBeta','MCC','BM','MK','FM','PT','TS','BA','Ham','Jac(TS)','AUC'],
        'Value':[Population, TP,TN,FP,FN,Prevalence,Accuracy,Precision,NPV,FDR,FOR,check_Pos,check_Neg,Recall,FPR,FNR,TNR,check_Pos2,check_Neg2,LRPos,LRNeg,DOR,F1,FBeta,MCC,BM,MK,FM,PT,TS,BA,Hamming,Jaccard,AUC]})
    return (mat_met)


def criar_tabela_video(tipo="RGB",filtro="0"):
    inicio=time.time()
    timestamps,nome_do_video,source_folder,pasta_pra_rodar=criando_novo_teste(tipo,filtro)
    fim = time.time()
    pre_proc_time = fim - inicio
    if nome_do_video==0:
        #acabou videos na pasta
        df=0
        file_name=0
        return df,file_name,nome_do_video,source_folder,pasta_pra_rodar
    else:
        inicio=time.time()
        try:
            print("CVSC START")
            falls,frame_num,lan_box,scores,timers_CVSC = cvsc()
            print("CVSC DONE!")
            df=pd.DataFrame({"frame N.":frame_num,
                             "time":timestamps,
                             "ROI":lan_box,
                             "Score Mean":scores,
                             "Fall":falls})
            print("DF Created")
        except Exception as erro:
            print("CVSC FAIL", erro)
            print("Rastreamento:")
            #traceback.print_exc()
            traceback_str = get_traceback()
            print(traceback_str)
            print("Fim do Rastreamento.")
            timers_CVSC=[0,0,0]
            fail = ["fail"]
            df=pd.DataFrame({"frame N.":fail,
                             "time":fail,
                             "ROI":fail,
                             "Score Mean":fail,
                             "Fall":fail})
        fim = time.time()
        CVSC_time = fim - inicio
        Total_time=pre_proc_time+CVSC_time
        df_time=pd.DataFrame({"Image pre-processing":pre_proc_time,
                              "ROI-M":timers_CVSC[0],
                              "OFC":timers_CVSC[1],
                              "GAN":timers_CVSC[2],
                            "CVSC time":CVSC_time,
                            "Total time":[Total_time]})
        video_path_temp=os.path.join(pasta_pra_rodar,"Fall37")
        file_name=os.path.join(video_path_temp,nome_do_video+".csv")
        file_name_time=os.path.join(video_path_temp,nome_do_video+" time.csv")
        df.to_csv(file_name)
        df_time.to_csv(file_name_time)
        return df,file_name,nome_do_video,source_folder,pasta_pra_rodar


#funçao para criar tabela do modelo
def criar_tabela_modelo(source_folder):
    Resultados_dos_CVSC=os.path.join(source_folder,"Resultados_dos_CVSC")
    for pasta_do_modelo in os.listdir(Resultados_dos_CVSC):
        if os.path.isdir(os.path.join(Resultados_dos_CVSC,pasta_do_modelo)):
        #if pasta_do_modelo != "metricas.csv":    
            caminho_pasta_do_modelo=os.path.join(Resultados_dos_CVSC,pasta_do_modelo)
            fail = ["fail"]
            df_do_modelo=pd.DataFrame({"Nome do video":fail,
                                      "Tem queda?":fail,
                                      "Fall detection":fail,
                                        "Image pre-processing":fail,
                                       "ROI-M":fail,
                                       "OFC":fail,
                                       "GAN":fail,
                                        "CVSC time":fail,
                                        "Total time":fail})
            for pasta_do_video in os.listdir(caminho_pasta_do_modelo):
                caminho_pasta_do_video=os.path.join(caminho_pasta_do_modelo,pasta_do_video)
                path_tabela_video=os.path.join(caminho_pasta_do_video,pasta_do_video+".csv")
                path_tabela_time=os.path.join(caminho_pasta_do_video,pasta_do_video+" time.csv")
                if os.path.exists(path_tabela_video):
                    nova_linha=[pasta_do_video]
                    if pasta_do_video[18:20]=="43":
                        #43 para queda e 01 para ADL
                        nova_linha.append(1)
                    if pasta_do_video[18:20]=="01":
                        nova_linha.append(0)

                    df_video=pd.read_csv(path_tabela_video)
                    if df_video["frame N."][0]=="fail":
                        nova_linha.append(0)
                    else:
                        resul=0
                        for falls in df_video["Fall"]:
                            if falls == 1:
                                resul=1
                                break
                        nova_linha.append(resul)
                    if os.path.exists(path_tabela_time):
                        df_time=pd.read_csv(path_tabela_time)
                        nova_linha.append(df_time["Image pre-processing"][0])
                        nova_linha.append(df_time["ROI-M"][0])
                        nova_linha.append(df_time["OFC"][0])
                        nova_linha.append(df_time["GAN"][0])
                        nova_linha.append(df_time["CVSC time"][0])
                        nova_linha.append(df_time["Total time"][0])
                    if df_do_modelo["Nome do video"][0]=="fail":
                        df_do_modelo.loc[0]=nova_linha
                    else:
                        df_do_modelo.loc[len(df_do_modelo)]=nova_linha
            file_name=os.path.join(caminho_pasta_do_modelo,pasta_do_modelo+".csv")
            if not os.path.isfile(file_name):
                df_do_modelo.to_csv(file_name)
                print("df do modelo criado:"+str(pasta_do_modelo))

def pos_execucao(file_name,nome_do_video,source_folder,pasta_pra_rodar,filtro="0",reset_dataset=False):
    Rodados =  os.path.join(source_folder,'Rodados')
    Resultados = os.path.join(source_folder,"Resultados")
    Fail = os.path.join(source_folder,"Fail")
    VIDEO_SOURCE = os.path.join(source_folder,str(nome_do_video)+".avi")
    if nome_do_video !=0:
        df_video=pd.read_csv(file_name)
        if df_video["frame N."][0]!="fail":
            #move resultados rodados para pasta Resultados
            shutil.move(os.path.join(pasta_pra_rodar,"Fall37"), Resultados)
            #renomeia
            shutil.move(os.path.join(Resultados,"Fall37"), os.path.join(Resultados,nome_do_video))
            #move video que rodou
            shutil.move(VIDEO_SOURCE, Rodados)

        if df_video["frame N."][0]=="fail":
            #move resultados para pasta Resultados
            shutil.move(os.path.join(pasta_pra_rodar,"Fall37"), Resultados)
            #renomeia
            shutil.move(os.path.join(Resultados,"Fall37"), os.path.join(Resultados,nome_do_video))
            #mover video que nao rodou pra fail
            shutil.move(VIDEO_SOURCE, Fail)
    
    if reset_dataset==True:
        path_nova_pasta_modelo=os.path.join(source_folder,"CVSC FMT "+str(filtro))
        #Renomear pasta Resultados
        shutil.move(Resultados,path_nova_pasta_modelo)
        #cria uma pasta geral para armazenar todos os resultados dos modelos
        Resultados_dos_CVSC=os.path.join(source_folder,"Resultados_dos_CVSC")
        if not os.path.exists(Resultados_dos_CVSC):
            os.makedirs(Resultados_dos_CVSC, exist_ok = True)
        #Verifica se a do ultimo modelo rodado ja existe na pasta geral de resultados
        if os.path.exists(os.path.join(Resultados_dos_CVSC,"CVSC FMT "+str(filtro))):
            #Se ela ja existe la, pode apagar
            shutil.rmtree(path_nova_pasta_modelo)
        if not os.path.exists(os.path.join(Resultados_dos_CVSC,"CVSC FMT "+str(filtro))):
            #move para essa pasta geral do ultimo modelo rodado
            shutil.move(path_nova_pasta_modelo,Resultados_dos_CVSC)
        #cria outra pasta resultados
        os.makedirs(Resultados)
        #move os arquivos em Fail para a pasta source_folder
        for file in os.listdir(Fail):
            shutil.move(os.path.join(Fail,file),source_folder)
        #move os arquivos em Rodados para a pasta source_folder
        for file in os.listdir(Rodados):
            shutil.move(os.path.join(Rodados,file),source_folder)

        

#funçao para criar tabela com as metricas pra cada modelo
def metricas(source_folder):
    Resultados_dos_CVSC=os.path.join(source_folder,"Resultados_dos_CVSC")
    beta=0.6
    df_metricas_geral=pd.DataFrame(columns=['Model name','Population','TP','TN','FP','FN','Prevalence','Accuracy','Precision','NPV','FDR','FOR','check_Pos','check_Neg','Recall','FPR','FNR','TNR','check_Pos2','check_Neg2','LR+','LR-','DOR','F1','FBeta','MCC','BM','MK','FM','PT','TS','BA','Ham','Jac(TS)','AUC', 'Image pre-processing(s)','ROI-M(s)','OFC(s)','GAN(s)','CVSC time(s)','Total time(s)'])
        
    for pasta_do_modelo in os.listdir(Resultados_dos_CVSC):
        if os.path.isdir(os.path.join(Resultados_dos_CVSC,pasta_do_modelo)):
        #if pasta_do_modelo != "metricas.csv": 
            caminho_pasta_do_modelo=os.path.join(Resultados_dos_CVSC,pasta_do_modelo)
            path_df_modelo=os.path.join(caminho_pasta_do_modelo,pasta_do_modelo+".csv")
            if os.path.exists(path_df_modelo):
                df_modelo=pd.read_csv(path_df_modelo)
                real_values=list(df_modelo["Tem queda?"])
                pred_values=list(df_modelo["Fall detection"])
                df_metricas_modelo=matrix_metrix(real_values,pred_values,beta)
                nova_linha=[pasta_do_modelo]
                for metrica in df_metricas_modelo['Value']:
                    nova_linha.append(metrica)
                
                time_total=0
                count=0
                for times in df_modelo['Image pre-processing']:
                    time_total= time_total+times
                    count=count+1
                nova_linha.append(time_total/count)
                
                time_total=0
                count=0
                for times in df_modelo['ROI-M']:
                    time_total= time_total+times
                    count=count+1
                nova_linha.append(time_total/count)
                
                time_total=0
                count=0
                for times in df_modelo['OFC']:
                    time_total= time_total+times
                    count=count+1
                nova_linha.append(time_total/count)
                
                time_total=0
                count=0
                for times in df_modelo['GAN']:
                    time_total= time_total+times
                    count=count+1
                nova_linha.append(time_total/count)
                
                time_total=0
                count=0
                for times in df_modelo['CVSC time']:
                    time_total= time_total+times
                    count=count+1
                nova_linha.append(time_total/count)
                time_total=0
                count=0
                for times in df_modelo['Total time']:
                    time_total= time_total+times
                    count=count+1
                nova_linha.append(time_total/count)

                df_metricas_geral.loc[len(df_metricas_geral)]=nova_linha

    file_name=os.path.join(Resultados_dos_CVSC,"metricas.csv")
    if not os.path.isfile(file_name):
        df_metricas_geral.to_csv(file_name)
    if os.path.isfile(file_name):
        os.remove(file_name)
        df_metricas_geral.to_csv(file_name)


def rodar_videos_pasta(tipo="RGB",filtro="0"):
    rodar_videos_na_pasta=1
    while rodar_videos_na_pasta == 1:
        df,file_name,nome_do_video,source_folder,pasta_pra_rodar=criar_tabela_video(tipo,filtro)
        if nome_do_video==0:
            #acabou videos na pasta
            rodar_videos_na_pasta=0
            pos_execucao(file_name,nome_do_video,source_folder,pasta_pra_rodar,filtro,reset_dataset=True)
            criar_tabela_modelo(source_folder)
            metricas(source_folder)
            
        else:
            #ainda tem video na pasta
            pos_execucao(file_name,nome_do_video,source_folder,pasta_pra_rodar,filtro,reset_dataset=False)




# função para rodar os filtros em RGB e IR
    #BGS_TYPES = ["FMT", "GMG", "MOG", "MOG2", "KNN", "CNT"]
    #BGS_TYPE = BGS_TYPES[3]
        

def rodar_filtros():
    tipos = ["RGB", "IR"]
    #BGS_TYPES = ["FMT", "CNT", "MOG2"]
    #BGS_TYPE = BGS_TYPES[3]
    filtros="CODBT" #em ingles
    filtros= list(filtros)
    #permutations = list(itertools.permutations(filtros))
    combinations=["0"]
    for num,filtro in enumerate(filtros):
        combinations.extend(list(itertools.combinations(filtros,num+1)))
    print(combinations)
    ######Remover filtros já rodados
    #print(combinations)
    #combinations.pop(1)
    #print(combinations)
    #####
    inicio3=time.time()
    for tipo in tipos:
        inicio2=time.time()
        for filtro in combinations:
            inicio=time.time()
            #bloco de codigo
            #filtro=''.join(filtro)  #descomenta para que o nome dos filtros nao aparecer mais como tuple
            rodar_videos_pasta(tipo,filtro)
            fim = time.time()
            print("Tipo: "+tipo+"Filtro: "+str(filtro))
            print ('duracao: %f' % (fim - inicio))
        print("fim do tipo: "+tipo)
        fim2 = time.time()
        print ('duracao dos testes no modelo: %f' % (fim2 - inicio2))
    fim3 = time.time()
    print("fim dos testes")
    print ('duracao: %f' % (fim3 - inicio3))


####Execução

TEXT_COLOR = (0, 255, 0)
TRACKER_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
BGS_TYPE = BGS_TYPES[3]
minArea = 500
maxArea= 15000

# Redireciona a saída para o arquivo
saida_arquivo = salvar_saidas()
# Todas as saídas a partir daqui serão redirecionadas para o arquivo e o console

rodar_filtros()

