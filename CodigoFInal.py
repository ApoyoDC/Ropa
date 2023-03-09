# %%
import pandas as pd
import datetime
import torch
import numpy as np
import cv2
from time import time

from tkinter import *
from PIL import Image
from PIL import ImageTk
import imutils
Log= pd.read_excel('Log.xlsx')

# %%


def SaveLog(imagen):
    Log= pd.read_excel('Log.xlsx')
    df2 = {'Fecha': str(datetime.datetime.now()), 'Imagen': imagen}
    Log = Log.append(df2, ignore_index = True)
    Log.to_excel('Log.xlsx',index=False)
def get_video_capture():
    """
    Creates a new video streaming object to extract video frame by frame to make prediction on.
    :return: opencv2 video capture object, with lowest quality frame available for video.
    """
    
    return cv2.VideoCapture(capture_index)

def load_model(model_name):
    """
    Loads Yolo5 model from pytorch hub.
    :return: Trained Pytorch model.
    """
    if model_name:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def score_frame( frame):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
    """
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

def class_to_label(x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """

    return classes[int(x)]

def plot_boxes(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    
    return frame

def __call__():
    """
    This function is called when class is executed, it runs the loop to read the video frame by frame,
    and write the output into a new file.
    :return: void
    """
    cap = get_video_capture()
    assert cap.isOpened()
    
    while True:
        
        ret, frame = cap.read()
        assert ret
        
        frame = cv2.resize(frame, (416,416))
        
        start_time = time()
        results = score_frame(frame)
        frame = plot_boxes(results, frame)
        
        end_time = time()
        fps = 1/np.round(end_time - start_time, 2)
        #print(f"Frames Per Second : {fps}")
            
        cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        
        cv2.imshow('YOLOv5 Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
def iniciar():
    global cap
    global contChaleco
    global contCasco
    global cont
    contChaleco=0
    contCasco=0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 0 = WebCam
    cont=len(Log)
    visualizar()
def visualizar():
    global cap
    global contChaleco
    global contCasco
    global cont
    aprobado=False
    assert cap.isOpened()
    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
            frame = imutils.resize(frame, width=640)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start_time = time()
            results = score_frame(frame)
            frame = plot_boxes(results, frame)

            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            #print(f"Frames Per Second : {fps}")
                
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)
            lblVideo.configure(image=img)
            lblVideo.image = img
            if len(results[0])==2:
                if (class_to_label(results[0][0])=='chaleco' and class_to_label(results[0][1])=='casco') or (class_to_label(results[0][0])=='casco' and class_to_label(results[0][1])=='chaleco'):
                    contChaleco+=1
                    contCasco+=1
            if contChaleco>4 and contCasco>4:
                contChaleco=0
                contCasco=0
                print('si se pudo')
                image = ImageTk.PhotoImage(file = "Images\Cheque.png")
                imageLabel.configure(image = image)
                imageLabel.image = image
                cont+=1
                cv2.imwrite('RegistroFotos/'+'empleado'+str(cont)+'.png',frame)
                SaveLog('RegistroFotos/'+'empleado'+str(cont)+'.png')
                aprobado=True
                
            else:
                image = ImageTk.PhotoImage(file = "Images\cerrar.png")
                imageLabel.configure(image = image)
                imageLabel.image = image
            if aprobado:
                aprobado=False
            lblVideo.after(1000, visualizar)
        else:
            lblVideo.image = ""
            cap.release()
def finalizar():
    global cap
    cap.release()      
        
# Create a new object and execute.


# %%
capture_index=0
model_name='best.pt'
capture_index = capture_index
model = load_model(model_name)
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)

# %%

root = Tk()
btnIniciar = Button(root, text="Iniciar", width=45, command=iniciar)
btnIniciar.grid(column=0, row=0, padx=5, pady=5)
btnFinalizar = Button(root, text="Finalizar", width=45, command=finalizar)
btnFinalizar.grid(column=1, row=0, padx=5, pady=5)
lblVideo = Label(root)
lblVideo.grid(column=0, row=1, columnspan=2)
imageLabel =Label(root)
imageLabel.grid(column=2, row=1, columnspan=2)


# %%
root.mainloop()


