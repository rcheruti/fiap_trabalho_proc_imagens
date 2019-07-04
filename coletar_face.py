
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import cv2
import numpy as np
import datetime
from os import listdir
from os.path import isfile, join

def exibir_imagem(imagem):
    figure(num=None, figsize=(15, 10))
    image_plt = mpimg.imread(imagem)
    plt.imshow(image_plt)
    plt.axis('off')
    plt.show()

# ----------------------------

# carregar classificador em cascata
face_classifier = cv2.CascadeClassifier('classificadores/haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.2, 5)
    
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face, x,y,w,h

    return None

# ----------------------------
# salvar imagens de treino

cap = cv2.VideoCapture( 0 )
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
contagem = 100
prefixo = 'faces/imagem_'
sufixo = '.png'

print("Iniciando coleta de amostras")
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    
    imgFace, x,y,w,h = face_extractor( frame )

    if imgFace is not None:
        cv2.imwrite( prefixo + str( contagem ) + sufixo, imgFace )
        contagem -= 1

    # quantidade de coletas faltando
    cv2.putText(
        frame, 'Coletas restantes: '+ str( contagem ) ,(10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0,64,200), 1, cv2.LINE_AA)
        
    cv2.imshow('Imagem de Treino', frame)
    btnPressionado = cv2.waitKey( delay ) & 0xFF
    if btnPressionado == ord('q') or btnPressionado == 13 or btnPressionado == 27 or contagem < 0 :
        break

cap.release()
cv2.destroyAllWindows()
print("Coleta de amostras concluída")
