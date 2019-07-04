
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

    return None, None,None,None,None

# ----------------------------

data_path = 'faces/'

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
training_data, labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    training_data.append(images)
    labels.append(0)

# Criando uma matriz da lista de labels
labels = np.asarray(labels, dtype=np.int32)

# Treinamento do modelo
print("Iniciando treinamento do modelo.")
model = cv2.face.LBPHFaceRecognizer_create()
model.train(training_data, labels)
print("Modelo treinado com sucesso.")

persons = { 0: 'Eu' }

# ----------------------------
# Detectar face

cap = cv2.VideoCapture( 0 )
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
nomeFinal = 'success_candidate.png'

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    
    imgFace, x,y,w,h = face_extractor( frame )
    conf = -1
    nomeFace = 'Ninguem'
    encontrado = False

    if imgFace is not None:
        temp = cv2.cvtColor(imgFace, cv2.COLOR_BGR2GRAY )
        ids, conf = model.predict( temp )
        if conf < 40:
            encontrado = True
            nomeFace = persons[ ids ]
            # colocar retângulo em torno da face: ( img, (x,y), (w,h), cor, tamLinha )
            cv2.rectangle( frame, (x,y), (x+w,y+h), (0,64,200), 2)

    # grau de confiança da face analisada em relação a uma face esperada
    cv2.putText(
        frame, 'Confianca: '+ str( conf ) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0,64,200), 1, cv2.LINE_AA)
    # face encontrada
    cv2.putText(
        frame, 'Reconhecido: '+ str( nomeFace ) ,(10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0,64,200), 1, cv2.LINE_AA)

    # gravar imagem final, com os resultados
    if encontrado:
        cv2.imwrite( nomeFinal, frame )

    cv2.imshow('Encontrar face na imagem', frame)
    btnPressionado = cv2.waitKey( delay ) & 0xFF
    if btnPressionado == ord('q') or btnPressionado == 13 or btnPressionado == 27 :
        break

cap.release()
cv2.destroyAllWindows()
