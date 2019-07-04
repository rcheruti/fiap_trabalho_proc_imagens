import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from scipy.spatial import distance as dist
import collections
from matplotlib.pyplot import figure

%matplotlib inline




def detector(imagem, template):
    # Conversão da imagem par escala de cinza
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # IMPLEMENTAR
    #
    # Escolha um detector de imagens adequado, configure e aplique um algoritmo de match
    # Esta função deve retornar o número de correspondências de uma imagem versus seu template
    
    # detector ORB
    detectorORB =     cv2.ORB_create( 500 )
    # para origem
    keyPoints =       detectorORB.detect( imagemCinza, None )
    keyPoints, desc = detectorORB.compute( imagemCinza, keyPoints )
    # para alvo
    keyPointsAlvo =   detectorORB.detect( template, None )
    keyPointsAlvo, descAlvo = detectorORB.compute( template, keyPointsAlvo )
    
    # parâmetros para o FLANN
    paramsIndex = {
      'algorithm':          6 ,
      'table_number':       6 ,
      'key_size':           12 ,
      'multi_probe_level':  1 ,
    }
    paramsBusca = {
      'checks':             50 ,
    }
    # comparador FLANN
    comparadorFLANN = cv2.FlannBasedMatcher( paramsIndex, paramsBusca )
    encontrados =     comparadorFLANN.knnMatch( desc, descAlvo, k = 2 )
    

    return len(encontrados)

cap = cv2.VideoCapture(0)


# IMPLEMENTAR
# Carregue a imagem do logotipo

image_template = cv2.imread('imagens/logo.png')
image_template = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY )


# Função de suporte para exibição de imagens no Jupyter

def exibir_imagem(imagem):
    figure(num=None, figsize=(15, 10))
    image_plt = mpimg.imread(imagem)
    plt.imshow(image_plt)
    plt.axis('off')
    plt.show()
    
    
    
while True:
    # Obtendo imagem da câmera
    ret, frame = cap.read()
    
    if ret:
        # Definindo região de interesse (ROI)
        height, width = frame.shape[:2]
        top_left_x = int(width / 3)
        top_left_y = int((height / 2) + (height / 4))
        bottom_right_x = int((width / 3) * 2)
        bottom_right_y = int((height / 2) - (height / 4))
    
        # Desenhar retângulo na região de interesse
        cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), 255, 3)

        # Obtendo região de interesse para validação do detector
        cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]

        # Executando o detector, definindo um limiar e fazendo a comparação para validar se o logotipo foi detectado
        # IMPLEMENTAR
        
        resultado = detector(cropped, image_template)
        print( resultado )

        cv2.imshow("Identificacao de Template", frame)
        
    # Se for teclado Enter (tecla 13) deverá sair do loop e encerrar a captura de imagem
    if cv2.waitKey(1) == 13: 
        break

cap.release()
cv2.destroyAllWindows()   