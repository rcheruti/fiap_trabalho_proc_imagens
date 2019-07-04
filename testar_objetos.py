
from utils import *
from darknet import Darknet
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure

# ----------------------------------------------

# Configurações na rede neural YOLOv3
# iremos carregar uma configuração que usa o tamanho da imagem que será analisada
# menor que o padrão, para tentar analisar a imagem em tempo real.
# A velocidade de detecção com a imagem menor que o padrão é aproximadamente 170ms.
cfg_file = 'cfg/yolov3_rapido.cfg'
m = Darknet(cfg_file)

# Pesos pré-treinados
weight_file = 'weights/yolov3.weights'
m.load_weights(weight_file)

# Rótulos de classes
namesfile = 'data/coco.names'
class_names = load_class_names(namesfile)

# ----------------------------------------------

nms_thresh = 0.6
iou_thresh = 0.4

# ----------------------------------------------

cap = cv2.VideoCapture( 0 )
nomeImg = 'imagens/local-entrevista.png'

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break

    copia   = frame.copy() # para salvar o original no arquivo mais tarde
    resImg  = cv2.resize(copia, (m.width, m.height))
    boxes   = detect_objects(m, resImg, iou_thresh, nms_thresh)
    # a função abaixo foi criado no arquivo "utils.py" para que retornace a imagem com os
    # retângulos para ser apresentada em tempo real
    copia   = plot_boxes_rapido( copia, boxes, class_names, plot_labels = True )

    cv2.imshow('Detector de objetos', copia)
    btnPressionado = cv2.waitKey( 1 ) & 0xFF
    if btnPressionado == ord('q') or btnPressionado == 13 or btnPressionado == 27 :
        cv2.imwrite( nomeImg, frame ) # salvar o frame originl (sem os retângulos)
        break

cap.release()
cv2.destroyAllWindows()


# ----------------------------------------------

# essa função irá criar um analisador Yolo no tamanho padrão
# para tentar obter um melhor resulatdo durante a detecção.
# A velocidade de detecção com a imagem no tamanho padrão é aproximadamente 750ms.
def analisar_imagem():
    # Configurações na rede neural YOLOv3
    cfg_file = 'cfg/yolov3.cfg'
    m = Darknet(cfg_file)

    # Pesos pré-treinados
    weight_file = 'weights/yolov3.weights'
    m.load_weights(weight_file)

    # Rótulos de classes
    namesfile = 'data/coco.names'
    class_names = load_class_names(namesfile)

    # ---
    # Definindo tamnaho do gráfico
    plt.rcParams['figure.figsize'] = [24.0, 14.0]

    # Carregar imagem para classificação
    img = cv2.imread('imagens/local-entrevista.png')

    # Conversão para o espaço RGB
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionamento para adatapção da primeira camada da rede neural 
    resized_image = cv2.resize(original_image, (m.width, m.height))

    # Deteteção de objetos na imagem
    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)

    # Objetos encontrados e nível de confiança
    print_objects(boxes, class_names)

    # Desenho no gráfico com os regângulos e rótulos
    plot_boxes(original_image, boxes, class_names, plot_labels = True)

    return boxes


boxes = analisar_imagem() # executar analise

# listar objetos detectados e as quantidades
objetos = list_objects(boxes, class_names)
objetosMap = {}
for item in objetos:
    if item in objetosMap:
        objetosMap[ item ] += 1
    else:
        objetosMap[ item ] = 1

for key in objetosMap.keys():
    print( '%s: %d' % (key, objetosMap[key]) )

