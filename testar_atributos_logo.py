
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import cv2
import numpy as np
import datetime

def exibir_imagem(imagem):
    figure(num=None, figsize=(15, 10))
    image_plt = mpimg.imread(imagem)
    plt.imshow(image_plt)
    plt.axis('off')
    plt.show()

# ----------------------------

# detector ORB
detectorORB =     cv2.ORB_create( 1500 )
# parâmetros para o FLANN
paramsIndex = {
    'algorithm':          6 ,
    'table_number':       6 ,
    'key_size':           12 ,
    'multi_probe_level':  1 ,
}
paramsBusca = {
    'checks':             80 ,
}
# comparador FLANN
comparadorFLANN = cv2.FlannBasedMatcher( paramsIndex, paramsBusca )

# ----------------------------

# lista de imagens
imagensStr = [
    'logos/img_preto.png',
    'logos/img_branco.png',
]
imagens = []
evidenciaNome = 'evidencia.png'

for imgStr in imagensStr :
    temp = cv2.imread( imgStr )
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY )
    temp = cv2.GaussianBlur( temp, (7,7), 0 )
    keyPoints =       detectorORB.detect( temp, None )
    keyPoints, desc = detectorORB.compute( temp, keyPoints )
    imagens.append( desc )
    pass

# ----------------------------
# ----------------------------
# resultado final

cap = cv2.VideoCapture( 0 )
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
num = 0
corte = 150


while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    start_time = datetime.datetime.now()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur( gray, (7,7), 0 )

    keyPointsAlvo = detectorORB.detect( gray, None )
    keyPointsAlvo, descAlvo = detectorORB.compute( gray, keyPointsAlvo )

    buscas = []

    for imgDesc in imagens:
        try:
            encontrados = comparadorFLANN.knnMatch( imgDesc, descAlvo, k = 2 )
        except cv2.error:
            encontrados = []
        # matchesMask = [ [0,0] for i in range( len( encontrados ) ) ]
        mEncontrados = 0
        try:
            for i, v in enumerate( encontrados ):
                if v is not False and len(v) > 1:
                    (m, n) = v
                    if m.distance < 0.6 * n.distance:
                        mEncontrados += 1
                        # matchesMask[ i ] = [ 1, 0 ]
        except ValueError:
            pass
        buscas.append( mEncontrados )
        if mEncontrados > corte :
            break
        pass

    maiorEncontrado = np.max( np.array( buscas ) )
    foiEncontrado =   maiorEncontrado > corte

    elapsed_time = datetime.datetime.now() - start_time

    # acerto
    cv2.putText(
        frame, str( maiorEncontrado ) ,(10,40), cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0,230,0) if (foiEncontrado) else (0,0,235),
        1, cv2.LINE_AA)

    # tempo para busca nas imagens
    cv2.putText(
        frame, 'Tempo (us): '+ str( elapsed_time.microseconds ) ,(10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (200,200,0), 1, cv2.LINE_AA)

    if foiEncontrado:
        cv2.imwrite( evidenciaNome, frame )
    
    cv2.imshow('Encontrar logo na imagem', frame)
    btnPressionado = cv2.waitKey( delay ) & 0xFF
    if btnPressionado == ord('q') or btnPressionado == 13 or btnPressionado == 27 :
        break


cap.release()
cv2.destroyAllWindows()

exibir_imagem( evidenciaNome )

