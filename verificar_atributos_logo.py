
import cv2
import numpy as np


def mostrarImagem( cv2Imagem, titulo = '' ):
  cv2.imshow(titulo, cv2Imagem)
  cv2.waitKey()
  cv2.destroyAllWindows()
  pass

# ----------------------------

imagem =          cv2.imread('imagens/logo.png')
imagemCinza =     cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY )
copiaImg =        imagem.copy()
imagemAlvo =      cv2.imread('imagens/logo.png')
imagemAlvoCinza = cv2.cvtColor(imagemAlvo, cv2.COLOR_BGR2GRAY )

# imagemCinza =     cv2.bilateralFilter( imagemCinza, 9,75,75 )
# imagemAlvoCinza = cv2.bilateralFilter( imagemAlvoCinza, 9,75,75 )
imagemCinza =     cv2.GaussianBlur( imagemCinza, (5,5), 0 )
imagemAlvoCinza = cv2.GaussianBlur( imagemAlvoCinza, (5,5), 0 )

# ----------------------------
# detector ORB
detectorORB =     cv2.ORB_create( 1500 )
# para origem
keyPoints =       detectorORB.detect( imagemCinza, None )
keyPoints, desc = detectorORB.compute( imagemCinza, keyPoints )
# para alvo
keyPointsAlvo =   detectorORB.detect( imagemAlvoCinza, None )
keyPointsAlvo, descAlvo = detectorORB.compute( imagemAlvoCinza, keyPointsAlvo )

# ----------------------------
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

# ----------------------------
# ----------------------------
# resultado final

matchesMask = [ [0,0] for i in range( len( encontrados ) ) ]
mEncontrados = 0

for i, v in enumerate( encontrados ):
  if v is not False and len(v) > 1:
    (m, n) = v
    if m.distance < 0.7 * n.distance:
      mEncontrados += 1
      matchesMask[ i ] = [ 1, 0 ]
print( 'Encontrados: %d' % mEncontrados )

draw_params = dict(
  matchColor = (0,255,0), 
  singlePointColor = (255,0,0), 
  matchesMask = matchesMask, 
  flags = 0 )

image_detected = cv2.drawMatchesKnn(
    imagemCinza, 
    keyPoints, 
    imagemAlvoCinza, 
    keyPointsAlvo, 
    encontrados, 
    None, 
    **draw_params )

mostrarImagem( image_detected )
