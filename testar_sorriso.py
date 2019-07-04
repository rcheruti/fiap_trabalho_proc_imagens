
import dlib
import numpy as np 
import cv2
from scipy.spatial import distance as dist

def mostrarPontos(img, pontos, qtd):
    temp = np.array( pontos )
    count = 0
    for x,y in temp:
        cv2.rectangle(img, (x,y), (x+1,y+1), (0,0, 200), 3)
        count += 1
        if count >= qtd:
            break
    pass

# ---------------------------------------------

predictor_68_path = "modelos/shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(predictor_68_path)
detector = dlib.get_frontal_face_detector()

FACE_POINTS =       list(range(0, 68))  # 68
MOUTH_POINTS =      list(range(48, 61)) # 13
RIGHT_BROW_POINTS = list(range(17, 22)) # 5
LEFT_BROW_POINTS =  list(range(22, 27)) # 5
RIGHT_EYE_POINTS =  list(range(36, 42)) # 6
LEFT_EYE_POINTS =   list(range(42, 48)) # 6
NOSE_POINTS =       list(range(27, 35)) # 8
JAW_POINTS =        list(range(0, 17))  # 17

# para calcular proporção da boca vamos comparar os valores dos índices:
# horizontal:
# 6 - 0
# vertical:
# 11 - 1
# 10 - 2
# 9 - 3
# 8 - 4
# 7 - 5
# o ponto 12 fica nocamente no começo (prox do ponto 0)
def mouth_aspect_ratio(pontos):
    pontos = np.array( pontos )
    # h1 = dist.euclidean( pontos[1] , pontos[11] )
    h2 = dist.euclidean( pontos[2] , pontos[10] )
    h3 = dist.euclidean( pontos[3] , pontos[9] )
    h4 = dist.euclidean( pontos[4] , pontos[8] )
    # h5 = dist.euclidean( pontos[5] , pontos[7] )

    w1 = dist.euclidean( pontos[0] , pontos[6] )

    ra = ( h2 + h3 + h4 ) / ( 3.0 * w1 )
    return ra

# para calcular proporção dos olhos vamos comparar os valores dos índices:
# horizontal:
# 3 - 0
# vertical:
# 5 - 1
# 4 - 2
def eye_aspect_ratio(pontos):
    pontos = np.array( pontos )
    h1 = dist.euclidean( pontos[1] , pontos[5] )
    h2 = dist.euclidean( pontos[2] , pontos[4] )

    w1 = dist.euclidean( pontos[0] , pontos[3] )

    ra = ( h1 + h2 ) / ( 2.0 * w1 )
    return ra

# ---------------------------------------------

def annotate_landmarks_convex_hull_image(im):
    im = im.copy()
    rects = detector(im, 1)
    
    if len(rects) == 0:
        return im, 0, 0
    
    landmarks_list = []
    
    for rect in rects:
        landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

        for k, d in enumerate(rects):
            cv2.rectangle(im, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)

            points = cv2.convexHull(landmarks[NOSE_POINTS])
            cv2.drawContours(im, [points], 0, (0, 255, 0), 1)

            points = cv2.convexHull(landmarks[MOUTH_POINTS])
            cv2.drawContours(im, [points], 0, (0, 255, 0), 1)
            
            points = cv2.convexHull(landmarks[RIGHT_BROW_POINTS])
            cv2.drawContours(im, [points], 0, (0, 255, 0), 1)

            points = cv2.convexHull(landmarks[LEFT_BROW_POINTS])
            cv2.drawContours(im, [points], 0, (0, 255, 0), 1)

            points = cv2.convexHull(landmarks[RIGHT_EYE_POINTS])
            cv2.drawContours(im, [points], 0, (0, 255, 0), 1)
            
            points = cv2.convexHull(landmarks[LEFT_EYE_POINTS])
            cv2.drawContours(im, [points], 0, (0, 255, 0), 1)
            
            mouth_aspect = mouth_aspect_ratio( landmarks[MOUTH_POINTS] )
            eye_aspect_l = eye_aspect_ratio( landmarks[LEFT_EYE_POINTS] )
            eye_aspect_r = eye_aspect_ratio( landmarks[RIGHT_EYE_POINTS] )
            eye_aspect_m = (eye_aspect_r + eye_aspect_l) / 2

    return im, mouth_aspect, eye_aspect_m

# ---------------------------------------------

def bocejo1():
    cap = cv2.VideoCapture( 0 )
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    nomeImg = 'imagens/bocejo.png'
    qtd = 0
    bocejo_minimo = 0.55
    detectado = False

    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break

        frame, mouth_aspect, _ = annotate_landmarks_convex_hull_image(frame)
        prevDetectado = detectado
        detectado = False

        if mouth_aspect > bocejo_minimo:
            detectado = True
            if prevDetectado is False:
                qtd += 1

        cv2.putText(
            frame, 'Bocejo Aspect Ratio: '+ str( mouth_aspect ) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0,64,200), 1, cv2.LINE_AA)
        cv2.putText(
            frame, 'Bocejando' if detectado else 'Neutro' ,(10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0,64,200), 1, cv2.LINE_AA)

        if detectado:
            cv2.imwrite( nomeImg, frame )

        cv2.imshow('Detector de bocejos', frame)
        btnPressionado = cv2.waitKey( delay ) & 0xFF
        if btnPressionado == ord('q') or btnPressionado == 13 or btnPressionado == 27 :
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Bocejos identificados: %s" % str(qtd))


# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------

def olhos1():
    cap = cv2.VideoCapture( 0 )
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    nomeImg = 'imagens/olhos_fechados.png'
    qtd = 0
    olho_maximo = 0.20
    detectado = False

    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break

        frame, _, eye_aspect = annotate_landmarks_convex_hull_image(frame)
        prevDetectado = detectado
        detectado = False

        if eye_aspect < olho_maximo:
            detectado = True
            if prevDetectado is False:
                qtd += 1

        cv2.putText(
            frame, 'Olhos Aspect Ratio: '+ str( eye_aspect ) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0,64,200), 1, cv2.LINE_AA)
        cv2.putText(
            frame, 'Fechados' if detectado else 'Abertos' ,(10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0,64,200), 1, cv2.LINE_AA)

        if detectado:
            cv2.imwrite( nomeImg, frame )

        cv2.imshow('Detector de olhos fechados', frame)
        btnPressionado = cv2.waitKey( delay ) & 0xFF
        if btnPressionado == ord('q') or btnPressionado == 13 or btnPressionado == 27 :
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Olhos fechados identificados: %s" % str(qtd))

olhos1()

# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------

def sorriso1():
    cap = cv2.VideoCapture( 0 )
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    nomeImg = 'imagens/sorriso.png'
    qtd = 0
    sorriso_minimo = 0.30
    sorrimo_maximo = 0.36
    detectado = False

    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break

        frame, mouth_aspect, _ = annotate_landmarks_convex_hull_image(frame)
        prevDetectado = detectado
        detectado = False

        if mouth_aspect < sorriso_minimo or mouth_aspect > sorrimo_maximo:
            detectado = True
            if prevDetectado is False:
                qtd += 1

        cv2.putText(
            frame, 'Mouth Aspect Ratio: '+ str( mouth_aspect ) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0,64,200), 1, cv2.LINE_AA)
        cv2.putText(
            frame, 'Sorrindo' if detectado else 'Neutro' ,(10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0,64,200), 1, cv2.LINE_AA)

        if detectado:
            cv2.imwrite( nomeImg, frame )

        cv2.imshow('Detector de sorrisos', frame)
        btnPressionado = cv2.waitKey( delay ) & 0xFF
        if btnPressionado == ord('q') or btnPressionado == 13 or btnPressionado == 27 :
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Sorrisos identificados: %s" % str(qtd))

# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------

def teste1():
    cap = cv2.VideoCapture( 0 )
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    nomeImg = 'imagens/sorriso.png'
    qtd = 0
    sorriso_minimo = 0.22
    sorrimo_maximo = 0.30

    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        
        retangulos = detector(frame, 1)
        aspectRatio = 0
        sorrindo = False

        for ret in retangulos:
            marcos = np.matrix([[p.x, p.y] for p in predictor(frame, ret).parts()])
            pontos = cv2.convexHull( marcos[MOUTH_POINTS] )
            cv2.drawContours(frame, [pontos], 0, (0, 255, 0), 2)

            aspectRatio = mouth_aspect_ratio( marcos[MOUTH_POINTS] )
            if aspectRatio < sorrimo_maximo and aspectRatio > sorriso_minimo:
                sorrindo = True
                qtd += 1
        
        cv2.putText(
            frame, 'Mouth Aspect Ratio: '+ str( aspectRatio ) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0,64,200), 1, cv2.LINE_AA)
        cv2.putText(
            frame, 'Sorrindo' if sorrindo else 'Neutro' ,(10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0,64,200), 1, cv2.LINE_AA)
        
        if sorrindo:
            cv2.imwrite( nomeImg, frame )

        cv2.imshow('Contornos da face', frame)
        btnPressionado = cv2.waitKey( delay ) & 0xFF
        if btnPressionado == ord('q') or btnPressionado == 13 or btnPressionado == 27 :
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Sorrisos identificados: %s" % str(qtd))
