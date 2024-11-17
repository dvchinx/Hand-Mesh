import cv2
import mediapipe as mp

dispositivoCaptura = cv2.VideoCapture(0)

mpManos = mp.solutions.hands

manos =  mpManos.Hands(static_image_mode = False,
                       max_num_hands = 1,
                       min_detection_confidence = 0.7,
                       min_tracking_confidence  = 0.7)

mpDibujar = mp.solutions.drawing_utils

while True:
    succes,img = dispositivoCaptura.read()
    
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    results = manos.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDibujar.draw_landmarks(img,handLms,mpManos.HAND_CONNECTIONS)
            
    cv2.imshow('Image',img)
    cv2.waitKey(1)