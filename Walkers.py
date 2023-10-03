import cv2
# Crear nuestro clasificador de cuerpos

cascada = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# Inicializar la captura de video para nuestro archivo de video
cap = cv2.VideoCapture('walking.avi')

# Comenzar el bucle una vez que el video esté cargado exitosamente
while True:
    
    # Leer el primer cuadro
    ret, frame = cap.read()
    
    # Convertir cada cuadro a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pasar el cuadro a nuestro clasificador de cuerpos
    body = cascada.detectMultiScale(gray, 1.2, 3)
    bodies = len(body)
    cv2.putText(frame,str(bodies),(75,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    for (x,y,w,h) in body:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (30,100,40), 4)
    
    # Extraer las cajas envolventes para cualquier cuerpo identificado
    
    cv2.imshow('detectado', frame)

    
    if cv2.waitKey(1) == 32: #32 es la barra espaciadora
        break

cap.release()
cv2.destroyAllWindows()
