import cv2
cascade = cv2.CascadeClassifier("F:\software\opencv-3.3.0\data\haarcascades\haarcascade_frontalface_alt2.xml")
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rect = cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=9,minSize=(100,100),flags = cv2.CASCADE_SCALE_IMAGE)
    for x,y,z,w in rect:
        cv2.rectangle(frame,(x,y),(x+z,y+w),(0,0,255),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break  
cap.release()
cv2.destroyAllWindows()

