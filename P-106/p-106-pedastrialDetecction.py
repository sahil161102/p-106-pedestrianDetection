import cv2

#Load the Cascade Classifier File C:\Python310\Lib\site-packages\cv2\data
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml') 

# Define a video capture object
vid = cv2.VideoCapture(0)

while(True):
   
    # Capture the video frame by frame
    ret, frame = vid.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    body = body_classifier.detectMultiScale(gray, 1.1, 5)

    # Draw the rectangle around each face
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit Window by Spacebar Key
    if cv2.waitKey(25) == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()