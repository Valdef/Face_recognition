import cv2
import numpy as np
import face_recognition

imgVal= face_recognition.load_image_file("Val_dataset/IMG_01.jpg")
imgVal = cv2.cvtColor(imgVal, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("Val_dataset/IMG_08.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgVal)[0]
faceEnc = face_recognition.face_encodings(imgVal)[0]
cv2.rectangle(imgVal, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
faceEncTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,255), 2)

results = face_recognition.compare_faces([faceEnc], faceEncTest)
faceDist = face_recognition.face_distance([faceEnc], faceEncTest)

if bool(results[0]): results = "Face Match"
else: results = "Face Not Match"

cv2.putText(imgTest, f'{results} {np.round(faceDist, 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow("Val", imgVal)
cv2.imshow("Val Test", imgTest)
cv2.waitKey(0)