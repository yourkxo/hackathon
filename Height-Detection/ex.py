# install opencv "pip install opencv-python"
import cv2
import os

# Command to run the Body_Detection.py file
cmd = ' Body_Detection.py'

# Distance from camera to object (face) measured in centimeters
Known_distance =  300
# Width of face in the real world or Object Plane in centimeters
Known_width = 21

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Font style
fonts = cv2.FONT_HERSHEY_COMPLEX

# Face detector object
face_detector = cv2.CascadeClassifier("/Users/thanachunn/Height-Detection/haarcascade_frontalface_default.xml")

# Focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

# Distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance

# Function to get face data from the image
def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
        face_width = w
    return face_width

# Initialize the camera object to get frame from it
cap = cv2.VideoCapture(0)

# Loop through frame incoming from camera/video
while True:
    _, frame = cap.read()
    face_width_in_frame = face_data(frame)
    if face_width_in_frame != 0:
        # Calculate Focal length
        Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, face_width_in_frame)
        Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)
        Distance = round(Distance)
        if Distance in range(330, 360):
            print("Stand there and don't move")
            os.startfile("/Users/thanachunn/Height-Detection/Body_Detection.py")
            break
        elif Distance < 330:
            print("Step back")
        else:
            print("Come a little closer")
        cv2.putText(frame, f"Distance: {round(Distance, 2)} cms", (30, 35), fonts, 0.6, GREEN, 2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Close the camera
cap.release()

# Close the windows
cv2.destroyAllWindows()
