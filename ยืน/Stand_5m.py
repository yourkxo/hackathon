import cv2
import numpy as np

# Load YOLO model
cfg_file = r'/Users/thanachunn/darknet/cfg/yolov3.cfg'
weights_file = r'/Users/thanachunn/Downloads/yolov3.weights'
net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
classes = ['person']  # Only detect 'person' class

# Function to detect persons in a frame
def detect_persons(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    confidences = []
     
    for out in outs:
        for detection in out:
            scores = detection[5:]  
            class_id = np.argmax(scores) 
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.2:  # Class ID 0 corresponds to 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append((x, y, w, h))
                confidences.append(float(confidence))
    return boxes, confidences

# Function to measure person height
def measure_person_height(box, pillar_height_cm):
    _, _, _, h = box
    pixels_per_cm = pillar_height_cm / 100
    person_height_cm = h / pixels_per_cm
    return person_height_cm

# Function to draw pillar
def draw_pillar(frame, pillar_height_pixels):
    cv2.line(frame, (50, 0), (50, pillar_height_pixels), (255, 0, 0), thickness=2)

# Function to process video stream
def process_video():
    video = cv2.VideoCapture(0)  # Use 0 for default webcam
    if not video.isOpened():
        print("Error: Failed to open camera.")
        return
    
    width = 1240
    height = 670
    pillar_height_cm = 171.8
    '''
       ความสูงของกล้อง = 100cm
       ที่ระยะ 3 m ค่า phc = 291.5
       ที่ระยะ 4 m ค่า phc = 220
       ที่ระยะ 5 m ค่า phc = 171.8
    '''
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        frame = cv2.resize(frame, (width, height))
        boxes, confidences = detect_persons(frame)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.4)
        
        if indices is not None and len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                person_height_pixels = measure_person_height(box, pillar_height_cm)
                height_text = f"H: {person_height_pixels:.1f} cm"
                
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {i + 1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, height_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 150), 2)
        
        draw_pillar(frame, int(pillar_height_cm * (height / 100)))
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('r'):
            break
    
    video.release()
    cv2.destroyAllWindows()

# Run the real-time video processing
process_video()