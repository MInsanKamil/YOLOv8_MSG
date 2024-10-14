import cv2
from ultralytics.models.yolo.model import YOLO

# Load YOLOv8n model
model = YOLO('best.pt')

def get_object_position(x_bbox, width_frame):
    # Define the boundaries for left, center, and right
    batas_kiri = width_frame / 3
    batas_tengah = 2 * width_frame / 3
    
    if x_bbox <= batas_kiri:
        return "Kiri"
    elif batas_kiri < x_bbox <= batas_tengah:
        return "Tengah"
    else:
        return "Kanan"
    
# Open video capture (0 for default camera)
cap = cv2.VideoCapture(0)

# Set frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Capture frame from video
    ret, frame = cap.read()
    
    if not ret:
        break

    # Get frame dimensions
    height_frame, width_frame = frame.shape[:2]
    
    # Perform object detection
    results = model(frame)
    
    # Loop through each detection
    for box in results[0].boxes:  # Access the first frame's boxes
        # Get the bounding box coordinates
        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
        
        # Get the predicted class and confidence score
        class_id = int(box.cls[0].cpu().numpy())
        confidence = box.conf[0].cpu().numpy()
        
        # Get the class name from the model
        class_name = model.names[class_id]
        
        # Calculate x_bbox (center of the detected bounding box)
        x_bbox = (x_min + x_max) / 2

        # Determine object position
        posisi_objek = get_object_position(x_bbox, width_frame)

        # Draw bounding box
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Create label with class name, confidence, and position
        label = f'{class_name} ({confidence:.2f}) - {posisi_objek}'
        
        # Draw the label above the bounding box
        cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame with detections
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
