import cv2
import numpy as np
import tensorflow as tf
import winsound  # For Windows beep alert (Mac/Linux alternative given below)
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model_scratch.keras")

# Class labels
classes_dict = {
    0: "Safe Driving",
    1: "Texting Right",
    2: "Talking on Phone - Right",
    3: "Texting Left",
    4: "Talking on Phone - Left",
    5: "Operating Radio",
    6: "Drinking",
    7: "Reaching Behind",
    8: "Hair & Makeup",
    9: "Talking to Passenger"
}

# Parameters
FRAME_THRESHOLD = 20  # Trigger alert if distracted for 20 frames
frame_count = 0

# Start webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for model
    img = cv2.resize(frame, (64, 64))  # Resize image
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)

    # Predict distraction level
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    label = classes_dict[predicted_class]

    # Display prediction result on screen
    color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Track distraction
    if predicted_class != 0:
        frame_count += 1
    else:
        frame_count = 0

    # Trigger alert if distracted for too long
    if frame_count >= FRAME_THRESHOLD:
        print("⚠ ALERT: Driver Distracted! ⚠")
        winsound.Beep(1000, 500)  # Windows beep sound
        frame_count = 0  # Reset counter after alert

    # Show webcam feed
    cv2.imshow("Driver Monitoring", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
