import cv2
import os
import numpy as np
import random

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
dataset_path = "face_dataset"
os.makedirs(dataset_path, exist_ok=True)

def capture_faces(person_name):
    save_path = os.path.join(dataset_path, person_name)
    os.makedirs(save_path, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))
            img_path = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(img_path, face_resized)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved 100 images for {person_name}")

def augment_image(img):
    if random.choice([True, False]):
        img = cv2.flip(img, 1)
    if random.choice([True, False]):
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def train_recognizer():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=10, grid_x=8, grid_y=8)
    faces, labels = [], []
    label_map = {}
    label_counter = 0
    
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        label_map[label_counter] = folder
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = augment_image(img)
            faces.append(img)
            labels.append(label_counter)
        label_counter += 1
    
    if not faces:
        print("No face data found! Capture images first.")
        return
    
    faces, labels = np.array(faces, dtype=np.uint8), np.array(labels, dtype=np.int32)
    face_recognizer.train(faces, labels)
    face_recognizer.save("face_model.yml")
    
    with open("label_map.txt", "w") as f:
        for label, name in label_map.items():
            f.write(f"{label},{name}\n")
    
    print("Training complete! Model saved.")

def recognize_faces():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    if not os.path.exists("face_model.yml"):
        print("Error: Train the model first.")
        return
    face_recognizer.read("face_model.yml")
    
    names = {}
    if os.path.exists("label_map.txt"):
        with open("label_map.txt", "r") as f:
            for line in f:
                label, name = line.strip().split(",")
                names[int(label)] = name
    
    check_name = input("Enter the name to check: ")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))
            label, confidence = face_recognizer.predict(face_resized)
            confidence_text = f"{100 - confidence:.2f}%"
            recognized_name = names.get(label, "Unknown")
            
            threshold = 65 if len(names) > 1 else 50
            
            if recognized_name == check_name and confidence < threshold:
                text = f"{recognized_name} ({confidence_text})"
                color = (0, 255, 0)
            else:
                text = "Not " + check_name
                color = (0, 0, 255)
            
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("\n1. Capture Faces\n2. Train Model\n3. Recognize Faces\n4. Exit")
        choice = input("Enter your choice: ")
        
        if choice == "1":
            name = input("Enter person's name: ")
            capture_faces(name)
        elif choice == "2":
            train_recognizer()
        elif choice == "3":
            recognize_faces()
        elif choice == "4":
            break
        else:
            print("Invalid choice, try again.")