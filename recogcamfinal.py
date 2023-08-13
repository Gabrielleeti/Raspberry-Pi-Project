import pickle
from collections import Counter
from pathlib import Path
import requests
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import time
import cv2

DEFAULT_ENCODINGS_PATH = Path("output/encodings2.pkl")
    

def create_directory_if_not_exists(directory_path):
    directory_path.mkdir(exist_ok=True)

def encode_known_faces(model="hog", encodings_location=DEFAULT_ENCODINGS_PATH):
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
    create_directory_if_not_exists(Path("training"))
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        names.extend([name] * len(face_encodings))
        encodings.extend(face_encodings)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)



def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown encoding and all known encodings, find the known
    encoding with the most matches and return the recognized name and confidence.
    """
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)

    if votes:
        most_common_name, num_votes = votes.most_common(1)[0]
        total_votes = len(boolean_matches)
        confidence = (num_votes / total_votes) * 100
        return most_common_name, confidence

    return None, None


def recognize_faces_webcam(model="hog", encodings_location=DEFAULT_ENCODINGS_PATH):
    """
    Recognize faces from webcam and display the results in real-time.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    video_capture = cv2.VideoCapture(0)  # Use default webcam (index 0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_face_locations = face_recognition.face_locations(input_image, model=model)
        input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

        for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
            name, confidence = _recognize_face(unknown_encoding, loaded_encodings)
            if confidence is not None and confidence < 45:
                name = "Unknown"  
            if name is None:
                name = "Unknown"
            if confidence is not None and confidence > 45:                
                
                                             
                try:
                    requests.get(f"http://127.0.0.1:5000/add_name/{name}")
                except:
                    pass
            print(name,confidence)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Uncomment this line if you want to use the webcam
recognize_faces_webcam()

# Uncomment this line if you want to train
# encode_known_faces()