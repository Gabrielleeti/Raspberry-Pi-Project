import argparse
import pickle
from collections import Counter
from pathlib import Path

import face_recognition
from PIL import Image, ImageDraw, ImageFont
import cv2

DEFAULT_ENCODINGS_PATH = Path("output/encodings2.pkl")
BOUNDING_BOX_COLOR = (0, 0, 255)  # Red color for bounding boxes (BGR format)
TEXT_COLOR = (255, 255, 255)  # White color for text (BGR format)

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    """
    Given an unknown image, get the locations and encodings of any faces and
    compare them against the known encodings to find potential matches.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)
    
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name, confidence = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name, confidence)

    del draw
    pillow_image.save("output3.jpg")
    pillow_image.show()


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


def _display_face(draw, bounding_box, name, confidence):
    """
    Draws bounding boxes around faces, a caption area, and text captions with confidence.
    """
    top, right, bottom, left = bounding_box
    font = ImageFont.truetype("LufgaMedium.ttf", 50)
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name, font=font)
    draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill=BOUNDING_BOX_COLOR, outline=BOUNDING_BOX_COLOR)
    draw.text((text_left, text_top), f"{name} ({confidence:.2f}%)", fill=TEXT_COLOR, font=font)


def recognize_faces_webcam(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
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
            if not name:
                name = "Unknown"

            cv2.rectangle(frame, (bounding_box[3], bounding_box[0]), (bounding_box[1], bounding_box[2]), BOUNDING_BOX_COLOR, 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{name} ({confidence:.2f}%)", (bounding_box[3] + 6, bounding_box[2] - 6), font, 0.5, TEXT_COLOR, 1)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Uncomment this line if you want to use the webcam
recognize_faces_webcam()

# Uncomment this line if you want to recognize faces from an image file
# recognize_faces(image_location="foreign.jpg")
