import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam.
# It includes some basic performance tweaks to make things run faster.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
oscar_image = face_recognition.load_image_file("oscar.jpg")
oscar_face_encoding = face_recognition.face_encodings(oscar_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    oscar_face_encoding,
    biden_face_encoding
]

known_face_names = [
    "Oscar",
    "Joe Biden"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Check camera
    if not ret:
        print("Failed to read webcam")
        break

    # Only process every other frame of video to save time
    if process_this_frame:

        # Resize frame of video to 1/4 size
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.25,
            fy=0.25
        )

        # Convert BGR to RGB safely for dlib
        rgb_small_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        # Make array contiguous to avoid dlib errors
        rgb_small_frame = np.ascontiguousarray(
            rgb_small_frame
        )

        # Find all faces
        face_locations = face_recognition.face_locations(
            rgb_small_frame
        )

        # Find all face encodings
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame,
            face_locations
        )

        face_names = []

        for face_encoding in face_encodings:

            # Compare faces
            matches = face_recognition.compare_faces(
                known_face_encodings,
                face_encoding
            )

            name = "Unknown"

            # Find best match
            face_distances = face_recognition.face_distance(
                known_face_encodings,
                face_encoding
            )

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display results
    for (top, right, bottom, left), name in zip(
        face_locations,
        face_names
    ):

        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 0, 255),
            2
        )

        # Draw label background
        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            (0, 0, 255),
            cv2.FILLED
        )

        # Draw name text
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            font,
            1.0,
            (255, 255, 255),
            1
        )

    # Display video
    cv2.imshow("Video", frame)

    # Quit with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
video_capture.release()
cv2.destroyAllWindows()
