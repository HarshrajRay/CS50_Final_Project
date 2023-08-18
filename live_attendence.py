# Import necessary libraries
import os
import cv2
import face_recognition
import csv
from datetime import datetime

# Function to load known images and names from a folder
def load_known_faces(folder_path):
    known_images = []
    known_names = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Create the full file path
            file_path = os.path.join(folder_path, filename)
            # Add the file path to the list of known images
            known_images.append(file_path)
            # Extract the name from the file name (without the extension)
            name = os.path.splitext(filename)[0]
            # Add the name to the list of known names
            known_names.append(name)

    return known_images, known_names

# Function to mark attendance in a CSV file
def mark_attendance_in_csv(attendance_list):
    # Get the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create or open the CSV file in append mode
    with open('attendance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the attendance data (name and time) into the CSV file
        for name in attendance_list:
            writer.writerow([name, current_time])

# Load known images and names for attendance from a folder named 'known_faces'
known_images, known_names = load_known_faces("known_faces")

# Load and encode known images
known_encodings = []
for image_file in known_images:
    image = face_recognition.load_image_file(image_file)
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings.append(encoding)

# Initialize variables for face recognition
face_locations = []
face_encodings = []
face_names = []
attendance_list = []

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame from the video feed
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Initialize an empty list to store names for the current frame
    face_names = []

    # Check each face encoding found in the current frame
    for face_encoding in face_encodings:
        # Compare the face encoding with known encodings
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # Find the best match for the face
        if True in matches:
            matched_index = matches.index(True)
            name = known_names[matched_index]

        # Add the name to the face_names list for the current frame
        face_names.append(name)

    # Display the name and bounding box for each recognized face
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw the face rectangle and label on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (86, 232, 96), 2)
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition Attendance System', frame)

    # Update the attendance list with only the new names detected in this frame
    for name in face_names:
        if name != "Unknown" and name not in attendance_list:
            attendance_list.append(name)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy all windows
video_capture.release()
cv2.destroyAllWindows()

# Mark attendance in the CSV file
mark_attendance_in_csv(attendance_list)

# Print the attendance list
print("Attendance:")
for name in attendance_list:
    print(name)

