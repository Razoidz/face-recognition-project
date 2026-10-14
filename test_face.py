import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ----------------------------------
# INITIALIZE FACE RECOGNITION MODEL
# ----------------------------------

app = FaceAnalysis()
app.prepare(ctx_id=-1)


# ----------------------------------
# SAVE ENROLLED FACES
# ----------------------------------

def save_faces(stored_faces):
    np.savez("enrolled_faces.npz", **stored_faces)


# ----------------------------------
# LOAD ENROLLED FACES
# ----------------------------------

def load_faces():
    if not os.path.exists("enrolled_faces.npz"):
        return {}

    with np.load("enrolled_faces.npz") as data:
        stored_faces = {}

        for name in data.files:
            stored_faces[name] = data[name].copy()

    return stored_faces


# ----------------------------------
# ENROLL A PERSON
# ----------------------------------

def enroll_person(name, image, app, stored_faces):
    faces = app.get(image)

    if len(faces) != 1:
        return False

    embedding = faces[0].normed_embedding
    stored_faces[name] = embedding

    return True


# ----------------------------------
# FIND A MATCH
# ----------------------------------

def find_match(new_face, stored_faces, threshold):
    distances = {}

    for name, embedding in stored_faces.items():
        distance = np.linalg.norm(new_face - embedding)
        distances[name] = distance

    if not distances:
        return "Unknown"

    closest_person = min(distances, key=distances.get)

    if distances[closest_person] < threshold:
        return closest_person

    return "Unknown"


# ----------------------------------
# MAIN PROGRAM
# ----------------------------------

stored_faces = load_faces()

print("Enrolled people:", list(stored_faces.keys()))

while True:
    print("\n--- FACE RECOGNITION MENU ---")
    print("1. Enroll a person")
    print("2. Recognize a face")
    print("3. View enrolled people")
    print("4. Exit")

    choice = input("Choose an option: ").strip()

    if choice == "1":
        name = input("Enter the person's name: ").strip()

        if not name:
            print("Name cannot be empty.")
            continue

        image_path = input("Enter the image path: ").strip()
        image = cv2.imread(image_path)

        if image is None:
            print("Could not read the image. Check the path.")
            continue

        if name in stored_faces:
            print("That name is already enrolled.")
            continue

        success = enroll_person(name, image, app, stored_faces)

        if success:
            save_faces(stored_faces)
            print(f"{name} enrolled successfully!")
        else:
            print("Enrollment failed.")
            print("Use an image with exactly one detectable face.")

    elif choice == "2":
        image_path = input("Enter the image path to recognize: ").strip()
        image = cv2.imread(image_path)

        if image is None:
            print("Could not read the image. Check the path.")
            continue

        faces = app.get(image)

        if not faces:
            print("No faces detected.")
            continue

        threshold = 0.80

        for index, face in enumerate(faces, start=1):
            result = find_match(
                face.normed_embedding,
                stored_faces,
                threshold
            )

            print(f"Face {index}: {result}")

    elif choice == "3":
        if stored_faces:
            print("\nEnrolled people:")

            for name in stored_faces:
                print(f"- {name}")
        else:
            print("No people have been enrolled yet.")

    elif choice == "4":
        print("Exiting program.")
        break

    else:
        print("Invalid choice. Please select 1, 2, 3, or 4.")
 