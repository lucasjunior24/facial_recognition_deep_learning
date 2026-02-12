import cv2.face

# Create a new, empty recognizer instance
recognizer = cv2.face.EigenFaceRecognizer.create()

# Load the saved data from the file
recognizer.read("eigenface_model.yml")

