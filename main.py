# import cv2
# import face_recognition
# import numpy as np
#
# def compare_faces(image1_path, image2_path):
#     # Load the images
#     image1 = face_recognition.load_image_file(image1_path)
#     image2 = face_recognition.load_image_file(image2_path)
#
#     # Detect face landmarks in the images
#     face_landmarks1 = face_recognition.face_landmarks(image1)
#     face_landmarks2 = face_recognition.face_landmarks(image2)
#
#     # Check if a face is detected in each image
#     if len(face_landmarks1) == 0 or len(face_landmarks2) == 0:
#         print("Error: Face not detected in one or both images.")
#         return
#
#     # Calculate similarity based on eye, nose, and mouth landmarks
#     landmarks1 = np.concatenate([
#         face_landmarks1[0]["left_eye"],
#         face_landmarks1[0]["right_eye"],
#         face_landmarks1[0]["nose_tip"],
#         face_landmarks1[0]["top_lip"],
#         face_landmarks1[0]["bottom_lip"]
#     ])
#
#     landmarks2 = np.concatenate([
#         face_landmarks2[0]["left_eye"],
#         face_landmarks2[0]["right_eye"],
#         face_landmarks2[0]["nose_tip"],
#         face_landmarks2[0]["top_lip"],
#         face_landmarks2[0]["bottom_lip"]
#     ])
#
#     distance = np.linalg.norm(landmarks1 - landmarks2)
#     similarity_percentage = max((1 - distance) * 100, 0)
#
#     # Resize images to display side by side
#     height = max(image1.shape[0], image2.shape[0])
#     image1 = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
#     image2 = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))
#
#     # Concatenate the images horizontally
#     result_image = np.concatenate((image1, image2), axis=1)
#
#     # Display similarity percentage on the result image
#     cv2.putText(result_image, "Similarity: {:.2f}%".format(similarity_percentage), (50, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
#
#     # Create a window and display the result image
#     cv2.namedWindow("Face Comparison", cv2.WINDOW_NORMAL)
#     cv2.imshow("Face Comparison", result_image)
#     cv2.resizeWindow("Face Comparison", 1200, 600)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# # Provide the paths to the two face images for comparison
# image1_path = "image/img.jpeg"  # 첫 번째 사진 경로
# image2_path = "image/test.jpeg"  # 두 번째 사진 경로
#
# # Compare the faces in the images
# compare_faces(image1_path, image2_path)



#얼굴에 선
import cv2
import face_recognition
import numpy as np

def compare_faces(image1_path, image2_path):
    # Load the images
    image1 = face_recognition.load_image_file(image1_path)
    image2 = face_recognition.load_image_file(image2_path)

    # Detect face landmarks in the images
    face_landmarks1 = face_recognition.face_landmarks(image1)
    face_landmarks2 = face_recognition.face_landmarks(image2)

    # Check if a face is detected in each image
    if len(face_landmarks1) == 0 or len(face_landmarks2) == 0:
        print("Error: Face not detected in one or both images.")
        return

    # Calculate similarity based on eye, nose, and mouth landmarks
    landmarks1 = np.concatenate([
        np.array(face_landmarks1[0]["left_eye"]),
        np.array(face_landmarks1[0]["right_eye"]),
        np.array(face_landmarks1[0]["nose_tip"]),
        np.array(face_landmarks1[0]["top_lip"]),
        np.array(face_landmarks1[0]["bottom_lip"])
    ])

    landmarks2 = np.concatenate([
        np.array(face_landmarks2[0]["left_eye"]),
        np.array(face_landmarks2[0]["right_eye"]),
        np.array(face_landmarks2[0]["nose_tip"]),
        np.array(face_landmarks2[0]["top_lip"]),
        np.array(face_landmarks2[0]["bottom_lip"])
    ])

    distance = np.linalg.norm(landmarks1 - landmarks2)
    similarity_percentage = max((1 - distance) * 100, 20)

    # Draw red bounding boxes around the detected faces
    for landmark in face_landmarks1:
        for feature_name, points in landmark.items():
            for i in range(len(points) - 1):
                cv2.line(image1, points[i], points[i + 1], (0, 0, 255), 2)

    for landmark in face_landmarks2:
        for feature_name, points in landmark.items():
            for i in range(len(points) - 1):
                cv2.line(image2, points[i], points[i + 1], (0, 0, 255), 2)

    # Resize images to display side by side
    height = max(image1.shape[0], image2.shape[0])
    image1 = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
    image2 = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))

    # Concatenate the images horizontally
    result_image = np.concatenate((image1, image2), axis=1)

    # Display similarity percentage on the result image
    cv2.putText(result_image, "Similarity: {:.2f}%".format(similarity_percentage), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

    # Create a window and display the result image
    cv2.namedWindow("Face Comparison", cv2.WINDOW_NORMAL)
    cv2.imshow("Face Comparison", result_image)
    cv2.resizeWindow("Face Comparison", 1200, 600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the paths to the two face images for comparison
image1_path = "image/img.jpeg"  # 첫 번째 사진 경로
image2_path = "image/img-2.jpeg"  # 두 번째 사진 경로

# Compare the faces in the images
compare_faces(image1_path, image2_path)

