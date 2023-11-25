#FACE ANONYMIZER

import os
import argparse
import cv2
import mediapipe as mp

def blur_faces(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detection_results = face_detection.process(img_rgb)
    H, W, _ = img.shape 

    if detection_results.detections is not None:
        for detection in detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Ensure the bounding box is within the image dimensions
            x, y, w, h = max(0, x), max(0, y), min(iw - 1, w), min(ih - 1, h)

            # Blur the entire face region
            img[y:y + h, x:x + w, :] = cv2.blur(img[y:y + h, x:x + w, :], (20, 20))

    return img

def process_image(file_path, output_dir):
    img = cv2.imread(file_path)
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        img = blur_faces(img, face_detection)
        output_path = os.path.join(output_dir, 'Output.jpeg')
        cv2.imwrite(output_path, img)

def process_video(file_path, output_dir):
    cap = cv2.VideoCapture(file_path)
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        ret, frame = cap.read()
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'Output.mp4'), cv2.VideoWriter_fourcc(*'MPV4'), 25, (frame.shape[1], frame.shape[0]))

        while ret:
            frame = blur_faces(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()

def process_webcam(output_dir):
    cap = cv2.VideoCapture(0)
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        ret, frame = cap.read()

        while ret:
            frame = blur_faces(frame, face_detection)
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            ret, frame = cap.read()

        cap.release()

if _name_ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='webcam')
    parser.add_argument("--file_path", default=None)
    args = parser.parse_args()

    output_directory = r"C:\Users\tssri\OneDrive\Desktop\SRM\Semester 5\Computer Vision\Mini-Project\Output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.mode == "image":
        process_image(args.file_path, output_directory)

    elif args.mode == "video":
        process_video(args.file_path, output_directory)

    elif args.mode == "webcam":
        process_webcam(output_directory)
