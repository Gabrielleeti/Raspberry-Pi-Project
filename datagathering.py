import cv2
import os

def extract_frames(video_path, output_folder,id, num_frames=100):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the step to evenly sample 'num_frames' frames from the video
    step = max(1, total_frames // num_frames)

    frame_count = 0
    frame_number = 0

    while frame_count < num_frames:
        ret, frame = video_capture.read()
        
        # Check if we've reached the end of the video
        if not ret:
            break
        
        # Save the frame to the output folder
        output_path = os.path.join(output_folder, f"{id}{frame_number:01d}.jpg")
        cv2.imwrite(output_path, frame)

        frame_count += 1
        frame_number += step

    video_capture.release()
    print(f"{frame_count} frames extracted and saved to {output_folder}.")

# Replace 'input_video.mp4' with the path to your video file
# Replace 'output_folder' with the path to the folder where you want to save the frames
extract_frames('IMG_4646.mp4', 'training/comfort',"comfort", num_frames=100)
