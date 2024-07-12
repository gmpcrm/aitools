import cv2
import os
from tqdm import tqdm


def save_frames_at_intervals(source, intervals, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(source)

    # Get the frame rate and frame count of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frame width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the frame count
    count = 0

    base_name = os.path.basename(source)

    # Create dictionaries to hold VideoWriters for each interval
    video_writers = {}
    for interval in intervals:
        filename = os.path.join(target_dir, f"{base_name}.{interval}.mp4")
        video_writers[interval] = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        print(f"Created {filename} for interval {interval}")

    # Loop over the frames of the video with a progress bar
    for count in tqdm(range(frame_count), desc="Processing frames"):
        # Read the next frame
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Save the frame to the respective MP4 files
        for interval in intervals:
            if count % interval == 0:
                video_writers[interval].write(frame)

    # Release the video capture object and VideoWriters
    cap.release()
    for writer in video_writers.values():
        writer.release()


def process_videos_in_directory(source_dir, intervals, target_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".mp4"):
                source = os.path.join(root, file)
                print(f"Processing {source}")
                save_frames_at_intervals(source, intervals, target_dir)


source_dir = "c:/video"
target_dir = "c:/frames"
intervals = [5, 10, 15, 20]
intervals = [20]

process_videos_in_directory(source_dir, intervals, target_dir)
