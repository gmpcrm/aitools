from pathlib import Path
import sys
import cv2
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.yolo_model8 import AIDesYOLOv8DataModel

device = "cuda"
if device == "cuda":
    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        print("CUDA is not available. Switching to CPU")
        device = "cpu"


def process_video(video_path, model_path, output_folder):
    model = AIDesYOLOv8DataModel(model_path)
    if device == "cuda":
        model.model = model.model.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(frame, verbose=False)[0].boxes.data.tolist()
        for x1, y1, x2, y2, confidence, class_id in result:
            class_id = int(class_id)
            class_name = model.get_class_name(class_id)
            print(
                f"{frame_index:06d} {class_name}: {confidence:.2f} [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]"
            )

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # Display frame index
            cv2.putText(
                frame,
                f"Frame: {frame_index}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Save the frame
            output_image_path = output_folder_path / f"image_{frame_index:06d}.jpg"
            cv2.imwrite(str(output_image_path), frame)

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


model_path = "models/augmented_phone_9c.pt"
input_file = "c:/output/cam1001/cam10012024-05-14_06-41-43.5.mp4"
output_folder = "c:/results/cam10012024-05-14_06-41-43.5"
process_video(input_file, model_path, output_folder)
