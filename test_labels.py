from pathlib import Path
import extract_frames, remove_duplicates, detect_florence, create_dataframe, filter_labels, create_dataset

if False:
    source_folder = "c:/proplex/cases/Положительные кейсы/хорошая маркировка"
    target_folder = "c:/proplex/label/frames30fps"
    extractor = extract_frames.run(
        source_folder=source_folder,
        target_folder=target_folder,
    )

if False:
    source_folder = "/mnt/c/proplex/label/frames30fps/"
    target_folder = "/mnt/c/proplex/label/frames30fps.nodupes/"
    remover = remove_duplicates.run(
        source_folder,
        target_folder,
        ccthreshold=0.995,
        remove=False,
    )

    files = remover.get_files_to_remove()

if True:
    source_folder = "c:/proplex/label/frames30fps"
    target_folder = "c:/proplex/label/florence"
    yolo_model = "profile00210s.pt"
    yolo_model = "profile00210l3.pt"
    # yolo_model = "label00310l6.pt"
    general_prompt = "locate and extract inscription text on a white surface"
    query = "<CAPTION_TO_PHRASE_GROUNDING>"
    detector = detect_florence.run(
        source_folder=source_folder,
        target_folder=target_folder,
        models_path="c:/Projects/models",
        yolo_id=0,
        yolo_model=yolo_model,
        border=0,
        save_original=True,
        save_boxes=True,
        save_yolo=True,
        draw_boxes=True,
        device="cpu",
        mode="images",
        query=query,
        general_prompt=general_prompt,
    )
