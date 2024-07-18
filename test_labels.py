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

if False:
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

if False:
    source_folder = "c:/proplex/label/florence"
    target_folder = "c:/proplex/label/florence.1"
    bbox_folder = "c:/proplex/label/florence.1/bbox"
    label_filter = ["locate and extract inscription text"]
    filter = filter_labels.run(
        source_folder=source_folder,
        target_folder=target_folder,
        label_filter=label_filter,
        extract_bbox=True,
        filter_match="exact",
        bbox_folder=bbox_folder,
        save_json=True,
    )

if False:
    source_folder = "c:/proplex/label/florence"
    target_folder = "c:/proplex/label/florence.2"
    bbox_folder = "c:/proplex/label/florence.2/bbox"
    label_filter = ["extract inscription text"]
    filter = filter_labels.run(
        source_folder=source_folder,
        target_folder=target_folder,
        label_filter=label_filter,
        extract_bbox=True,
        filter_match="exact",
        bbox_folder=bbox_folder,
        save_json=True,
    )

if True:
    source_folder = "c:/proplex/label/bbox.1"
    target_folder = "c:/proplex/label/bbox.1.florence"
    yolo_model = None
    general_prompt = "locate detect 1 2 3 4 5 6 7 8 9 0 char symbol"
    general_prompt = None
    query = "<CAPTION_TO_PHRASE_GROUNDING>"
    query = "<MORE_DETAILED_CAPTION>"
    query = "<OCR_WITH_REGION>"
    detector = detect_florence.run(
        source_folder=source_folder,
        target_folder=target_folder,
        models_path="c:/Projects/models",
        yolo_id=-1,
        yolo_model=yolo_model,
        border=0,
        save_original=True,
        save_boxes=True,
        draw_boxes=True,
        device="cpu",
        mode="images",
        query=query,
        general_prompt=general_prompt,
    )
