from pathlib import Path
import extract_frames, remove_duplicates, detect_florence, create_dataframe, filter_labels, create_dataset

if True:
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
