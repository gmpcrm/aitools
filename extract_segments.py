import ffmpeg
import os
import argparse


class VideoSegmentProcessor:
    def __init__(self, segments_file, target_folder=None):
        self.segments_file = segments_file
        self.target_folder = target_folder

    def extract_clips(self, input_file, segments):
        if not os.path.exists(input_file):
            print(f"Файл {input_file} не существует. Пропуск.")
            return

        base_name = os.path.basename(input_file)
        file_name, file_extension = os.path.splitext(base_name)
        target_path = (
            self.target_folder if self.target_folder else os.path.dirname(input_file)
        )

        os.makedirs(target_path, exist_ok=True)

        for idx, segment in enumerate(segments):
            start_time, duration = map(int, segment.split(":"))
            output_file = os.path.join(
                target_path, f"{file_name}.{idx}{file_extension}"
            )

            if os.path.exists(output_file):
                print(f"Сегмент {output_file} уже существует. Пропуск.")
                continue

            (
                ffmpeg.input(input_file, ss=start_time, t=duration)
                .output(output_file, codec="copy")
                .run(overwrite_output=True)
            )

            print(f"Извлечен клип {idx}: {output_file}")

    def parse_segments_file(self):
        video_segments = []
        with open(self.segments_file, "r") as f:
            for line in f:
                video_file, segments_str = line.strip().split(",", 1)
                segments = segments_str.split(",")
                video_segments.append((video_file.strip('"'), segments))
        return video_segments

    def process_videos(self):
        video_segments = self.parse_segments_file()
        for video_file, segments in video_segments:
            self.extract_clips(video_file, segments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video segments.")
    parser.add_argument(
        "--segments",
        default="segments_split.txt",
        help="Путь к файлу с обработанными сегментами",
    )
    parser.add_argument(
        "--target_folder",
        help="Путь к папке для новых файлов",
    )

    args = parser.parse_args()

    processor = VideoSegmentProcessor(
        segments_file=args.segments, target_folder=args.target_folder
    )
    processor.process_videos()
