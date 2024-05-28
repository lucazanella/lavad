from pathlib import Path


def find_unprocessed_videos(video_list, output_dir, pathname):
    processed_videos = set(Path(output_dir).glob(pathname))

    if processed_videos:
        # Identify the most recently processed video
        last_processed_video_path = sorted(processed_videos)[-1]
        last_processed_video_idx = find_last_processed_video_index(
            video_list, last_processed_video_path
        )
        last_processed_video_idx += 1

        if last_processed_video_idx == len(video_list):
            return []
        else:
            return video_list[last_processed_video_idx:]  # Videos that are not fully processed
    else:
        return video_list  # If there are no processed videos, return the entire list


def find_last_processed_video_index(video_list, last_processed_video_path):
    last_processed_video_name = last_processed_video_path.stem
    return [video.path.name for video in video_list].index(last_processed_video_name)
