"""
Video Frame Extraction Tool

This script processes videos in a directory structure, extracts frames,
and saves them to a mirrored directory structure.
"""

import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2

from config.logger_config import logger
from config.defaults import (
    VALID_EXTENSIONS,
    DEFAULT_FRAME_INTERVAL,
    DEFAULT_QUALITY,
    DEFAULT_FORMAT,
)


def extract_frames_from_video(
    video_path,
    output_dir,
    frame_interval=DEFAULT_FRAME_INTERVAL,
    quality=DEFAULT_QUALITY,
    format=DEFAULT_FORMAT,
):
    """
    Extract frames from a video file at specified intervals.

    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the extracted frames
        frame_interval (int): Extract one frame every N frames
        quality (int): JPEG compression quality (0-100)
        format (str): Output image format (jpg, png)

    Returns:
        int: Number of frames extracted
    """
    os.makedirs(output_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return 0

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Total frames: {total_frames}, FPS: {fps}, Duration: {duration:.2f}s")
    logger.info(
        f"Extraction settings: Every {frame_interval} frames, Quality: {quality}, Format: {format}"
    )

    count = 0
    frame_number = 0
    start_time = time.time()

    expected_frames = total_frames // frame_interval + (
        1 if total_frames % frame_interval > 0 else 0
    )
    with tqdm(total=expected_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                output_file = os.path.join(
                    output_dir, f"frame_{frame_number:06d}.{format}"
                )

                if format.lower() in ["jpg", "jpeg"]:
                    encoding_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                elif format.lower() == "png":
                    encoding_params = [
                        cv2.IMWRITE_PNG_COMPRESSION,
                        min(9, quality // 10),
                    ]
                else:
                    encoding_params = []

                cv2.imwrite(output_file, frame, encoding_params)
                count += 1
                pbar.update(1)

            frame_number += 1

    video.release()
    elapsed_time = time.time() - start_time
    extraction_rate = count / elapsed_time if elapsed_time > 0 else 0

    logger.info(f"Extracted {count} frames to {output_dir} in {elapsed_time:.2f}s")
    logger.info(f"Extraction rate: {extraction_rate:.2f} frames/second")

    return count


def process_directory(
    input_root,
    output_root,
    frame_interval=DEFAULT_FRAME_INTERVAL,
    quality=DEFAULT_QUALITY,
    format=DEFAULT_FORMAT,
):
    """
    Process all videos in a directory structure, maintaining the same structure in output.

    Args:
        input_root (str): Root directory containing input videos
        output_root (str): Root directory for output files
        frame_interval (int): Extract one frame every N frames
        quality (int): Image quality (0-100)
        format (str): Output image format

    Returns:
        tuple: (total_videos, successful_videos, total_frames)
    """
    start_time = time.time()
    total_videos = 0
    successful_videos = 0
    total_frames = 0

    input_root_path = Path(input_root)
    output_root_path = Path(output_root)

    logger.info(f"Starting batch processing from directory: {input_root}")
    logger.info(
        f"Using frame interval: {frame_interval}, quality: {quality}, format: {format}"
    )

    for root, _, files in os.walk(input_root):
        video_files = [
            f for f in files if any(f.lower().endswith(ext) for ext in VALID_EXTENSIONS)
        ]

        if not video_files:
            continue

        for video_file in video_files:
            total_videos += 1
            video_path = os.path.join(root, video_file)

            rel_path = Path(root).relative_to(input_root_path)
            video_name = Path(video_file).stem
            output_dir = output_root_path / rel_path / video_name

            try:
                logger.info(f"Processing video {total_videos}: {video_path}")
                video_start_time = time.time()

                frames_extracted = extract_frames_from_video(
                    video_path,
                    str(output_dir),
                    frame_interval=frame_interval,
                    quality=quality,
                    format=format,
                )

                video_processing_time = time.time() - video_start_time

                if frames_extracted > 0:
                    successful_videos += 1
                    total_frames += frames_extracted
                    logger.info(
                        f"Completed video in {video_processing_time:.2f}s: {frames_extracted} frames extracted"
                    )
                else:
                    logger.warning(f"No frames extracted from {video_path}")

            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}", exc_info=True)

    processing_time = time.time() - start_time
    avg_time = processing_time / total_videos if total_videos > 0 else 0
    avg_frames = total_frames / successful_videos if successful_videos > 0 else 0
    extraction_rate = total_frames / processing_time if processing_time > 0 else 0

    summary = (
        "\nProcessing Summary:\n"
        "------------------\n"
        f"Total videos processed: {total_videos}\n"
        f"Successfully processed: {successful_videos}\n"
        f"Failed: {total_videos - successful_videos}\n"
        f"Total frames extracted: {total_frames}\n"
        f"Average frames per video: {avg_frames:.1f}\n"
        f"Total processing time: {processing_time:.2f} seconds\n"
        f"Average time per video: {avg_time:.2f} seconds\n"
        f"Overall extraction rate: {extraction_rate:.2f} frames/second"
    )

    print(summary)
    logger.info(summary)

    return total_videos, successful_videos, total_frames


def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(
        description="Extract frames from videos maintaining directory structure"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input directory containing videos"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for extracted frames"
    )
    parser.add_argument(
        "--interval",
        "-n",
        type=int,
        default=DEFAULT_FRAME_INTERVAL,
        help=f"Extract one frame every N frames (default: {DEFAULT_FRAME_INTERVAL} - extract all frames)",
    )
    parser.add_argument(
        "--quality",
        "-q",
        type=int,
        default=DEFAULT_QUALITY,
        choices=range(1, 101),
        metavar="[1-100]",
        help=f"Image quality for JPEG (1-100, default: {DEFAULT_QUALITY} - maximum quality)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["jpg", "png"],
        default=DEFAULT_FORMAT,
        help=f"Output image format (default: {DEFAULT_FORMAT})",
    )

    args = parser.parse_args()

    start_time = time.time()
    logger.info(
        f"Starting video frame extraction with parameters: interval={args.interval}, "
        f"quality={args.quality}, format={args.format}"
    )

    process_directory(
        args.input,
        args.output,
        frame_interval=args.interval,
        quality=args.quality,
        format=args.format,
    )

    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
