"""
Video Frame Extraction Tool

This script processes videos in a directory structure, extracts frames,
and saves them to a mirrored directory structure.
"""

import os
import time
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2

from config.logger_config import logger
from config.defaults import (
    VALID_EXTENSIONS,
    DEFAULT_FRAME_INTERVAL,
    DEFAULT_QUALITY,
    DEFAULT_FORMAT,
    DEFAULT_LANGUAGE_FILTER,
    AVAILABLE_LANGUAGES,
)
from lib.video_filename_parser import parse_video_filename


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
                if count % 100 == 0:
                    logger.debug(f"Extracted {count} frames so far...")

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
    language_filter=DEFAULT_LANGUAGE_FILTER,
):
    """
    Process all videos in a directory structure, maintaining the same structure in output.

    Args:
        input_root (str): Root directory containing input videos
        output_root (str): Root directory for output files
        frame_interval (int): Extract one frame every N frames
        quality (int): Image quality (0-100)
        format (str): Output image format
        language_filter (str): Filter videos by language code (e.g., 'EN', 'HI')

    Returns:
        tuple: (total_videos, successful_videos, total_frames)
    """
    start_time = time.time()
    total_videos = 0
    successful_videos = 0
    total_frames = 0
    filtered_videos = 0

    input_root_path = Path(input_root)
    output_root_path = Path(output_root)

    logger.info(f"Starting batch processing from directory: {input_root}")
    logger.info(
        f"Using frame interval: {frame_interval}, quality: {quality}, format: {format}"
    )
    if language_filter:
        logger.info(f"Filtering videos by language: {language_filter}")

    for root, _, files in os.walk(input_root):
        video_files = [
            f for f in files if any(f.lower().endswith(ext) for ext in VALID_EXTENSIONS)
        ]

        if not video_files:
            continue

        for video_file in video_files:
            total_videos += 1
            video_path = os.path.join(root, video_file)

            # Filter by language if specified
            if language_filter:
                try:
                    video_metadata = parse_video_filename(video_file)
                    video_language = video_metadata.get("language")

                    if video_language != language_filter:
                        logger.debug(
                            f"Skipping {video_file} as it is {video_language} (filtered for {language_filter})"
                        )
                        filtered_videos += 1
                        continue
                except Exception as e:
                    logger.warning(
                        f"Could not parse filename {video_file} for language filtering: {e}"
                    )
                    # If we can't determine the language, we'll process it anyway
                    pass

            rel_path = Path(root).relative_to(input_root_path)
            video_name = Path(video_file).stem
            output_dir = output_root_path / rel_path / video_name

            try:
                frames_extracted = extract_frames_from_video(
                    video_path,
                    str(output_dir),
                    frame_interval=frame_interval,
                    quality=quality,
                    format=format,
                )

                if frames_extracted > 0:
                    successful_videos += 1
                    total_frames += frames_extracted

            except Exception as e:
                logger.error(f"Failed to process video {video_path}: {str(e)}")

    processing_time = time.time() - start_time
    processed_videos = total_videos - filtered_videos
    avg_time = processing_time / processed_videos if processed_videos > 0 else 0
    avg_frames = total_frames / successful_videos if successful_videos > 0 else 0
    extraction_rate = total_frames / processing_time if processing_time > 0 else 0

    summary = (
        "\nProcessing Summary:\n"
        "------------------\n"
        f"Total videos found: {total_videos}\n"
        f"Filtered by language: {filtered_videos}\n"
        f"Videos processed: {processed_videos}\n"
        f"Successfully processed: {successful_videos}\n"
        f"Failed: {processed_videos - successful_videos}\n"
        f"Total frames extracted: {total_frames}\n"
        f"Average frames per video: {avg_frames:.1f}\n"
        f"Total processing time: {processing_time:.2f} seconds\n"
        f"Average time per video: {avg_time:.2f} seconds\n"
        f"Overall extraction rate: {extraction_rate:.2f} frames/second"
    )

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
    parser.add_argument(
        "--language",
        "-l",
        choices=AVAILABLE_LANGUAGES,
        default=DEFAULT_LANGUAGE_FILTER,
        help=f"Filter videos by language (e.g., 'EN' for English, 'HI' for Hindi)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (INFO level)",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug output (DEBUG level)"
    )

    args = parser.parse_args()

    # Configure console logging level based on command line arguments
    if args.debug:
        from config.logger_config import console_handler

        console_handler.setLevel(logging.DEBUG)
    elif args.verbose:
        from config.logger_config import console_handler

        console_handler.setLevel(logging.INFO)

    start_time = time.time()
    logger.info(
        f"Starting video frame extraction with parameters: interval={args.interval}, "
        f"quality={args.quality}, format={args.format}, language_filter={args.language}"
    )

    process_directory(
        args.input,
        args.output,
        frame_interval=args.interval,
        quality=args.quality,
        format=args.format,
        language_filter=args.language,
    )

    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
