"""
Default configuration values for the video frame extraction tool.
"""

# Valid video file extensions
VALID_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

# Frame extraction defaults
DEFAULT_FRAME_INTERVAL = 1
DEFAULT_QUALITY = 100
DEFAULT_FORMAT = "png"

# Language filter settings
DEFAULT_LANGUAGE_FILTER = "EN"  # None means process all languages
AVAILABLE_LANGUAGES = [
    "EN",
    "HI",
]  # Available languages based on LANGUAGE_DICT in video_filename_parser
