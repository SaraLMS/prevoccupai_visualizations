"""
Functions to extract device start times from filenames.

Available Functions
-------------------
[Public]
get_device_filename_timestamp(...): Scan a folder of raw data files, extract device names and their start times from filenames, and return them as a dictionary.
-------------------

[Private]
_extract_timestamp_from_filename(...): Parse the timestamp from an OpenSignals-style filename and convert it to 'hh:mm:ss.000' format.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from typing import Dict
import re

# internal imports
import load

STUDIO_DATA = 'StudioData'
# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def get_device_filename_timestamp(folder_path: str) -> Dict[str, str]:
    """
    Extracts the start time for each device based on the timestamps in the filenames within a folder.

    This function scans the specified folder, identifies files corresponding to devices, and parses each filename
    to extract the device name and its associated timestamp. The extracted time (formatted as 'hh:mm:ss.000')
    is assumed to represent the start time of data collection for that device.

    :param folder_path: Path to the folder containing the raw data files.
    :return: A dictionary mapping each device name to its extracted start time.
             Example: {"watch": "11:00:01.000", "F0A55C68B2E1": "11:05:34.000"}
    """

    # innit dictionary to store the results
    start_times_dict: Dict[str, str] = {}

    for filename in os.listdir(folder_path):

        # ignore mvc
        if STUDIO_DATA in filename:
            continue

        # extract device from filename
        device_name = load.extract_device_from_filename(filename)

        # extract timestamp from filename
        device_timestamp = _extract_timestamp_from_filename(filename)

        # update dictionary
        start_times_dict[device_name] = device_timestamp

    return start_times_dict


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _extract_timestamp_from_filename(filename: str) -> str:
    """
    Extracts the time portion from an OpenSignals filename and converts it to 'hh:mm:ss.000' format.

    Example:
        Input:  "opensignals_ANDROID_ROTATION_VECTOR_2022-05-02_11-00-01"
        Output: "11:00:01.000"

    :param filename: The filename string containing a timestamp
    :return: The timestamp in the 'hh:mm:ss.000' format
    """
    # Regex to extract the timestamp from filename - format is hh-mm-ss
    match = re.search(r'_(\d{2}-\d{2}-\d{2})(?:\.\w+)?$', filename)

    if not match:
        raise ValueError(f"No valid time found in filename: {filename}")

    # Change format to hh:mm:ss.000
    time_str = match.group(1)

    return time_str

