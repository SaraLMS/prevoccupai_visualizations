"""
Functions to parse the filenames and extract relevant information (i.e.: device name, sensor data, ext)

Available Functions
-------------------
[Public]
get_file_paths_by_device(...): Function to group raw data file paths according to the device
extract_sensor_from_filename(...): Extracts the sensor name from the filename of the sensor data
extract_device_from_filename(...): Extracts the device name from the raw data filename
-------------------

[Private]
-------------------
"""


# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import re
from typing import Union, Dict, List, Optional

# internal imports
from constants import WATCH, PHONE, ANDROID, ANDROID_WEAR, AVAILABLE_ANDROID_PREFIXES, AVAILABLE_ANDROID_SENSORS


# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
MIN_BYTES = 1500
STUDIO_DATA = 'StudioData'

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def get_file_paths_by_device(folder_path: Union[str, os.PathLike]) -> Dict[str, List[str]]:
    """
    Scans a folder and groups raw sensor data file paths by the device.

    This function iterates through all sensor data files in the specified folder and organizes them by device.
    Each file typically represents data from a specific sensor (e.g., accelerometer, gyroscope) on a particular device.
    Files that do not match a known device (e.g., logger files) are ignored.

    This function ignores the sensor files that are either empty or only have the header.

    The result is a dictionary where:
    - Keys are device identifiers (e.g., "phone", "watch", or a MuscleBan MAC address).
    - Values are lists of full file paths corresponding to sensor data from that device.

    :param folder_path: Path to the folder containing the sensor data files
    :return: A dictionary mapping each device to a list of its corresponding sensor data file paths.
    """

    # innit dictionary to store the paths
    paths_dict = {}

    try:
        # list the files in folder_path
        files = os.listdir(folder_path)
    except FileNotFoundError:

        # raise error in case the folder path is invalid
        raise ValueError(f"The folder at path {folder_path} was not found.")

    # iterate through the files inside the folder
    for filename in files:

        # ignore mvc
        if STUDIO_DATA in filename:
            continue

        # If less than 1 kb than it's either empty or only has the header - ignore
        if os.path.getsize(os.path.join(folder_path, filename)) <= MIN_BYTES:
            continue

        # check the device based on the filename
        device = extract_device_from_filename(filename)

        # found logger file
        if device is None:
            continue

        # add device to dictionary
        if device not in paths_dict.keys():

            # add dict entry
            paths_dict[device] = [filename]

        else:
            # if device is already on the dictionary, add filepath
            paths_dict[device].append(filename)

    return paths_dict


def extract_sensor_from_filename(filename: str) -> str:
    """
    Extracts the sensor name based on the filename. Works only for sensor data acquired  using the OpenSignals
    application.

    :param filename: A str with the filename
    :return: The sensor prefix based on the sensor name found on the filename
    """

    # iterate through the sensor file prefixes and sensor names
    for sensor_prefix, sensor_name in zip(AVAILABLE_ANDROID_PREFIXES, AVAILABLE_ANDROID_SENSORS):

        # find the prefix in the filename
        if sensor_prefix in filename:

            # get sensor prefix ( ex.: "ACCELEROMETER" -> ACC)
            return sensor_name

    raise ValueError(f"No valid sensor found in filename: {filename}")


def extract_device_from_filename(filename: str) -> Optional[str]:
    """
    Extracts the device name from a sensor data filename.

    This function identifies the device used to collect the data based on the filename format used by OpenSignals.
    It can detect files from a smartphone, smartwatch, or MuscleBan device.

    - Returns "phone" if the filename indicates a smartphone.
    - Returns "watch" if the filename indicates a smartwatch.
    - Returns the MAC address string if a MuscleBan device is detected (identified the mac address).

    :param filename: str corresponding to the filename
    :return: str containing the device name or None if no device is found
    """

    # check for smartwatch file
    if ANDROID_WEAR in filename:

        return WATCH

    # check for smartphone file
    elif ANDROID in filename:

        return PHONE

    # check for MBan file by mac address - exactly 12 uppercase letters or digits
    elif match := re.search(r'[A-Z0-9]{12}', filename):

        # return the mac address string
        return match.group()

    else:

        # found the logger file
        return None


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #