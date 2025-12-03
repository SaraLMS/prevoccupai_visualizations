"""
Functions to load and parse logger files in order to extract the initial data collection timestamps for each device
(e.g., smartwatch, smartphone, or MuscleBan sensor).

Available Functions
-------------------
[Public]
load_logger_file_info(...): Parses a logger file and extracts the data collection start timestamp for each device.
-------------------

[Private]
_check_logger_file(...): Checks whether a valid logger file exists in the folder.
_filter_logger_file(...): Filters the logger DataFrame to include only rows relevant to device start times.
_get_device_start_time(...): Retrieves the data collection start timestamp for a given device.
_find_android_logger_timestamps(...): Finds the initial timestamp for Android devices (WATCH or PHONE).
_find_mban_logger_timestamps(...): Finds the last recorded timestamp for MuscleBan devices.
-------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import pandas as pd
from typing import Optional, Dict, Set
import re

# internal imports
from constants import WATCH, PHONE
from load.parser import extract_device_from_filename
from visualize.parser import get_device_filename_timestamp

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
TIMESTAMP = 'timestamp'
LOG = 'log'
LOGGER_FILE_COLUMNS = [TIMESTAMP, LOG]
WATCH_IDENTIFIER = 'WEAR'
NOISE_RECORDER = 'NOISERECORDER'

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def load_logger_file_info(folder_path: str) -> Optional[Dict[str, str]]:
    """
    Parses a logger file and extracts the initial data collection timestamp for each detected device in a given folder.

    This function scans the contents of the specified folder to:
    1. Identify which devices are present inside the folder (based on filenames).
    2. Locate the logger file, which contains log messages with timestamps.
    3. Load and filter the logger file to isolate entries related to device data collection.
    4. Determine the start time of data collection for each detected device, based on log entries.

    The function supports both Android devices (WATCH and PHONE) and MuscleBan (Plux Wireless Biosignals).

    :param folder_path: Path to the folder containing the logger file and sensor data files.
    :return: A dictionary mapping each detected device to the timestamp of its data collection start,
             or None if no logger file is found.
             Example: {"WATCH": "10:15:23.123", "F0A55C68B2E1": "10:16:45.987"}
    """
    # innit dictionary to store the start times of the devices
    start_times_dict: Dict[str, str] = {}

    # check which devices are inside the folder
    detected_devices: Set[str] = set()

    # variable for holding the logger filepath
    logger_filepath: Optional[str] = None

    # (1) identify the devices based on the filename and locate the logger file
    for filename in os.listdir(folder_path):

        # check if it found a device
        device = extract_device_from_filename(filename)

        # found a device, add to set
        if device:
            detected_devices.add(device)

         # found the logger file
        else:
            logger_filepath = os.path.join(folder_path, filename)

    # (2) load and filter the logger file
    # load raw logger file into dataframe
    logger_df = pd.read_csv(logger_filepath, sep='\t', header=None, skiprows=3, names=LOGGER_FILE_COLUMNS)

    # filter dataframe, to get only the timestamps for each device and sensor
    logger_df = _filter_logger_file(logger_df)

    # (3) Find the start time of the data collection for each detected device, based on the log entries
    for device in detected_devices:

        # get the start time for the device and store in a single entry dictionary
        device_start_times_dict = _get_device_start_time(logger_df, device)

        # check if any of the timestamps was not found - might not be on the logger file
        # check if any value has ''
        if any(timestamp == '' for timestamp in device_start_times_dict.values()):

            # get all timestamps from the folder names
            folder_device_start_times_dict = get_device_filename_timestamp(folder_path)

            # fill in only the missing timestamps (those with '')
            for device_name, timestamp in device_start_times_dict.items():

                # get the device that has a missing timestamp and fill it with the folder_name timestamp
                if timestamp == '' and device_name in folder_device_start_times_dict:
                    device_start_times_dict[device_name] = folder_device_start_times_dict[device_name]

        # update dictionary
        start_times_dict.update(device_start_times_dict)

    return start_times_dict


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _filter_logger_file(logger_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the logger DataFrame to retain only entries indicating the first data reception from each sensor/device.

    This function assumes the input DataFrame has two columns:
    - The first column contains timestamps.
    - The second column (identified by the constant `LOG`) contains log messages.

    It selects only the rows where the log message starts with:
        "SENSOR_DATA: received first data from"
    These entries indicate the initial timestamp of data received from a sensor/device.

    After filtering, the function removes the prefix
        "SENSOR_DATA: received first data from"
    from each log entry, leaving only the name of the corresponding sensor/device
    (e.g., "SENSOR_DATA: received first data from WEAR_ACCELEROMETER" becomes "WEAR_ACCELEROMETER").

    :param logger_df: A pandas DataFrame containing timestamped log messages.
    :return: The filtered dataframe
    """

    # keep only the rows that have : "SENSOR_DATA: received first data from"
    logger_df = logger_df[logger_df[LOG].str.contains("SENSOR_DATA: received first data from")]

    # remove the prefix, keeping only the device name on the log column
    logger_df.loc[:, LOG] = logger_df[LOG].str.replace("SENSOR_DATA: received first data from", "",
                                                                 regex=False)

    return logger_df


def _get_device_start_time(logger_df: pd.DataFrame, device: str) -> Dict[str, str]:
    """
    Retrieves the initial timestamp indicating when the specified device began collecting data,
    based on entries in the logger DataFrame.

    For Android devices, the function returns the timestamp of the first sensor that began acquiring data.
    For MuscleBan devices, which may reconnect multiple times, it returns the timestamp of the last recorded entry.

    This function assumes the input DataFrame contains:
    - A first column with timestamps.
    - A second column with log messages.

    Returns a dictionary containing one entry:
    - Key: The name of the device.
    - Value: The corresponding start timestamp from the logger file.

    :param logger_df: A pandas DataFrame containing timestamped log messages.
    :param device: The name of the device. Valid devices are: WATCH, PHONE, and any muscleban mac address without the colons.
    :return:
    """
    # search for the initial timestamp depending on the device
    if device == WATCH:

        # search for the first entry with "WEAR"
        start_times_dict = _find_android_logger_timestamps(logger_df, watch=True)

    elif device == PHONE:

        # search for the first entry with "NOISERECORDER"
        start_times_dict = _find_android_logger_timestamps(logger_df)

    # found muscleban
    else:

        # find the LAST entry with the device mac address
        start_times_dict = _find_mban_logger_timestamps(logger_df, device)

    # remove last 4 characters of the timestamp string (e.g., ".000") and replace hh:mm:ss to hh-mm-ss
    start_times_dict[device] = start_times_dict[device][:-4].replace(":", "-")

    return start_times_dict


def _find_android_logger_timestamps(filtered_logger_df: pd.DataFrame, watch: bool = False) -> Dict[str, str]:
    """
    Extracts the initial timestamp indicating when an Android device (watch or phone) began collecting data,
    based on a filtered logger DataFrame.

    This function assumes the logger DataFrame has already been filtered to contain only relevant log entries,
    such as those related to the start of data collection for each sensor.

    - If `watch` is False (i.e., the device is a phone), the function searches for the first log entry containing
      the phone-specific identifier (NOISE_RECORDER).
    - If `watch` is True, it searches for the first log entry containing the watch-specific identifier
      (`WATCH_IDENTIFIER`).

    :param filtered_logger_df: A pandas DataFrame containing the timestamps and associated filtered log messages.
    :param watch: Boolean flag indicating whether the device is a watch (True) or a phone (False). Default = False
    :return: A dictionary with a single entry: {device_name: start_timestamp}, where the timestamp is the first
             occurrence of the device starting data collection.
    """

    # init dictionary to store the start times for each device
    start_times_dict: Dict[str, str] = {}

    # flags to check if the device log has been found - there are multiple logs for the same device
    found_device: bool = False

    # if device is phone - search for noise recorder in the logs
    if not watch:
        identifier = NOISE_RECORDER
        device = PHONE

    # if it's watch search for WEAR
    else:
        identifier = WATCH_IDENTIFIER
        device = WATCH

    # loop over the logs until the device timestamp is found
    i=0
    while i < len(filtered_logger_df) and not found_device:

        row = filtered_logger_df.iloc[i]

        # check for the first wear entry on the logger file
        if not found_device and identifier in row[LOG]:

            # update flag
            found_device = True

            # save watch start time
            start_times_dict[device] = row[TIMESTAMP]

        i+=1

    # if the timestamp is not found on the logger file put empty str
    if not found_device:
        start_times_dict[device] = ''

    return  start_times_dict


def _find_mban_logger_timestamps(filtered_logger_df: pd.DataFrame, device: str) -> Dict[str, str]:
    """
    Extracts the timestamp of the most recent connection event for a MuscleBan (MBAN) device,
    based on the provided logger DataFrame.

    MuscleBan devices may disconnect and reconnect multiple times during a session.
    To reliably determine the actual start time of data collection, this function searches for the
    last occurrence of the device's MAC address in the log entries.

    The function looks for MAC address patterns (format: XX:XX:XX:XX:XX:XX) within the log messages, strips
    the colons from the detected address, and compares it to the provided `device` string.

    :param filtered_logger_df: A pandas DataFrame containing the timestamps and associated filtered log messages.
    :param device: The MAC address of the MBAN device (with colons removed), used to identify its log entry.
    :return: A dictionary with one entry: {device: timestamp} where the timestamp is the last occurrence
             of the device’s MAC address in the log.
    """
    # flag to check if device was found
    found_device: bool = False

    # init dictionary to store the start times for each device
    start_times_dict: Dict[str, str] = {}

    # Reverse the DataFrame to find the last occurrence
    reversed_df = filtered_logger_df.iloc[::-1]

    # search for rows with a mac address: XX:XX:XX:XX:XX:XX (X are numbers or upper case letters)
    mac_pattern = r"\b(?:[0-9A-F]{2}:){5}[0-9A-F]{2}\b"

    for _, row in reversed_df.iterrows():

        # Search for MAC address in log entry
        if match := re.search(mac_pattern, row[LOG]):

            # remove colons since the device name provided does not have the columns
            mac_stripped = match.group().replace(":", "")

            # Compare to input device
            if mac_stripped == device:

                # update flag
                found_device = True

                # add timestamp to dict
                start_times_dict[device] = row[TIMESTAMP]

    # if the timestamp is not found on the logger file put empty str
    if not found_device:
        start_times_dict[device] = ''

    return start_times_dict
