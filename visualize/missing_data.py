"""
Functions to detect and reconstruct missing sensor acquisitions.

Available Functions
-------------------
[Public]
get_missing_data(...): Identify missing acquisition times and durations for each device (except the phone), based on expected daily acquisition patterns.
-------------------

[Private]
_has_close_time(...): Check whether a given timestamp is within the tolerance window of another timestamp in a list.
_get_missing_timestamps(...): Compare expected acquisition times with actual ones to determine which are missing.
_find_unique_timestamps(...): Extract unique acquisition timestamps across devices (excluding phone), accounting for tolerance in start times.
-------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# internal imports
from constants import PHONE, ACQUISITION_TIME_SECONDS
from utils import get_most_common_acquisition_times

# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
LENGTH = 'length'
START_TIMES = 'start_times'
END_TIMES = 'end_times'
TIME_FORMAT = "%H-%M-%S"
# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def get_missing_data(subject_folder_path: str, acquisitions_dict: Dict[str, Dict[str, list]], fs: int = 100,
                     tolerance_seconds: int = 600) -> Dict[str, Dict[str, list]]:
    """
    Identify and return missing data (start_time and length) for each device (except the phone).

    This function assumes that the smartwatch and two muscleBAN should all acquire data at the same time, four times
    per day. Due to potential connection issues, some acquisitions may be missing. This function determines which
    timestamps are missing for each device, based on what was recorded by the others.

    In the case that all devices failed to acquire in one of these four scheduled acquisitions, this function uses the
    average acquisition times (based on the common acquisition times during the five days of acquisition of the subject),
    to get the missing timestamp.

    Steps:
    (1) Obtain all unique timestamps of all devices, except the phone. As there is always some delay, timestamps that are
        less than ten minute apart are considered to be the same one.

    (2) Compare the start_times in the acquisitions_dict with the unique timestamps to get a list with missing timestamps
        for each device.

    (3) If the length of the actual start times and the missing start times is less than four then there was one scheduled
        acquisition that either did not happen or all devices failed. In this case find the last remaining missing timestamps
        based on the average start times of the entire week.

    (4) For each missing timestamp, a default duration (20-minute acquisition) is used to calculate the end_times of
    the acquisitions

    :param subject_folder_path: Path to the folder containing all data from the subject
    :param acquisitions_dict: A dictionary where keys are device names, and values are dictionaries with two lists:
             - 'length': List of signal lengths.
             - 'start_times': List of corresponding start timestamps.
             Example:
             {
                 "phone": {"start_times": ["11:20:20.000"], "end_times": ["11:40:20.000"]},
                 "watch": {"start_times": ["10:20:50.000", "12:00:00.000"], "end_times": ["10:40:50.000", "12:20:00.000"]}
             }
    :param fs: the sampling frequency of the sensors in Hz. Default = 100.
    :param tolerance_seconds: Time in seconds used to consider two start times as referring to the same acquisition.
                            Default = 600. (e.g., 12:00:00 and 12:03:00 are considered to be the same start time).
    :return: A dictionary in the same format as `acquisitions_dict`, but containing only the missing acquisitions
             detected for each device.
    """

    # dictionary to store the missing data with the same format as the dictionary storing the actual acquisitions
    missing_data_dict: Dict[str, Dict[str, list]] = {}

    # check if there are missing acquisitions
    for device, data in acquisitions_dict.items():

        # skip if it's phone
        if device == PHONE:
            continue

        # if any device that is not phone didn't acquire 4 times, then it's missing
        if  len(data[START_TIMES]) < 4:

            # (1) get the unique timestamps - all timestamps of the devices that should acquire at the same time
            unique_timestamps_list = _find_unique_timestamps(acquisitions_dict, tolerance_seconds)

            # (2) compare the actual timestamps with the unique to get missing timestamps for the device
            missing_times_list = _get_missing_timestamps(unique_timestamps_list, data[START_TIMES])

            # (3) check if there are still missing timestamps - no device connected for the scheduled acquisition
            if len(data[START_TIMES]) + len(missing_times_list) < 4:

                # create list with the actual timestamps and the missing timestamps found in get_missing_time_from_device
                temp_list = data[START_TIMES] + missing_times_list

                # Get the most common expected acquisition based on the average of all days
                average_times_list = get_most_common_acquisition_times(subject_folder_path, acquisitions_dict[PHONE][START_TIMES][0])

                # use the averages to get only the timestamps that are missing on both devices
                missing_times_list.extend(_get_missing_timestamps(average_times_list, temp_list))

                # handle the case where the acquisition time are so mismatched that data[START_TIMES] + missing_times_list > 4
                if len(data[START_TIMES]) + len(missing_times_list) > 4:

                    # iterate through the missing times list to remove the wrong ones
                    for missing_time in missing_times_list:

                        # there should be a difference of at leats 30 minutes between an actual acquisition time and the
                        # calculated missing time for it to be
                        if _has_close_time(datetime.strptime(missing_time, TIME_FORMAT),
                                           [datetime.strptime(time, TIME_FORMAT) for time in data[START_TIMES]],
                                           2000):

                            # remove 'fake' missing time from list
                            missing_times_list.remove(missing_time)

            # initialize if device not in dict
            if device not in missing_data_dict:
                missing_data_dict[device] = {
                    START_TIMES: [],
                    END_TIMES: []
                }

                # (4) append missing timestamps + end times
                # get time to add to the start time
                durations = [ACQUISITION_TIME_SECONDS] * len(missing_times_list)

                # calculate end time
                computed_ends = compute_end_times(missing_times_list, durations)

                # add to the nested dict
                missing_data_dict[device][START_TIMES].extend(missing_times_list)
                missing_data_dict[device][END_TIMES].extend(computed_ends)

    return missing_data_dict


def compute_end_times(start_times: List[Optional[str]], lengths_seconds: List[float]) -> List[Optional[str]]:
    """
    Computes end times given start_times and durations in seconds.

    """
    end_times = []

    for start, dur_sec in zip(start_times, lengths_seconds):
        if start is None:
            end_times.append(None)
            continue

        # parse HH:MM:SS or with milliseconds
        try:
            t0 = datetime.strptime(start, TIME_FORMAT)
        except ValueError:
            t0 = datetime.strptime(start, f"{TIME_FORMAT}.%f")

        t_end = (t0 + timedelta(seconds=dur_sec)).time()
        end_times.append(t_end.strftime(TIME_FORMAT))

    return end_times
# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _has_close_time(time: datetime, time_list_dt: List[datetime], tolerance_seconds: int) -> bool:
    """
    Check whether a given timestamp is within the tolerance window of any timestamp in a list.

    This function is used to determine if an acquisition time is "close enough"
    to another (i.e., represents the same scheduled acquisition).

    :param time: A datetime object to compare.
    :param time_list_dt: List of datetime objects representing existing acquisition times.
    :param tolerance_seconds: Time difference (in seconds) allowed for considering two times as the same.
    :return: True if `time` is within the tolerance of any timestamp in `time_list_dt`, otherwise False.
    """
    return any(abs((time - t).total_seconds()) <= tolerance_seconds for t in time_list_dt)


def _get_missing_timestamps(unique_timestamps_list: List[datetime], acquisitions_times_list: List[str],
                            tolerance_seconds=600) -> List[str]:
    """
    Identify which expected acquisition times are missing for a device.

    This function compares the unique expected acquisition times (found with the devices that acquired data) against the actual
    acquisition times recorded by a device (that had missing acquisitions), and returns those that are missing.

    :param unique_timestamps_list: List of datetime objects representing all expected acquisitions.
    :param acquisitions_times_list: List of acquisition start times (string format) for the device.
    :param tolerance_seconds: Allowed deviation (in seconds) for considering times as equal. Default = 600.
    :return: List of missing acquisition times (string format, TIME_FORMAT).
    """

    # innit list to store the missing times
    missing_times: List[str] = []

    # change the sensor start times to datetime
    device_timestamp_dt = [datetime.strptime(timestamp, TIME_FORMAT) for timestamp in acquisitions_times_list]

    # iterate through the unique timestamps
    for timestamp in unique_timestamps_list:

        # check if there is a timestamp that is NOT similar to the one in unique_timestamps_list
        if not _has_close_time(timestamp,device_timestamp_dt, tolerance_seconds):

            # add to the list with missing times in the correct format
            missing_times.append(timestamp.strftime(TIME_FORMAT))

    return missing_times


def _find_unique_timestamps(acquisitions_dict: Dict[str, Dict[str, list]], tolerance_seconds: int) -> List[datetime]:
    """
    Finds a set of start times that are expected for all devices, except the smartphone. this is done by getting the
    unique timestamps for all three devices (watch, mBAN right, and mBAN left), with a tolerance, since the devices don't start
    acquiring exactly at the same time.

    :param acquisitions_dict: Dictionary with device acquisition data.
                              Each entry contains 'start_times' and 'length'.
    :param tolerance_seconds: Allowed deviation (in seconds) for considering timestamps as the same acquisition.
    :return: A list of unique acquisition times (datetime objects).
    """

    # list for holding all timestamps found for the 3 devices that acquire at the same time
    all_daily_timestamps: List[datetime] = []

    for device, data in acquisitions_dict.items():

        # skip if its phone
        if device == PHONE:
            continue

        # change to datetime objects to perform mathematics
        acquisition_times_dt = [datetime.strptime(time, TIME_FORMAT) for time in data[START_TIMES]]
        all_daily_timestamps.extend(acquisition_times_dt)

    # since these devices don't start exactly at the same time, remove the timestamps that are very similar based on tolerance_seconds
    # list for holding the unique timestamps
    filtered_timestamps: List[datetime] = []

    # iterate through the sorted list
    for timestamp in sorted(all_daily_timestamps):

        # if list is empty add the first timestamp
        if not filtered_timestamps:

            filtered_timestamps.append(timestamp)

        else:
            # check if this timestamp is similar to the previous value add only if it's not
            if not _has_close_time(timestamp, filtered_timestamps, tolerance_seconds):
                filtered_timestamps.append(timestamp)


    return filtered_timestamps