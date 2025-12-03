"""
Functions for interpolating android sensor data to obtain equidistant sampling.

Available Functions
-------------------
[Public]
cubic_spline_interpolation(...): Apply cubic spline interpolation to resample sensor data at a given frequency.
slerp_interpolation(...): Perform SLERP (Spherical Linear Interpolation) over a quaternion time series.
------------------
[Private]
_convert_android_timestamp_to_seconds(...): Converts the time column from the android timestamp which is in nanoseconds to seconds.
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import resample_poly
from constants import TIME_COLUMN_NAME
import math
# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #
# minimum difference between two HR instances (time)
MIN_HR_DIFF = 2


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def cubic_spline_interpolation(sensor_df: pd.DataFrame, fs: int = 100) -> pd.DataFrame:
    """
    Apply cubic spline interpolation to resample sensor data at a given frequency.
    This function interpolates time-series sensor data using cubic splines. The first column of `sensor_df` is assumed
    to be the time axis, while the remaining columns contain sensor measurements.

    :param sensor_df: A DataFrame containing timestamps in the first column and sensor data in the remaining columns.
    :param fs: The target sampling frequency in Hz. Default: 100 (Hz)
    :return: A DataFrame containing the resampled time axis and interpolated sensor values.
    """

    # extract time axis
    time_axis = sensor_df.iloc[:, 0]

    # convert time axis to seconds (and cast to numpy.array)
    time_axis = _convert_android_timestamp_to_seconds(time_axis).values

    # extract signals (and cast to numpy.array
    signals = sensor_df.iloc[:, 1:].values

    # define the new time axis
    time_axis_inter = np.arange(time_axis[0], time_axis[-1], 1 / fs)

    # list for holding the interpolated signals
    interpolated_signals = []

    # cycle over the signal channels
    for channel in range(signals.shape[1]):
        # init cubic spline interpolator
        cubic_spline_interpolator = CubicSpline(time_axis, signals[:, channel], bc_type='natural')

        # interpolate the signal and append to interpolated signals
        interpolated_signals.append(cubic_spline_interpolator(time_axis_inter))

    # create interpolated DataFrame
    interpolated_df = pd.DataFrame(np.column_stack([time_axis_inter] + interpolated_signals), columns=sensor_df.columns)

    return interpolated_df


def slerp_interpolation(rotvec_df: pd.DataFrame, fs: int = 100) -> pd.DataFrame:
    """
    Perform SLERP (Spherical Linear Interpolation) over a quaternion time series.
    This function interpolates a given time series of quaternions using SLERP, resampling it
    to match a specified target sampling frequency.

    :param rotvec_df: A DataFrame containing timestamp values in the first column and quaternion
                      components (x, y, z, w) in the subsequent columns.
    :param fs: The target sampling frequency in Hz. Default: 100 (Hz)
    :return: A DataFrame containing the interpolated timestamps and quaternions.
    """

    # extract time axis
    time_axis = rotvec_df.iloc[:, 0]

    # convert time axis to seconds (and cast to numpy.array)
    time_axis = _convert_android_timestamp_to_seconds(time_axis).values

    # get the quaterion data
    quaternion_data = rotvec_df.iloc[:, 1:].values

    # convert quaternions to Rotation objects
    rotations = R.from_quat(quaternion_data)

    # define new time axis
    time_axis_inter = np.arange(time_axis[0], time_axis[-1], 1 / fs)

    # init SLERP interpolator
    slerp_interpolator = Slerp(time_axis, rotations)

    # interpolate the rotations
    interpolated_rotations = slerp_interpolator(time_axis_inter)

    # create interpolated DataFrame (stack time axis and quaternion data + convert result back to quaterions)
    rotvec_interpolated_df = pd.DataFrame(np.column_stack((time_axis_inter, interpolated_rotations.as_quat())),
                                          columns=rotvec_df.columns)

    return rotvec_interpolated_df


def zero_order_hold_interpolation(sensor_df: pd.DataFrame, fs: int = 100) -> pd.DataFrame:
    """
    Apply zero order hold interpolation (repeats the previous value) to resample sensor data at a given frequency.
    This function interpolates time-series sensor data by repeating the previous value. The first column of `sensor_df`
    is assumed to be the time axis, while the remaining column contain sensor measurements.

    :param sensor_df: A DataFrame containing timestamps in the first column and HR sensor data in the remaining column
    :param fs: The target sampling frequency in Hz. Default: 100 (Hz)
    :return: A DataFrame containing the resampled time axis and interpolated sensor values.
    """
    # extract time axis
    time_axis = sensor_df.iloc[:, 0]

    # convert time axis to seconds (and cast to numpy.array)
    time_axis = _convert_android_timestamp_to_seconds(time_axis).values

    # extract signals (and cast to numpy.array
    signals = sensor_df.iloc[:, 1:].values

    # define the new time axis
    time_axis_inter = np.arange(time_axis[0], time_axis[-1], 1 / fs)

    # list for holding the interpolated signals
    interpolated_signals = []

    # cycle over the signal channels
    for channel in range(signals.shape[1]):

        # init zero hold interpolator
        zero_order_hold_interpolator = interp1d(time_axis, signals[:, channel], kind='previous')

        # interpolate the signal and append to interpolated signals
        interpolated_signals.append(zero_order_hold_interpolator(time_axis_inter))

    # create interpolated DataFrame
    interpolated_df = pd.DataFrame(np.column_stack([time_axis_inter] + interpolated_signals), columns=sensor_df.columns)

    return interpolated_df


def interpolate_heart_rate_sensor(sensor_df: pd.DataFrame, fs: int = 100) -> pd.DataFrame:
    """
    Function to interpolate heart rate data from a smartwatch sensor. The sensor acquires data in segments of approximately
    1 minute at 1 Hz, followed by a 3-minute pause, giving the battery limits of the device.

    This function extracts the segments during which the sensor was actively acquiring data and applies zero-order hold
    interpolation (repeats the previous value) to resample the signal to fs Hz (default: 100 Hz).
    All interpolated segments are then merged into a single DataFrame.

    :param sensor_df: A DataFrame containing timestamps in the first column and HR sensor data in the remaining column
    :param fs: The target sampling frequency in Hz. Default: 100 (Hz)
    :return: A DataFrame containing the interpolated timestamps and heart rate data.
    """
    # list for holding the interpolated segments of the HR sensor
    interpolated_segments = []

    # extract time axis
    time_axis = sensor_df.iloc[:, 0]

    # convert time axis to seconds (and cast to numpy.array)
    time_axis = _convert_android_timestamp_to_seconds(time_axis).values

    # the HR sensor acquires for approx 1 minute and stops for the next 3
    # find where the indices of when the sensor stopped acquiring - where the difference is not 1
    breaks = np.where(np.diff(time_axis) > MIN_HR_DIFF)[0]

    # get all start indices - inserts the index 0 at the beginning and the start
    start_indices = np.insert(breaks + 1, 0, 0)

    # get all stop indies -> len(time_axis) -1 adds the last index as a stop
    stop_indices = np.append(breaks, len(time_axis) - 1)

    # cycle over the start and stop indices
    for start, stop in zip(start_indices, stop_indices):

        # get the segment of the HR sensor (+1 to include the stop)
        hr_segment_df = sensor_df.iloc[start:stop+1]

        # get only the segment of the full tima axis (+1 to include the stop)
        time_axis_segment = time_axis[start:stop+1]

        # if segments have less than 2 samples, can not interpolate
        if len(hr_segment_df) < 2:
            continue

        time_axis_segment[0] = np.floor(time_axis_segment[0])

        # define the new time axis
        time_axis_inter = np.arange(time_axis_segment[0], time_axis_segment[-1], 1 / fs)

        # init zero hold interpolator
        zero_order_hold_interpolator = interp1d(time_axis_segment, hr_segment_df.values[:, 1], kind='previous')

        # interpolate HR data
        interpolated_hr_data = zero_order_hold_interpolator(time_axis_inter)

        # create interpolated DataFrame (stack time axis and sensor data)
        interpolated_segment_df = pd.DataFrame(np.column_stack((time_axis_inter, interpolated_hr_data)),
                                              columns=sensor_df.columns)

        # append to list of segments
        interpolated_segments.append(interpolated_segment_df)

    # concat all segments into one df
    interpolated_df = pd.concat(interpolated_segments, ignore_index=True)

    return interpolated_df


def resample_signals(sensor_df: pd.DataFrame, fs, fs_new) -> pd.DataFrame:
    """
    Function to resample signals using polyphase filtering. If fs_new > fs, the function upsamples the signal.
    If fs_new < fs, this function downsamples the signals. This function also generates a time axis in seconds based
    on the length of the signals and on the new sampling frequency. The first column of sensor_df is considered to be
    a time-axis (or a index column) and the remaining columns are considered signals.

    :param sensor_df: A DataFrame containing timestamps or indices in first column and sensor data in the remaining columns.
    :param fs: The original sampling frequency.
    :param fs_new: The target sampling frequency in Hz.
    :return: A DataFrame where the first column is the timestamps in seconds and the remaining are resampled data.
    """

    # list for holding the resampled signals
    resampled_signals = []

    # extract signals (and cast to numpy.array)
    signals = sensor_df.iloc[:, 1:].values

    # calculate resampling factor based on original fs and the new fs
    if fs > fs_new:
        factor = round(fs/fs_new)
        resample = 'down'

    else:
        factor = round(fs_new/fs)
        resample = 'up'

    # cycle over the signal channels
    for channel in range(signals.shape[1]):

        if resample == 'up':

            # upsample signals using polyphase filtering
            resampled_signal = resample_poly(signals[:, channel], up=factor, down=1)

        else:

            # downsample signals using polyphase filtering
            resampled_signal = resample_poly(signals[:, channel], up=1, down=factor)

        # append to resample signals
        resampled_signals.append(resampled_signal)

    # generate new time axis with the new sampling frequency
    time_axis_inter = _generate_time_column_from_samples(resampled_signals[0].shape[0], fs_new)

    # create interpolated DataFrame and change time column name
    resampled_df = pd.DataFrame(np.column_stack([time_axis_inter] + resampled_signals),
                                   columns=[TIME_COLUMN_NAME] + list(sensor_df.columns[1:]))

    return resampled_df

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _convert_android_timestamp_to_seconds(time_column: pd.Series) -> pd.Series:
    """
    Converts the time column from the android timestamp which is in nanoseconds to seconds.

    :param time_column: The column of the sensor dataframe containing the time axis
    :return: the converted time axis as pd.Series.
    """

    # get the first entry of the column
    time_0 = time_column.iloc[0]

    # subtract the first entry from the entire column
    time_column = time_column - time_0

    # convert from nanoseconds to seconds
    time_column = time_column * 1e-9

    return time_column

def _generate_time_column_from_samples(signal_size:int, fs: int):

    # get time (seconds) between each sample
    delta_t = 1/fs

    # generate time column in seconds
    time_column = np.arange(signal_size) * delta_t

    return time_column

