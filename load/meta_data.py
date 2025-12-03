"""
Functions for loading the meta-data contained in 'subjects_info.csv'.

Available Functions
-------------------
[Public]
load_meta_data(): loads the meta-data contained in subjects_info.csv into a pandas.DataFrame.

------------------
[Private]

"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
from typing import List, Optional
from constants import PHONE, WATCH, MBAN_LEFT, MBAN_RIGHT

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def load_meta_data():
    """
    loads the meta-data contained in subjects_info.csv into a pandas.DataFrame.
    :return: DataFrame containing the meta-data
    """

    return pd.read_csv('subjects_info.csv', sep=';', encoding='utf-8', index_col='subject_id')


def get_muscleban_side(meta_data_df, mac_address):
    """
    Extracts the side of the muscleban from the meta_data_df based on the mac address of the device
    :param meta_data_df: pd.DataFrame containing the subject meta-data contained in subjects_info.csv
    :param mac_address: str containing the mac address without the colons
    :return: str containing the muscleban side
    """
    # Search in mBAN_left column
    if mac_address in meta_data_df[MBAN_LEFT].values:
        return MBAN_LEFT

    # Search in mBAN_right column
    elif mac_address in meta_data_df[MBAN_RIGHT].values:
        return MBAN_RIGHT

    # If not found
    return None


def get_expected_devices(meta_data_df, group: str, device_num: str) -> List[str]:

    # list with expected device - phone and watch are added manually as they are not on the metadata df
    expected_devices = [PHONE, WATCH]

    # Filter the DataFrame for the given group and device number
    row = meta_data_df[(meta_data_df['group'] == group) & (meta_data_df['device_num'] == device_num)]

    # get the mac address of the musclebans
    mban_left = row.iloc[0][MBAN_LEFT]
    mban_right = row.iloc[0][MBAN_RIGHT]

    # muscleban mac addresses to the list
    expected_devices.extend([mban_left, mban_right])

    return expected_devices
