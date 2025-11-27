# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import load
import os
from visualize.visualize_acquisitions import visualize_daily_acquisitions, visualize_week_acquisitions

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
VISUALIZE_DAY = False
VISUALIZE_WEEK = True
SUBJECT_FOLDER_PATH = "E:\\Backup PrevOccupAI_PLUS Data\\data\\group1\\sensors\\LIBPhys #001"
DATE = "2025-09-25"
# ------------------------------------------------------------------------------------------------------------------- #
# program starts here
# ------------------------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':

    # visualize the acquisitions of a single day
    if VISUALIZE_DAY:

        visualize_daily_acquisitions(SUBJECT_FOLDER_PATH, DATE, fs=100)

    # visualize all daily acquisitions from all subject of the WEEK
    if VISUALIZE_WEEK:

        visualize_week_acquisitions(SUBJECT_FOLDER_PATH)

    # data_dict = load.load_data_from_same_recording(os.path.join(GROUP_FOLDER_PATH, SUBJECT_NUM, DATE, '10-30-00'))