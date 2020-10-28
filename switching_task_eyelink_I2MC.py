# -*- coding: utf-8 -*-
"""
Created on Wed Oct 07 19:34:00 2020

@author: Sebastián Moyano

PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

Description:
Set of functions to import txt data files from EyeLink, apply I2MC function
and compute scores for the switching task based on Kóvacs & Mehler (2009).
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import os
# I2MC function could be found in Roy Hessels GitHub
from usethis_I2MC_functions import *

# =============================================================================
# SET PARAMETERS
# =============================================================================

# load directory file
directory_boxes_block1 = 'C:/...'
directory_boxes_block2 = 'C:/...'
directory_reactive_block1 = 'C:/...'
directory_reactive_block2 = 'C:/...'

# directory to save results
directory_saving_results = 'C:/...'

# dictionary grouping directory by blocks
dict_boxes_blocks = {'block1': [directory_boxes_block1, 'target_loc_block_1'],
                     'block2': [directory_boxes_block2, 'target_loc_block_2']}

# specify parameters
options = {'xres': 1920,                        # screen resolution x axis
           'yres': 1080,                        # screen resolution y axis
           'freq': 500,                         # sampling rate
           'missingx': 99999,                   # missing values x axis
           'missingy': 99999,                   # missing values y axis
           'scrSz': [52, 30],                   # screen size (cm)
           'disttoscreen': 60,                  # distance to screen (cm)
           'minFixDur': 100}                    # fixation minimum duration

columns = {'subject': 'RECORDING_SESSION_LABEL',        # participant's ID column name
           'trial_start_time': 'TRIAL_START_TIME',      # trial start time column name
           'sample_time': 'TIME',                       # sample time column name
           'timestamp': 'TIMESTAMP',                    # timestamp column name
           'right_gaze_x': 'RIGHT_GAZE_X',              # x axis coordinates right gaze column name
           'right_gaze_y': 'RIGHT_GAZE_Y',              # y axis coordinates right gaze column name
           'left_gaze_x': 'LEFT_GAZE_X',                # x axis coordinates left gaze column name
           'left_gaze_y': 'LEFT_GAZE_Y',                # y axis coordinates left gaze column name
           'trial': 'TRIAL_INDEX',                      # trial column name
           'target_loc_block_1': 'target_side_1',       # target location first block column name
           'target_loc_block_2': 'target_side_2',       # target location second block column name
           'anticipations': 'ANTICIPATIONS'}            # others

AOIs = {'x_left_AOI': [0, 764],         # coordinates for x coordinates left AOI
        'y_left_AOI': [0, 1080],        # coordinates for y coordinates left AOI
        'x_right_AOI': [1156, 1920],    # coordinates for x coordinates right AOI
        'y_right_AOI': [0, 1080],       # coordinates for y coordinates right AOI
        'x_central_AOI': [764, 1156],   # coordinates for x coordinates central AOI
        'y_central_AOI': [0, 1080]}     # coordinates for y coordinates central AOI

# =============================================================================
# IMPORT FUNCTION (EYELINK 1000 PLUS)
# =============================================================================


def import_eyelink1000Plus(fname):

    """
    Imports txt files as Pandas DataFrame and creates the Timestamp

    Input:
        fname: directory with name of the file

    Output:
        DataFrame with loaded data for a single participant

    Notes:
        Eyelink Support: once the distance is beyond a range that allows for the
        distance to be recorded, you will end up with the value -3276.8, rather
        than just a dot (empty field). However, x and y gaze coordinates are
        reported as missing values "."

        If R or L_IN_BLINK = 1, gaze position is reported as an empty field (Data
        Viewer outputs empty fields as a dot by default), so all cells containing
        gaze-position-related values will be empty
    """

    loaded_data = pd.read_csv(fname, sep='\t', header=0, index_col=False, decimal=',')

    loaded_data[columns['sample_time']] = loaded_data[columns['timestamp']] - loaded_data[columns['trial_start_time']]

    return loaded_data

# =============================================================================
#  TRIAL SELECTION FUNCTION
# =============================================================================


def trial_selection(imported_data):

    """
    Split the data file per trial and apply the I2MC function before convert
    data into numpy arrays. At the end maps the position of the target for
    each trial.

    If X or Y gaze coordinates are more than 1 monitor away it is considered
    as missing data.

    Iterates over TRIAL_INDEX set. We create a set as we have multiple
    sample per trial. We need the list use get_group() to group by the
    data by trial (TRIAL_INDEX is an integer number, not string)

    Input:
        imported_data: DataFrame with loaded data for participant

    Output:
        f_fix: data with fixations information for a single file
        f_data: raw data used in the I2MC function for a single file
    """

    data = {}
    imported_data = imported_data.copy(deep=True)
    f_fix = None
    f_data = None
    t_fix = None
    t_data = None
    trials = set(imported_data[columns['trial']])
    min_trial = min(trials)

    for trial in trials:

        loaded_data = imported_data.groupby([columns['trial']]).get_group(trial)

        print('Currently working on trial: ' + str(trial))

        data['time'] = loaded_data[columns['sample_time']].to_numpy()    # timestamp
        data['L_X'] = loaded_data[columns['left_gaze_x']].to_numpy()     # left gaze x axis
        data['L_Y'] = loaded_data[columns['left_gaze_y']].to_numpy()     # left gaze y axis
        data['R_X'] = loaded_data[columns['right_gaze_x']].to_numpy()    # right gaze x axis
        data['R_Y'] = loaded_data[columns['right_gaze_y']].to_numpy()    # right gaze y axis
        data['trial'] = loaded_data[columns['trial']].to_numpy()         # trial
        
        # left eye
        lMiss1 = np.logical_or(data['L_X'] < -options['xres'], data['L_X'] > 2*options['xres'])
        lMiss2 = np.logical_or(data['L_Y'] < -options['yres'], data['L_Y'] > 2*options['yres'])
        lMiss = np.logical_or(lMiss1, lMiss2)
        data['L_X'][lMiss] = options['missingx']
        data['L_Y'][lMiss] = options['missingy']
        
        # right eye
        rMiss1 = np.logical_or(data['R_X'] < -options['xres'], data['R_X'] > 2*options['xres'])
        rMiss2 = np.logical_or(data['R_Y'] < -options['yres'], data['R_Y'] > 2*options['yres'])
        rMiss = np.logical_or(rMiss1, rMiss2)
        data['R_X'][rMiss] = options['missingx']
        data['R_Y'][rMiss] = options['missingy']
        
        I2MC_fix, I2MC_data, I2MC_options = I2MC(data, options)
        
        if trial == min_trial:
            f_fix = pd.DataFrame(I2MC_fix)
            f_data = pd.DataFrame(I2MC_data)
            f_fix['trial'] = trial
            f_data['trial'] = trial
        else:
            t_fix = pd.DataFrame(I2MC_fix)
            t_data = pd.DataFrame(I2MC_data)
            t_fix['trial'] = trial
            t_data['trial'] = trial
            f_fix = pd.concat([f_fix, t_fix], axis=0)
            f_data = pd.concat([f_data, t_data], axis=0)
                    
        t_fix = pd.DataFrame()
        t_data = pd.DataFrame()
        data = {}
        imported_data = imported_data.copy(deep=True)
        
    return f_fix, f_data

# =============================================================================
#  LOAD FILES
# =============================================================================


def load_files(directory):

    """
    Load all the files names into a list to use it in applyI2MCallfiles function

    Input:
        None

    Output:
        files: list of filenames in the folder
    """

    files = []
    for filename in os.listdir(directory):
        if os.path.isfile(filename) and filename.endswith(".txt") and filename not in files:
            files.append(filename)

    return files

# =============================================================================
#  APPLY FUNCTION I2MC TO ALL FILES IN THE DIRECTORY
# =============================================================================


def apply_I2MC_all_files_by_block(directory, block, target_side_block):

    """
    Import the data for a single participant through importEyelink1000Plus function.
    Apply I2MC function to every trial using trial_selection function, and adds
    the participant's code slicing from the filename.

    Iterate over all the files in the folder and apply functions. Concatenate
    results for all files in a single DataFrame.

    Input:
        directory: directory with files
        block (str): name of block to be analyzed
        target_side_block (str): name of the column containing the target side
                                 for each block from the columns dictionary

    Output:
        f_fixations: DataFrame with fixation data for all the files
        f_data: DataFrame with raw data used in the I2MC function for all the files
        files: list of files in the folder
    """

    # set working directory
    os.chdir(directory)

    # import files
    files = load_files(directory)
    f_fixations = None
    f_data = None

    # dictionaries
    dict_stim_position = {}
    dict_anticipations_count = {}
    dict_trial_max = {}

    for filename in files:
        fname = directory + '/' + filename
        imported_data = import_eyelink1000Plus(fname)

        print('Yay! Applying I2MC function to participant ' + filename)

        fixations, data = trial_selection(imported_data)

        if files[0] == filename:
            f_fixations = fixations
            f_data = data
            f_fixations['subject'] = filename[12:15]
            f_data['subject'] = filename[12:15]
        else:
            t_fixations = fixations
            t_data = data
            t_fixations['subject'] = filename[12:15]
            t_data['subject'] = filename[12:15]
            f_fixations = pd.concat([f_fixations, t_fixations], axis=0)
            f_data = pd.concat([f_data, t_data], axis=0)
        
        t_fixations = pd.DataFrame()
        t_data = pd.DataFrame()

        # dictionary with target side of each block per participant
        dict_stim_position[filename[12:15]] = imported_data[columns[target_side_block]].max()
        # dictionary with number of anticipations of each block per participant
        dict_anticipations_count[filename[12:15]] = imported_data[columns['anticipations']].max()
        # dictionary with trial max number of each block per participant
        dict_trial_max[filename[12:15]] = imported_data[columns['trial']].max()
        
    f_fixations['block'] = block
    f_fixations.reset_index(drop=True)
    
    return f_fixations, f_data, dict_stim_position, dict_anticipations_count, dict_trial_max, files

# =============================================================================
#  DETERMINE AOI FIXATION
# =============================================================================


def AOI_fixation(f_fixations):

    """
    Provides fixation (position 1 or position 2) per data sample.
    Uses data only if it is between 0 and 1920 on the x axis and
    between 0 and 1080 on the y axis.
    Left AOI is defined between 0 and 764 px. Right AOI between
    1156 and 1920

    Input:
        f_fixations: DataFrame with fixations data after I2MC function

    Output:
        df_fix_aoi: DataFrame with AOI fixations data added

    Notes:
        Left and right AOI cover all the y axis and part of the x axis.
    """

    # x and y coordinates inside screen
    condition_in_screen_xpos = (f_fixations['xpos'] > 0) & (f_fixations['xpos'] < options['xres'])
    condition_in_screen_ypos = (f_fixations['ypos'] > 0) & (f_fixations['ypos'] < options['yres'])
    # x and y coordinates for left AOI
    condition_in_left_xpos = (f_fixations['xpos'] > AOIs['x_left_AOI'][0]) & \
                             (f_fixations['xpos'] <= AOIs['x_left_AOI'][1])
    # x and y coordinates for right AOI
    condition_in_right_xpos = (f_fixations['xpos'] >= AOIs['x_right_AOI'][0]) & \
                              (f_fixations['xpos'] < AOIs['x_right_AOI'][1])
    # x and y coordinates for central AOI
    condition_in_central_xpos = (f_fixations['xpos'] > AOIs['x_central_AOI'][0]) & \
                                (f_fixations['xpos'] < AOIs['x_central_AOI'][1])
    # general conditions
    condition_in_screen = condition_in_screen_xpos & condition_in_screen_ypos
    # masks
    mask_in_left = condition_in_screen & condition_in_left_xpos
    mask_in_right = condition_in_screen & condition_in_right_xpos
    mask_in_central = condition_in_screen & condition_in_central_xpos
    # label fixations AOIs
    f_fixations['AOI_fix'] = ''
    f_fixations.loc[mask_in_left, 'AOI_fix'] = 1
    f_fixations.loc[mask_in_right, 'AOI_fix'] = 2
    f_fixations.loc[mask_in_central, 'AOI_fix'] = 0

    df_fix_aoi = f_fixations.copy(deep=True)

    return df_fix_aoi


# =============================================================================
#  REDUCE DATA
# =============================================================================


def reduce_data(df_fix_aoi):
    """
    Computes mean fixation duration and total fixation time from the number
    of fixations registered per trial and AOI.

    Input:
        df_fix_aoi: DataFrame with AOI fixations data

    Output:
        df_trial_reduced: reduced data with statistics computed
    """

    df_trial_reduced = df_fix_aoi.groupby(['subject', 'block', 'trial', 'AOI_fix']).agg(sum_fix=('dur', 'sum'),
                                                                                        mean_fix=('dur', 'mean'),
                                                                                        min_start_time=('startT', 'min'),
                                                                                        max_end_time=('endT', 'max'))
    df_trial_reduced.reset_index(drop=False, inplace=True)

    return df_trial_reduced


# =============================================================================
#  POSITION STIMULUS
# =============================================================================


def position_stimulus(df_trial_reduced, dict_stim_position):

    """
    Sets stimulus appearance location per trial.

    Input:
        df_trial_reduced: DataFrame with data reduced
        dict_stim_position: dictionary with stimulus position per participant

    Output:
        df_fix_aoi_reduced_loc: DataFrame with location of stimulus appearance per trial

    Notes:
        A dictionary is necessary as the start position per participant is
        counterbalanced
    """

    df_trial_reduced['position'] = df_trial_reduced['subject'].map(dict_stim_position)
    df_fix_aoi_position = df_trial_reduced.copy(deep=True)

    return df_fix_aoi_position

# =============================================================================
#  ACCURACY ANTICIPATORY FIXATIONS
# =============================================================================


def accuracy_anticipations(df_fix_aoi_position):
    
    """
    Labels as correct anticipations those anticipatory fixation
    recorded in the same location in which the stimulus is presented
    during the reactive period. Otherwise they are labeled as
    incorrect anticipations

    Input:
        df_fix_aoi_position: DataFrame with anticipatory fixations and
                             location in which the stimulus appears in
                             a column.

    Output:
        df_valid_anticipations: DataFrame with anticipatory fixations labeled
                                as correct or incorrect in a new column.
    """
    
    # conditions valid anticipation
    condition_correct_anticipation_right = (df_fix_aoi_position['AOI_fix'] == 2) & \
                                           (df_fix_aoi_position['position'] == 'Right')
    condition_correct_anticipation_left = (df_fix_aoi_position['AOI_fix'] == 1) & \
                                          (df_fix_aoi_position['position'] == 'Left')

    # conditions invalid anticipation
    condition_incorrect_anticipation_right = (df_fix_aoi_position['AOI_fix'] == 1) & \
                                             (df_fix_aoi_position['position'] == 'Right')
    condition_incorrect_anticipation_left = (df_fix_aoi_position['AOI_fix'] == 2) & \
                                            (df_fix_aoi_position['position'] == 'Left')
    # general conditions
    mask_correct_anticipation = condition_correct_anticipation_right | condition_correct_anticipation_left
    mask_incorrect_anticipation = condition_incorrect_anticipation_right | condition_incorrect_anticipation_left

    df_fix_aoi_position.loc[mask_correct_anticipation, 'correct_anticipation'] = 1
    df_fix_aoi_position.loc[mask_incorrect_anticipation, 'incorrect_anticipation'] = 1

    df_valid_anticipations = df_fix_aoi_position.copy(deep=True)
    
    return df_valid_anticipations

# =============================================================================
#  LABEL CORRECT OF INCORRECT ANTICIPATION IF OCCUR IN THE SAME TRIAL BASED
#  ON FIXATION DURATION
# =============================================================================


def choose_correct_or_incorrect_anticipation(df_valid_anticipations):

    """
    Sets final correct anticipation to 0 if the fixation duration for
    incorrect anticipation is higher than correct. Otherwise, sets final
    incorrect anticipation to 0. This conditions are only applied when
    both correct and incorrect anticipations are recorded, otherwise
    data is not modified.

    Splits the DataFrame into one for correct and other for incorrect
    anticipations. Keeps correct or incorrect anticipations, respectively.
    Merge both DataFrames to have columns for correct and incorrect
    anticipations and apply masks to the merged DataFrame.

    Input:
        df_valid_anticipations: DataFrame after applying accuracy_anticipations()

    Output:
        df_merged: DataFrame validating correct or incorrect anticipations based
                   on fixations durations
    """
    
    df_correct_anticipations = df_valid_anticipations.drop(labels=['incorrect_anticipation'], axis=1)
    df_incorrect_anticipations = df_valid_anticipations.drop(labels=['correct_anticipation'], axis=1)

    df_correct_anticipations = df_correct_anticipations[df_correct_anticipations['correct_anticipation'] == 1]
    df_incorrect_anticipations = df_incorrect_anticipations[df_incorrect_anticipations['incorrect_anticipation'] == 1]
    df_merged = df_correct_anticipations.merge(df_incorrect_anticipations, how='outer',
                                               on=['subject', 'trial', 'block', 'position'],
                                               suffixes=('_correct', '_incorrect'))

    # condition both correct and incorrect anticipation
    condition_correct_and_incorrect = (df_merged['correct_anticipation'] == 1) & \
                                      (df_merged['incorrect_anticipation'] == 1)

    # conditions correct or incorrect anticipation
    condition_correct_anticipation = df_merged['sum_fix_correct'] > df_merged['sum_fix_incorrect']
    condition_incorrect_anticipation = df_merged['sum_fix_correct'] < df_merged['sum_fix_incorrect']

    # masks
    condition_correct_valid = condition_correct_and_incorrect & condition_correct_anticipation
    condition_incorrect_valid = condition_correct_and_incorrect & condition_incorrect_anticipation

    # we already have in those conditions label 1
    df_merged.loc[condition_correct_valid, 'incorrect_anticipation'] = 0
    df_merged.loc[condition_incorrect_valid, 'correct_anticipation'] = 0
    
    return df_merged

# =============================================================================
#  ACCURACY REACTIVE FIXATIONS
# =============================================================================


def accuracy_reactive(reactive):

    """
    Sets correct reactive fixations to the stimulus location of appearance,
    and compute the number of correct fixations per participant. A dictionary
    with the number of reactive fixations per participant is returned for it
    use it in the statistics function.

    Input:
        reactive: DataFrame with fixations in the reactive period

    Output:
        df_reactive_completed: DataFrame with correct reactive fixations
                               in the reactive period and the number of
                               correct reactive fixations per participant
        dict_reactive_by_subject: dictionary with the number of reactive
                                  fixations per participant as values and
                                  participant's ID as keys.
    """

    reactive['correct_reactive'] = 0

    # conditions
    condition_reactive_correct_left = (reactive['AOI_fix'] == 1) & (reactive['position'] == 'Left')
    condition_reactive_correct_right = (reactive['AOI_fix'] == 2) & (reactive['position'] == 'Right')
    # general condition
    condition_reactive = condition_reactive_correct_left | condition_reactive_correct_right
    # label correct reactive looks
    reactive.loc[condition_reactive, 'correct_reactive'] = 1
    # count number of reactive looks by subject
    reactive_correct = reactive.groupby(['subject', 'block']).agg(n_reactive=('correct_reactive', 'sum'))
    reactive_correct.reset_index(drop=False, inplace=True)
    dict_reactive_by_subject = dict(zip(reactive_correct.subject, reactive_correct.n_reactive))

    df_reactive_completed = reactive.copy(deep=True)

    return df_reactive_completed, dict_reactive_by_subject


# =============================================================================
#  SHORT REPORT OF DATA
# =============================================================================


def short_report(df_merged, dict_anticipations_count, dict_trial_max, dict_stim_position, anticipations=True):

    """
    Count the number of correct and incorrect anticipations, as well as the
    mean duration of the correct and incorrect anticipations.

    If the data contains data of reactive fixations, computes the number of
    correct reactive fixations and the mena duration of reactive fixations.

    Input:
        df_merged: DataFrame after applying choose_correct_or_incorrect_anticipation()
                   function.
        dict_anticipations_count: dictionary with the number of anticipations registered
                                  with the ANTICIPATIONS variable by participant
                                  (IDs as keys, variable as values)
        dict_trial_max: dictionary with the number of maximum trials completed in the task
                        by participant (IDs as keys, variable as values)
        dict_stim_position: dictionary with the position in which the stimulus was presented
                            by participant (IDs as keys, variable as values)
        anticipations (boolean): True if the df_merged contains data about anticipatory fixations.
                                 False if the df_merged contains data about reactive fixations

    Output:
        df_short_report: DataFrame reducing the data per participant and block.

    """

    if anticipations:

        df_reduce_trials = df_merged.groupby(['subject', 'block']).agg(correct_anticipations=
                                                                       ('correct_anticipation', 'sum'),
                                                                       incorrect_anticipations=
                                                                       ('incorrect_anticipation', 'sum'),
                                                                       fix_mean_correct_anticipation=
                                                                       ('sum_fix_correct', 'mean'),
                                                                       fix_mean_incorrect_anticipation=
                                                                       ('sum_fix_incorrect', 'mean'))

        df_reduce_trials.reset_index(drop=False, inplace=True)

        df_reduce_trials['anticipations_count_variable'] = df_reduce_trials['subject'].map(dict_anticipations_count)
        df_reduce_trials['trial_max_completed'] = df_reduce_trials['subject'].map(dict_trial_max)
        df_reduce_trials['position'] = df_reduce_trials['subject'].map(dict_stim_position)

        df_short_report = df_reduce_trials

    elif not anticipations:

        df_reduce_trials = df_merged.groupby(['subject', 'block']).agg(correct_reactive=
                                                                       ('correct_reactive', 'sum'),
                                                                       fix_mean_correct_reactive=
                                                                       ('sum_fix', 'mean'))

        df_reduce_trials.reset_index(drop=False, inplace=True)

        df_reduce_trials['trial_max_completed'] = df_reduce_trials['subject'].map(dict_trial_max)
        df_reduce_trials['position'] = df_reduce_trials['subject'].map(dict_stim_position)

        df_short_report = df_reduce_trials

    return df_short_report

# =============================================================================
#  ANALYZE ANTICIPATIONS AND REACTIVE FIXATIONS (CALLS FUNCTIONS)
# =============================================================================


def analyze_anticipations():

    """
    Calls the necessary functions to process the data for anticipatory fixations

    Input:
        None
    Output:
        df_boxes_result_blocks: DataFrame with detailed data for correct and incorrect
                                anticipations for each block per participant
        df_short_report_boxes: Dataframe with reduced data for each block per participant
    """

    list_df_boxes = []
    list_df_reports = []

    for block_key, block_values in dict_boxes_blocks.items():

        # apply I2MC function to each block
        df_data_fixations, df_data_raw, dict_stim_position, dict_anticipations_count, dict_trial_max, files = \
            apply_I2MC_all_files_by_block(block_values[0], block_key, block_values[1])

        # get AOI for each fixation
        df_aoi_fixations_boxes = AOI_fixation(df_data_fixations)
        # get mean fixation duration and total fixation duration per trial and AOI
        df_trial_reduced_boxes = reduce_data(df_aoi_fixations_boxes)
        # get position of stimulus appearance per trial
        df_aoi_fixations_position_boxes = position_stimulus(df_trial_reduced_boxes, dict_stim_position)
        # get accuracy of anticipations
        df_trial_corr_incorr_boxes = accuracy_anticipations(df_aoi_fixations_position_boxes)
        # choose correct or incorrect anticipation when both are recorded in the same trial
        df_trial_corr_incorr_chosen_boxes = choose_correct_or_incorrect_anticipation(df_trial_corr_incorr_boxes)
        # df without reducing data
        list_df_boxes.append(df_trial_corr_incorr_chosen_boxes)
        df_boxes_result_blocks = pd.concat(list_df_boxes)
        # get short report
        df_report_boxes = short_report(df_trial_corr_incorr_chosen_boxes, dict_anticipations_count, dict_trial_max,
                                       dict_stim_position, anticipations=True)
        # short reports of both blocks
        list_df_reports.append(df_report_boxes)
        df_short_report_boxes = pd.concat(list_df_reports)

    return df_boxes_result_blocks, df_short_report_boxes


def analyze_reactive():

    """
    Calls the necessary functions to process the data for reactive fixations

    Input:
        None
    Output:
        df_reactive_result_blocks: DataFrame with detailed data for correct reactive
                                   fixations for each block per participant
        df_short_report_reactive: Dataframe with reduced data for each block per participant
    """

    list_df_reactive = []
    list_df_reports = []

    for block_key, block_values in dict_reactive_blocks.items():

        # apply I2MC function to each block
        df_data_fixations, df_data_raw, dict_stim_position, dict_anticipations_count, dict_trial_max, files = \
            apply_I2MC_all_files_by_block(block_values[0], block_key, block_values[1])

        # get AOI for each fixation
        df_aoi_fixations_reactive = AOI_fixation(df_data_fixations)
        # get mean fixation duration and total fixation duration per trial and AOI
        df_trial_reduced_reactive = reduce_data(df_aoi_fixations_reactive)
        # get position of stimulus appearance per trial
        df_aoi_fixations_position_reactive = position_stimulus(df_trial_reduced_reactive, dict_stim_position)
        # get accuracy of reactive
        df_trial_corr_reactive, dict_reactive_subject = accuracy_reactive(df_aoi_fixations_position_reactive)
        # df without reducing data
        list_df_reactive.append(df_trial_corr_reactive)
        df_reactive_result_blocks = pd.concat(list_df_reactive)
        # get short report
        df_report_reactive = short_report(df_trial_corr_reactive, dict_anticipations_count, dict_trial_max,
                                          dict_stim_position, anticipations=False)
        # short reports of both blocks
        list_df_reports.append(df_report_reactive)
        df_short_report_reactive = pd.concat(list_df_reports)

    return df_reactive_result_blocks, df_short_report_reactive


# =============================================================================
#  ANALYZE (CALLS FUNCTIONS TO ANALYZE ANTICIPATORY AND REACTIVE FIXATIONS)
# =============================================================================  
  
  
def analyze_data():

    """
    Call functions to analyze anticipatory and reactive fixations, creating
    a long and short report for each type of fixations. Merge anticipatory and
    reactive fixations creating a general long (one trial per row) and short
    report (one subject per row).

    Input:
        None
    Output:
        df_trials_long_report: DataFrame with long report.
        df_trials_short_report: DataFrame with long report.
    """

    # save the results in a new folder created in the same path
    results_dir = os.path.join(directory_saving_results, 'AnalyzedI2MC')
    # if the directory doesn't exist, this loop creates it
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    # analyze anticipatory fixations
    df_anticipations, df_anticipations_report = analyze_anticipations()
    # analyze reactive fixations
    df_reactive, df_reactive_report = analyze_reactive()
    # merge short reports (anticipations and reactive fixations)
    df_trials_short_report = df_anticipations_report.merge(df_reactive_report, on=['subject', 'block', 'position',
                                                                                   'trial_max_completed'])
    df_trials_short_report = pd.pivot_table(df_trials_short_report, index='subject', columns='block').reset_index()
    # join columns multiindex
    df_trials_short_report.columns = df_trials_short_report.columns.map(lambda x: '_'.join([str(i) for i in x]))
    # save reports
    df_trials_short_report.to_csv(os.path.join(results_dir, 'switching_short_report.txt'), sep=';', decimal=',')

    return df_anticipations, df_reactive, df_trials_short_report


  # code to call
df_long_report, df_short_report = analyze_data()














