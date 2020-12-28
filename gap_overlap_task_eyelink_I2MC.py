# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:54:00 2019

@author: Sebastian Moyano
PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

Description:
Set of functions to import txt data files from EyeLink, apply I2MC function
and compute scores for the gap-overlap task.

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import os
# I2MC function could be found in Roy Hessels GitHub
from I2MC_functions import *

# =============================================================================
# SET PARAMETERS
# =============================================================================

# directory of each Sample Report
directory_OverlapTarget = 'C:/.../'
directory_OverlapCentral = 'C:/.../'
directory_GapTarget = 'C:/.../'
directory_GapCentral = 'C:/.../'

# directory to save results
directory_saving_results = 'C:/.../'

# rows to skip when loading data
nskip = 1

# specify parameters
options = {'xres': 1920,        # screen resolution x axis
           'yres': 1080,        # screen resolution y axis
           'freq': 500,         # sampling rate
           'missingx': 99999,   # missing values x axis
           'missingy': 99999,   # missing values y axis
           'scrSz': [52, 30],   # screen size (cm)
           'disttoscreen': 60,  # distance to screen (cm)
           'minFixDur': 100}    # fixation minimum duration

columns = {'trial_start_time': 'TRIAL_START_TIME',  # trial start time
           'sample_time': 'TIME',                   # recording time for each sample
           'timestamp': 'TIMESTAMP',                # timestamp (from the start of the trial, our files doesn't have it)
           'right_gaze_x': 'RIGHT_GAZE_X',          # x coordinates for right eye
           'right_gaze_y': 'RIGHT_GAZE_Y',          # y coordinates for right eye
           'left_gaze_x': 'LEFT_GAZE_X',            # x coordinates for left eye
           'left_gaze_y': 'LEFT_GAZE_Y',            # y coordinates for left eye
           'trial': 'TRIAL_INDEX',                  # trial index
           'target_loc': 'SIDE_CONDITION'}          # target location (if needed)

AOIs = {'x_left_AOI': [16, 616],         # coordinates for x coordinates left AOI
        'y_left_AOI': [140, 940],        # coordinates for y coordinates left AOI
        'x_right_AOI': [1302, 1902],     # coordinates for x coordinates right AOI
        'y_right_AOI': [140, 940],       # coordinates for y coordinates right AOI
        'x_central_AOI': [660, 1260],    # coordinates for x coordinates central AOI
        'y_central_AOI': [140, 940]}     # coordinates for y coordinates central AOI

# =============================================================================
# IMPORT FUNCTION (EYELINK 1000 PLUS)
# =============================================================================


def import_eyelink1000plus(fname):

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

    # load csv
    loaded_data = pd.read_csv(fname, sep='\t', header=0, index_col=False, decimal=',')
    # TIME provides the start of each sample from the Trial Start Time
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

    Iterates over TRIAL_INDEX list of set. We create a set as we have multiple
    sample per trial, and then create a list to iterate over. We need the list
    as we are using get_group() to groupby the data by trial (TRIAL_INDEX is an
    integer number, not string)

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
    dict_pos = {}
    # trials are not sequential. Create a set and then a list from set
    trials = sorted(list(set(imported_data['TRIAL_INDEX'])))  # sorted list of trials
    min_trial = min(trials)

    # iterate over trials
    for trial in trials:
        # we can use trial as it is an integer to get_group()
        loaded_data = imported_data.groupby([columns['trial']]).get_group(trial)
        current_trial = trial
        print('Currently working on trial: ' + str(trial))

        data['time'] = loaded_data[columns['sample_time']].to_numpy()    # timestamp
        data['L_X'] = loaded_data[columns['left_gaze_x']].to_numpy()     # left gaze x axis
        data['L_Y'] = loaded_data[columns['left_gaze_y']].to_numpy()     # left gaze y axis
        data['R_X'] = loaded_data[columns['right_gaze_x']].to_numpy()    # right gaze x axis
        data['R_Y'] = loaded_data[columns['right_gaze_y']].to_numpy()    # right gaze y axis
        data['trial'] = loaded_data[columns['trial']].to_numpy()         # trial
        data['position'] = loaded_data[columns['target_loc']]            # stimulus location

        # dictionary with target position by trial
        dict_pos[current_trial] = ''.join(loaded_data[columns['target_loc']].unique())
        
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
    
    f_fix['target_pos'] = f_fix['trial'].map(dict_pos)
    f_data['target_pos'] = f_data['trial'].map(dict_pos)
    
    return f_fix, f_data

# =============================================================================
#  LOAD FILES
# =============================================================================


def load_files(directory):

    """
    Load all the files names into a list to use it in applyI2MCallfiles function

    Input:
        directory: directory with files
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


def apply_I2MC_all_files(directory):

    """
    Import the data for a single participant through importEyelink1000Plus function.
    Apply I2MC function to every trial using trial_selection function, and adds
    the participant's code slicing from the filename.

    Iterate over all the files in the folder and apply functions. Concatenate
    results for all files in a single DataFrame.

    Input:
        directory: directory with files
    Output:
        f_fixations: DataFrame with fixation data for all the files
        f_data: DataFrame with raw data used in the I2MC function for all the files
        files: list of files in the folder
    """

    files = load_files(directory)
    f_fixations = None
    f_data = None

    for filename in files:
        fname = directory + '/' + filename
        imported_data = import_eyelink1000plus(fname)

        print('Yay! Applying I2MC function to participant ' + str(filename))

        fixations, data = trial_selection(imported_data)

        # if the file is the first in the list, create with that data f_fixations and f_data
        if files[0] == filename:
            f_fixations = fixations
            f_data = data
            f_fixations['subject'] = filename[16:24]    # change for different file name length
            f_data['subject'] = filename[16:24]
        # if the file is not the first in the list, concatenate to f_fixation and f_data
        else:
            t_fixations = fixations
            t_data = data
            t_fixations['subject'] = filename[16:24]
            t_data['subject'] = filename[16:24]
            f_fixations = pd.concat([f_fixations, t_fixations], axis=0)
            f_data = pd.concat([f_data, t_data], axis=0)
        
        t_fixations = pd.DataFrame()
        t_data = pd.DataFrame()
    
    f_fixations.reset_index(drop=True, inplace=True)
    
    return f_fixations, f_data, files

# =============================================================================
#  DETERMINE AOI FIXATION: TARGET
# =============================================================================


def AOI_fixations_target(df_fix):

    """
    Provides fixation location during target (location 1 [left], 2 [right], 0
    [central] or missing value) per data sample based on x and y coordinates.

    Input:
        df_fix: DataFrame with fixations information
    Output:
        df_fix_aoi: DataFrame with fixations information with fixation location
                    based on x and y coordinates
    """

    # x and y coordinates inside screen
    condition_inscreen_xpos = (df_fix['xpos'] > 0) & (df_fix['xpos'] < options['xres'])
    condition_inscreen_ypos = (df_fix['ypos'] > 0) & (df_fix['ypos'] < options['yres'])
    # x and y coordinates for left, right, central and y pos
    condition_in_left_xpos = (df_fix['xpos'] > AOIs['x_left_AOI'][0]) & \
                             (df_fix['xpos'] < AOIs['x_left_AOI'][1])
    condition_in_right_xpos = (df_fix['xpos'] > AOIs['x_right_AOI'][0]) & \
                              (df_fix['xpos'] < AOIs['x_right_AOI'][1])
    condition_in_central_xpos = (df_fix['xpos'] > AOIs['x_central_AOI'][0]) & \
                                (df_fix['xpos'] < AOIs['x_central_AOI'][1])
    # is the same in the three AOIs, let's just use one
    condition_in_ypos = (df_fix['ypos'] > AOIs['y_left_AOI'][0]) & \
                        (df_fix['ypos'] < AOIs['y_left_AOI'][1])
    # general conditions
    condition_in_screen = condition_inscreen_xpos & condition_inscreen_ypos

    # masks
    mask_left = condition_in_left_xpos & condition_in_ypos
    mask_right = condition_in_right_xpos & condition_in_ypos
    mask_central = condition_in_central_xpos & condition_in_ypos
    mask_outAOI = condition_in_screen & ~mask_central & ~mask_left & ~mask_right
    mask_missing = ~condition_in_screen

    # set values to 'AOI_fix' column based on met condition on each mask
    df_fix['AOI_fix'] = ''
    df_fix.loc[mask_left, 'AOI_fix'] = '1'
    df_fix.loc[mask_right, 'AOI_fix'] = '2'
    df_fix.loc[mask_central, 'AOI_fix'] = '0'
    df_fix.loc[mask_outAOI, 'AOI_fix'] = 'Out_AOIs'
    df_fix.loc[mask_missing, 'AOI_fix'] = 'Missing'

    df_fix_aoi = df_fix

    return df_fix_aoi

# =============================================================================
#  DETERMINE AOI FIXATION: CENTRAL
# =============================================================================


def AOI_fixations_central(df_fix):

    """
    Provides fixation locations during the central target (in central AOI,
    outside central AOI or missing value) per data sample based on x and y
    coordinates.

    Input:
        df_fix: DataFrame with fixations information
    Output:
        df_fix_aoi: DataFrame with fixations information with fixations
                    location based on x and y coordinates.
    """

    # x and y coordinates inside screen
    condition_in_screen_xpos = (df_fix['xpos'] > 0) & (df_fix['xpos'] < options['xres'])
    condition_in_screen_ypos = (df_fix['ypos'] > 0) & (df_fix['ypos'] < options['yres'])
    # x and y coordinates in central AOI, outside central AOI and inside ypos for AOIs
    condition_in_central_xpos = (df_fix['xpos'] > AOIs['x_central_AOI'][0]) & \
                                (df_fix['xpos'] < AOIs['x_central_AOI'][1])
    condition_in_ypos = (df_fix['ypos'] > AOIs['y_central_AOI'][0]) & \
                        (df_fix['ypos'] < AOIs['y_central_AOI'][1])
    # general conditions for inside, outside screen and central AOI
    condition_in_screen = condition_in_screen_xpos & condition_in_screen_ypos
    condition_in_central = condition_in_central_xpos & condition_in_ypos
    # masks
    mask_in_central = condition_in_screen & condition_in_central
    mask_out_central = condition_in_screen & ~condition_in_central
    mask_missing = ~condition_in_screen

    df_fix['AOI_fix'] = ''
    df_fix.loc[mask_in_central, 'AOI_fix'] = '0'
    df_fix.loc[mask_out_central, 'AOI_fix'] = 'Out_AOIs'
    df_fix.loc[mask_missing, 'AOI_fix'] = 'Missing'

    df_fix_aoi = df_fix

    return df_fix_aoi

# =============================================================================
#  GET CORRECT FIXATIONS TARGET
# =============================================================================


def valid_fix_target(df_fix_aoi):

    """
    Consider valid fixations in the target if the side of appearance is
    the same of the AOI in which the fixation is registered.

    Input:
        df_fix_aoi: DataFrame with data fixations and aoi fixations
    Output:
        df_fix_aoi_corr_target: DataFrame with data fixations and aoi fixations with new columns
                                indicating if the fixations was correct, incorrect of central
    """

    df_fix_aoi['fix_target_correct'] = 0
    df_fix_aoi['fix_target_incorrect'] = 0
    df_fix_aoi['fix_central_stim'] = 0
    # conditions
    valid_right = (df_fix_aoi['target_pos'] == 'right') & (df_fix_aoi['AOI_fix'] == '2')
    valid_left = (df_fix_aoi['target_pos'] == 'left') & (df_fix_aoi['AOI_fix'] == '1')
    valid_central = df_fix_aoi['AOI_fix'] == '0'
    invalid_right = ~valid_right
    invalid_left = ~valid_left
    # general conditions
    correct_target = valid_right | valid_left
    incorrect_target = invalid_right | invalid_left

    df_fix_aoi.loc[correct_target, 'fix_target_correct'] = 1
    df_fix_aoi.loc[incorrect_target, 'fix_target_incorrect'] = 1
    df_fix_aoi.loc[valid_central, 'fix_central_stim'] = 1

    df_fix_aoi_corr_target = df_fix_aoi

    return df_fix_aoi_corr_target

# =============================================================================
#  GET CORRECT FIXATIONS CENTRAL
# =============================================================================


def valid_fix_central(df_raw, df_fix_aoi):

    """
    Checks if there was gaze during the last 200 ms of the last sample
    of the central stimulus. If there was not gaze, the trial is excluded.
    Count the number of excluded and completed trials per subject.

    Input:
        df_raw: DataFrame with raw data
        df_fix_aoi: DataFrame with fixations information and AOIs
    Output:
        df_last_fix: DataFrame with the amount of time the participant
                     was not looking at the target during the last 200 ms.
                     The start timestamp of the last 200 ms and the end of
                     the central stimulus. Fixation start and end times.
        count_trials_excluded: number of trials excluded per participant
    """

    # select time of the last sample per subject and trial.
    df_trial_last_sample = pd.DataFrame(df_raw.groupby(['subject', 'trial'])['time'].last())
    df_trial_last_sample.reset_index(drop=False, inplace=True)
    # merge with df_fix_aoi per subject and trial
    df_offset_central = df_trial_last_sample.merge(df_fix_aoi, how='inner', on=['subject', 'trial'])
    # this last sample is the time in which the central stimulus ends
    df_offset_central.rename(columns={'time': 'end_last_200ms'}, inplace=True)
    # start timestamp of the last 200 ms of the central stimulus
    df_offset_central['start_last_200ms'] = df_offset_central['end_last_200ms'] - 200
    # subtract endT of the sample from end_last200ms
    df_offset_central['time_no_looking_end_trial'] = df_offset_central['end_last_200ms'] - df_offset_central['endT']
    # select the last sample per subject and trial with the corresponding subtraction
    df_last_fix_central = df_offset_central.groupby(['subject', 'trial'])['time_no_looking_end_trial',
                                                                          'start_last_200ms',
                                                                          'end_last_200ms',
                                                                          'startT', 'endT'].last()
    df_last_fix_central.reset_index(drop=False, inplace=True)

    df_last_fix_central['trial_filter'] = ''
    # conditions
    condition_excluded = df_last_fix_central['time_no_looking_end_trial'] > 0
    condition_included = df_last_fix_central['time_no_looking_end_trial'] == 0
    # set values
    df_last_fix_central.loc[condition_excluded, 'trial_filter'] = 0
    df_last_fix_central.loc[condition_included, 'trial_filter'] = 1
    # count the number of trials excluded by subject
    count_trials_excluded = df_last_fix_central[df_last_fix_central['time_no_looking_end_trial'] > 0].groupby(['subject']).size().reset_index(name='count')
    # dictionary with total trials completed by subject
    dict_total_trials = df_offset_central.groupby(['subject'])['trial'].unique().apply(lambda x: len(x)).to_dict()
    # map dictionary into df
    count_trials_excluded['total_trials'] = count_trials_excluded['subject'].map(dict_total_trials)
    
    return df_last_fix_central, count_trials_excluded

# =============================================================================
#  GET SACCADE LATENCY FROM FIRST CORRECT FIXATION IN TARGET
# =============================================================================


def saccade_latency(df_raw, df_last_fix_central):

    """"
    Select the first row of each trial from raw data grouped by subject and trial,
    and store the onset of the period in a DataFrame for each trial.

    Merge both dataframes, and subtract the period start time (target onset) from the
    fixation start time.

    Saccade latency is computed subtracting the target onset from the fixation.
    Nested function "first_saccade_latency) keeps the saccade latency for the first
    correct fixation in the target.

    Input:
        df_raw: DataFrame with raw data
        df_last_fix: DataFrame with information computed for the last fixation of a trial
    Output:
        df_first_corr_sacc_latency: saccade latency between target onset and first
                                    correct fixation on the target
    """

    # nested function: extract the saccade latency for the first correct fixation in the target
    def first_saccade_latency(df):
        df_corr = df.query('fix_target_correct == 1').groupby(['subject', 'trial']).head(1)

        return df_corr

    # extract target onset from raw data
    df_trial_first_sample = pd.DataFrame(df_raw.groupby(['subject', 'trial'])['time'].first())
    df_trial_first_sample.reset_index(drop=False, inplace=True)
    # add target onset to df_fix_aoi_corr_target
    df_onset_target = df_trial_first_sample.merge(df_last_fix_central, how='inner', on=['subject', 'trial'])
    df_saccade_latency = df_onset_target.copy(deep=True)

    # compute saccade lantency from start time of the first sample of the trial,  to start of fixation
    df_saccade_latency['Saccade_latency_StartFix_StartTrial'] = df_saccade_latency['startT'] - \
                                                                df_saccade_latency['time']
    # call first_saccade_latency nested function
    df_first_corr_sacc_latency = first_saccade_latency(df_saccade_latency)

    return df_first_corr_sacc_latency

# =============================================================================
#  COMPUTE MEDIAN SCORES (ONLY ON VALID TRIALS AND ABOVE THE SACCADE THRESHOLD)
# =============================================================================


def median_scores(df_central_stim, df_first_corr_sacc_latency, saccade_threshold):

    """
    Keeps only valid trials (fixation in the last 200ms on the central stimulus), drops
    validated trials that have a saccade latency below the threshold (integer) and calls
    computemedian nested function to compute median scores by subject.

    Input:
        df_central_stim: DataFrame with central stimulus information
        df_first_corr_sacc_latency: DataFrame with first saccade latencies on target
        saccade_threshold (int): saccade threshold as integer

    Output:
        df_median: DataFrame with median scores
    """

    def compute_median(df):
        d = {}
        d['median_SaccLat_StartFix_StartTrial'] = df['Saccade_latency_StartFix_StartTrial'].median()
        d['total_ValidTrials'] = df['trial'].count()
        d['total_NoDisengagement'] = df['Saccade_latency_StartFix_StartTrial'].isnull().sum()
        d['total_Disengagement'] = df['Saccade_latency_StartFix_StartTrial'].notnull().sum()

        return pd.Series(d)

    # keeps valid trials of central fixations
    df_valid_trials = df_central_stim.query('trial_filter == 1').merge(df_first_corr_sacc_latency[['subject', 'trial',
                                                                                                   'Saccade_latency_StartFix_StartTrial']],
                                                                      on=['subject', 'trial'], how='left')
    # drop valid trials with saccade latency below the threshold
    df_sacc_threshold = df_valid_trials.drop(df_valid_trials[df_valid_trials['Saccade_latency_StartFix_StartTrial']
                                                             < saccade_threshold].index, axis=0)
    # call compute_median function
    df_median = df_sacc_threshold.groupby(['subject']).apply(compute_median)

    return df_median

# =============================================================================
#  COMPUTE FINAL SCORES
# =============================================================================


def compute_scores(gap_count_trials_excluded, df_gap_median, overlap_count_trials_excluded, df_overlap_median):
    """
    Calculate scores merging information for gap and overlap trials.
    Extract a dictionary with total trials per participant as it will
    be needed to compute scores.

    Input:
        df_gap_count_trials_excluded: DataFrame with the count of trials
                                      excluded for gap condition
        df_gap_median: DataFrame with median computed scores for gap
                       condition
        overlap_count_trials_excluded: DataFrame with the count of
                                     trials excluded for overlap condition
        overlap_median: DataFrame with median computed scores for overlap
                        condition

    Output:
        df_results: DatFrame with resulting scores
    """

    # dictionaries with total trials per participant
    dict_gap_total_trials = dict(zip(gap_count_trials_excluded.subject, gap_count_trials_excluded.total_trials))
    dict_overlap_total_trials = dict(zip(overlap_count_trials_excluded.subject, overlap_count_trials_excluded.total_trials))

    # merge overlap median and gap median
    df_results = df_overlap_median.merge(df_gap_median, on='subject', how='inner', suffixes=('_Overlap',
                                                                                             '_Gap')).reset_index(drop=False)

    # map total trials into dfs. Total trials count without excluding trials
    df_results['Total_GapTrials'] = df_results['subject'].map(dict_gap_total_trials)
    df_results['Total_OverlapTrials'] = df_results['subject'].map(dict_overlap_total_trials)
    # total trials (all conditions)
    df_results['Total_Trials'] = df_results['Total_GapTrials'] + df_results['Total_OverlapTrials']
    # total valid trials (all conditions)
    df_results['Total_ValidTrials'] = df_results['total_ValidTrials_Overlap'] + df_results['total_ValidTrials_Gap']
    # index: median saccade latency overlap / median saccade lantecy gap
    df_results['index'] = df_results['median_SaccLat_StartFix_StartTrial_Overlap'] / \
                          df_results['median_SaccLat_StartFix_StartTrial_Gap']
    # proportion disengagement overlap
    df_results['prop_Disengagement_Over'] = df_results['total_Disengagement_Overlap'] / \
                                            df_results['total_ValidTrials_Overlap']
    # proportion disengagement gap
    df_results['prop_Disengagement_Gap'] = df_results['total_Disengagement_Gap'] / df_results['total_ValidTrials_Gap']
    # proportion disengagement total (all conditions)
    df_results['prop_Disengagement_Total'] = (df_results['total_Disengagement_Overlap'] +
                                              df_results['total_Disengagement_Gap']) / \
                                             (df_results['total_ValidTrials_Overlap'] +
                                              df_results['total_ValidTrials_Gap'])
    # proportion of valid trials over total trials
    df_results['prop_valid_Total'] = df_results['Total_ValidTrials'] / df_results['Total_Trials']
    # proportion of valid gap trials over total trials
    df_results['prop_valid_Gap_Total'] = df_results['total_ValidTrials_Gap'] / df_results['Total_GapTrials']
    # proportion of valid overlap trials over total trials
    df_results['prop_valid_Overlap_Total'] = df_results['total_ValidTrials_Overlap'] / df_results['Total_OverlapTrials']

    results_dir = os.path.join(directory_saving_results, 'AnalyzedI2MC')
    df_results.to_csv(os.path.join(results_dir, 'Gap_Overlap_results.txt'), sep=';', decimal=',')

    return df_results


# =============================================================================
#  GENERAL FUNCTIONS FOR DATA ANALYSIS
# =============================================================================


def analyze_overlap_central():

    """
        Calls the needed functions to compute scores for central overlap stimulus

        Input:
            None
        Output:
            df_last_fix_central: DataFrame computed scores for the
                                 central stimulus. Also if valid or not
            count_trials_excluded: number of trials excldued and total
                                   trials per participant
        Notes:
            Saves to pdf df_last_fix_central and count_trials_excluded
    """

    # set current directory
    os.chdir(directory_OverlapCentral)
    # save the results in a new folder created in the same path
    results_dir = os.path.join(directory_saving_results, 'AnalyzedI2MC')
    # if the directory doesn't exist, this loop creates it
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    print('Current Working Directory ', os.getcwd())
    directory = os.getcwd()
    df_data_fixations, df_data_raw, files = apply_I2MC_all_files(directory)
    df_aoi_fixations = AOI_fixations_central(df_data_fixations)
    df_last_fix_central, count_trials_excluded = valid_fix_central(df_data_raw, df_aoi_fixations)
    df_last_fix_central.to_csv(os.path.join(results_dir, 'Overlap_Central_Stim.txt'), sep=';', decimal=',')
    count_trials_excluded.to_csv(os.path.join(results_dir, 'Overlap_Central_Stim_Count_Excluded.txt'), sep=';', decimal=',')

    return df_last_fix_central, count_trials_excluded


def analyze_gap_central():

    """
        Calls the needed functions to compute scores for central gap stimulus

        Input:
            None
        Output:
            df_last_fix_central: DataFrame computed scores for the
                                 central stimulus. Also if valid or not
            count_trials_excluded: number of trials excluded and total
                                   trials per participant
        Notes:
            Saves to pdf df_last_fix_central and count_trials_excluded
    """

    # set current directory
    os.chdir(directory_GapCentral)
    # save the results in a new folder created in the same path
    results_dir = os.path.join(directory_saving_results, 'AnalyzedI2MC')
    # if the directory doesn't exist, this loop creates it
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    print('Current Working Directory ', os.getcwd())
    # get current working directory
    directory = os.getcwd()
    # apply I2MC function
    df_data_fixations, df_data_raw, files = apply_I2MC_all_files(directory)
    # get AOI for each fixation
    df_aoi_fixations = AOI_fixations_central(df_data_fixations)
    # validate central fixations (gaze in the last 200 ms)
    df_last_fix_central, count_trials_excluded = valid_fix_central(df_data_raw, df_aoi_fixations)
    # save to csv
    df_last_fix_central.to_csv(os.path.join(results_dir, 'Gap_Central_Stim.txt'), sep=';', decimal=',')
    count_trials_excluded.to_csv(os.path.join(results_dir, 'Gap_Central_Stim_Count_Excluded.txt'), sep=';', decimal=',')

    return df_last_fix_central, count_trials_excluded


def analyze_overlap_target(df_central_overlap, saccade_threshold):

    """
        Calls the needed functions to compute scores for overlap target

        Input:
            df_central_overlap: DataFrame with computed data for the
                                central stimulus
            saccade_threshold (int): threshold for saccade latencies
        Output:
            df_median: DataFrame with medians scores
        Notes:
            Saves to pdf df_first_corr_saccade_latency and df_median
    """

    # set current directory
    os.chdir(directory_OverlapTarget)
    # save the results in a new folder created in the same path
    results_dir = os.path.join(directory_saving_results, 'AnalyzedI2MC')
    # if the directory doesn't exist, this loop creates it
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    print('Current Working Directory ', os.getcwd())
    # get current working directory
    directory = os.getcwd()
    # apply I2MC function
    df_data_fixations, df_data_raw, files = apply_I2MC_all_files(directory)
    # get AOI for each fixation
    df_aoi_fixations = AOI_fixations_target(df_data_fixations)
    # get valid fixations (fixation in the same AOI of the target appearance)
    df_central_fix_valid = valid_fix_target(df_aoi_fixations)
    # get saccade latency for first correct fixation on target
    df_first_corr_sacc_latency = saccade_latency(df_data_raw, df_central_fix_valid)
    # save to csv
    df_first_corr_sacc_latency.to_csv(os.path.join(results_dir, 'Overlap_Saccade_Latency.txt'), sep=';', decimal=',')
    df_median = median_scores(df_central_overlap, df_first_corr_sacc_latency, saccade_threshold=saccade_threshold)

    return df_median

    
def analyze_gap_target(df_central_gap, saccade_threshold):

    """
    Calls the needed functions to compute scores for gap target

    Input:
        df_central_gap: DataFrame with computed data for the
                        central stimulus
        saccade_threshold (int): threshold for saccade latencies
    Output:
        df_median: DataFrame with medians scores
    Notes:
        Saves to pdf df_first_corr_saccade_latency and df_median
    """

    # set current directory
    os.chdir(directory_GapTarget)
    # save the results in a new folder created in the same path
    results_dir = os.path.join(directory_saving_results, 'AnalyzedI2MC')
    # if the directory doesn't exist, this loop creates it
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    print('Current Working Directory ', os.getcwd())
    # get current working directory
    directory = os.getcwd()
    # apply I2MC function
    df_data_fixations, df_data_raw, files = apply_I2MC_all_files(directory)
    # get AOI for each fixation
    df_aoi_fixations = AOI_fixations_target(df_data_fixations)
    # get valid fixations (fixation in the same AOI of the target appearance)
    df_central_fix_valid = valid_fix_target(df_aoi_fixations)
    # get saccade latency for first correct fixation on target
    df_first_corr_sacc_latency = saccade_latency(df_data_raw, df_central_fix_valid)
    # save to csv
    df_first_corr_sacc_latency.to_csv(os.path.join(results_dir, 'Gap_Saccade_Latency.txt'), sep=';', decimal=',')
    df_median = median_scores(df_central_gap, df_first_corr_sacc_latency, saccade_threshold=saccade_threshold)

    return df_median

# example how to call functions
# overlap_central, overlap_excluded = analyze_overlap_central()
# gap_central, gap_excluded = analyze_gap_central()
# overlap_median = analyze_overlap_target(overlap_central, saccade_threshold=120)
# gap_median = analyze_gap_target(gap_central, saccade_threshold=120)
# results = compute_scores(gap_excluded, gap_median, overlap_excluded, overlap_median)
