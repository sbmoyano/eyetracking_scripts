# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 03:36:10 2020

@author: Sebastian Moyano
PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

"""

import pandas as pd
import glob
import os

# paths
directory = r'C:/...'
# load files (csv)
all_files = glob.glob(directory + '/*.csv')

# set params
id_code = 'subject_nr'     # column with participant id
trial = 'trialType'        # column with trial types (congruent, incongruent, etc.)
RT = 'response_time'       # column with reaction times
accuracy = 'correct'       # column with correct and incorrect responses
correct = 1                # code for correct responses
timestamp = 'datetime'     # timestamp column (sometimes is useful to differentiate files with same id)
cols_to_keep = [id, trial, RT, accuracy, timestamp]


def process_file(dataframe, trial_type_col, trial_type_1, trial_type_2, cols_keep):

    """
    Function to process files (one at a time) from tabled based tasks. Used for alerting,
    spatial conflict and orienting tasks. The function does not compute percentages,
    only median reaction time by condition, count the number of correct responses,
    commission errors and omission errors in the file.

    Input:
    df: DataFrame with data
    trial_type_col: column names that identifies trial types
    trial_type_1: name of the first type of trial
    trial_type_2: name of the second type of trial

    Output:
    DataFrame with reduced scores
    """

    def compute_median(dataframe):
        results_median = pd.DataFrame(dataframe.groupby([id_code, timestamp, trial_type_col])[RT].median())
        results_median_pivot = pd.pivot_table(results_median, index=[id_code, timestamp],
                                              columns=[trial_type_col]).reset_index(drop=False)

        check_cols = [(RT, trial_type_1), (RT, trial_type_2)]

        for column_tuple in check_cols:
            if column_tuple not in results_median_pivot:
                results_median_pivot[column_tuple] = 0

        results_median_pivot.columns = ['subject', 'timestamp', trial_type_1, trial_type_2]

        return results_median_pivot

    def compute_errors(dataframe):

        def compute(x):

            d = {}
            d['number_correct'] = x[accuracy].count()
            d['number_errors'] = x[x[accuracy] == 0][accuracy].count()
            # omission errors have a reaction time equal to zero (no response)
            d['number_errors_omission'] = x[x[RT] == 0][RT].count()
            d['number_errors_commission'] = d['number_errors'] - d['number_errors_omission']

            return pd.Series(d)

        # compute errors grouping by subject, datetime and trial type
        df_trial_type = dataframe.groupby([id_code, timestamp, trial_type_col]).apply(compute)
        # pivot table
        df_trial_type_pivot = pd.pivot_table(df_trial_type, index=[id_code, timestamp],
                                             columns=[trial_type_col]).reset_index(drop=False)
        # flatten multiindex columns
        df_trial_type_pivot.columns = df_trial_type_pivot.columns.map('_'.join).str.strip('_')

        check_cols = ['number_errors_' + trial_type_1, 'number_errors_commission_' + trial_type_1,
                      'number_errors_omission_' + trial_type_1, 'number_correct_' + trial_type_1,
                      'number_errors_' + trial_type_2, 'number_errors_commission_' + trial_type_2,
                      'number_errors_omission_' + trial_type_2, 'number_correct_' + trial_type_2]

        for column_tuple in check_cols:
            if column_tuple not in df_trial_type_pivot:
                df_trial_type_pivot[column_tuple] = 0

        # rename columns
        df_trial_type_pivot = df_trial_type_pivot.rename(columns={id_code: 'subject',
                                                                  timestamp: 'timestamp',
                                                                  'number_errors_' + trial_type_1: 'errors_' + trial_type_1,
                                                                  'number_errors_' + trial_type_2: 'errors_' + trial_type_2,
                                                                  'number_errors_commission_' + trial_type_1: 'errors_commission_' + trial_type_1,
                                                                  'number_errors_commission_' + trial_type_2: 'errors_commission_' + trial_type_2,
                                                                  'number_errors_omission_' + trial_type_1: 'errors_omission_' + trial_type_1,
                                                                  'number_errors_omission_' + trial_type_2: 'errors_omission_' + trial_type_2,
                                                                  'number_correct_' + trial_type_1: 'total_correct_' + trial_type_1,
                                                                  'number_correct_' + trial_type_2: 'total_correct_' + trial_type_2})
        # compute total errors and total trials
        df_trial_type_pivot['total_errors'] = df_trial_type_pivot[['errors_' + trial_type_1,
                                                                   'errors_' + trial_type_2]].sum(axis=1)
        df_trial_type_pivot['total_correct'] = df_trial_type_pivot[['total_correct_' + trial_type_1,
                                                                    'total_correct_' + trial_type_2]].sum(axis=1)

        return df_trial_type_pivot

    # select columns of interest and drop NaN values
    dataframe = dataframe[cols_keep].dropna()
    # change dtypes
    dataframe = dataframe.astype({accuracy: int, RT: int, id: int})
    # drop reaction times below 200 ms but keep reaction times equal to zero (omission errors, no response)
    df_without200ms = dataframe[(dataframe[RT] > 200) | (dataframe[RT] == 0)]
    # filter by correct responses and reset index
    df_corr_without200ms = df_without200ms.query(accuracy + ' == ' + str(correct)).reset_index(drop=True)
    # compute median
    df_without200ms_median = compute_median(df_corr_without200ms)
    # compute errors
    df_without200ms_errors = compute_errors(df_without200ms)
    # merge dfs
    results_without200ms = df_without200ms_median.merge(df_without200ms_errors, how='inner', on=['subject', 'timestamp'])

    return results_without200ms


data = []

# example of for loop to load and process files inside a folder
for filename in all_files_alerting:
    print('Working on file ' + filename)
    df = pd.read_csv(filename, sep=',', header=0, index_col=False, decimal='.', error_bad_lines=False)
    data_processed = process_file(df, trial, 'Alert', 'NoAlert', cols_to_keep)
    data.append(data_processed)
    results = pd.concat(data)
    results_dir = os.path.join(directory, 'Analyzed_data')
    # if the directory doesn't exist, this loop creates it
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    results.to_csv(os.path.join(results_dir, 'reduced_data.csv'), sep=';', decimal='.')


