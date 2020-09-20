# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 03:36:10 2020

@author: Sebastián Moyano Flores
PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain.

"""
# =============================================================================
#  IMPORT MODULES
# =============================================================================

import pandas as pd
import glob
from extract_reduce_tablet_data import *

# =============================================================================
#  SET PARAMETERS
# =============================================================================

# set params
anticipations = 'anticipated_count'     # column with the number of anticipatory touches

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

# =============================================================================
#  COUNT ANTICIPATIONS FUNCTION
# =============================================================================


def count_anticipations(df):

    """
    Count anticipations in the alerting task.
    This function could be applied independently to the raw data,
    and then merge with the DataFrame with the rest of scores.

    Input:
        df: loaded DataFrame with raw data
    Output:
        df_anticipations_pivot: pivot table for the df_anticipations
                               with the number of anticipations in alert,
                               no alert and total anticipations
    """

    def anticipations(data):

        """
        Nested function that counts the number of anticipations in
        the alerting files.
        """

        d = {}
        d['anticipations'] = len(data[data[anticipations] != 0].index)

        return pd.Series(d)

    df_anticipations = df.groupby([id_code, timestamp, trial]).apply(anticipations)
    df_anticipations_pivot = pd.pivot_table(df_anticipations, index=[id_code, timestamp],
                                            columns=[trial]).reset_index(drop=False)
    df_anticipations_pivot.columns = df_anticipations_pivot.columns.map('_'.join).str.strip('_')
    check_cols = ['anticipations_Alert', 'anticipations_NoAlert']
    for column_tuple in check_cols:
        if column_tuple not in df_anticipations_pivot:
            df_anticipations_pivot[column_tuple] = 0
    df_anticipations_pivot = df_anticipations_pivot.rename(columns={id_code: 'subject',
                                                                    timestamp: 'datetime'})
    df_anticipations_pivot['total_anticipations'] = df_anticipations_pivot[['anticipations_Alert',
                                                                            'anticipations_NoAlert']].sum(axis=1)

    return df_anticipations_pivot

# =============================================================================
#  EXAMPLE
# =============================================================================


data = []

for filename in all_files_alerting:
    print('Working on alerting file ' + filename)
    df = pd.read_csv(filename, sep=',', header=0, index_col=False, decimal='.', error_bad_lines=False)
    df_processed_without200ms = process_file(df, trial, 'Alert', 'NoAlert', cols_to_keep)
    # count number of anticipations
    df_anticipations_pivot = count_anticipations(df)
    # merge with other data
    results_without200 = df_processed_without200ms.merge(df_anticipations_pivot, how='inner',
                                                         on=['subject', 'datetime'])

    data.append(results_without200)
    results = pd.concat(data)

    results.to_csv('C:....csv', sep=';', decimal=',')


