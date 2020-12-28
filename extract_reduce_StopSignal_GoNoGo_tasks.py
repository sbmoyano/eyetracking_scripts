# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 03:36:10 2020

@author: Sebastian Moyano
PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

Functions to reduce data from a Stop Signal and Go-NoGo task. Data structure
is provided after exporting files using E-merge and E-DataAid software from
E-prime.
"""

# =============================================================================
#  IMPORT MODULES
# =============================================================================

import pandas as pd
import glob
import os

# =============================================================================
#  SET PARAMETERS
# =============================================================================

# path
directory = 'C:/Users/sebas/ownCloud/DATOS W5/Stopsignaltask Backup 18mayo2018/Exported/StopSignal.csv'
# load files (csv)
file = pd.read_csv(directory, sep=';')
# rename columns to remove dots
file.rename(columns={'PreTarget.ACC[SubTrial]': 'PreTarget_ACC',
                     'PreTarget.RT[SubTrial]': 'PreTarget_RT',
                     'PreTarget.Duration[SubTrial]': 'PreTarget_Duration',
                     'Target.ACC[SubTrial]': 'Target_ACC',
                     'Target.RT[SubTrial]': 'Target_RT',
                     'Target.Duration[SubTrial]': 'Target_Duration',
                     'Task[SubTrial]': 'Task',
                     'TrialType[SubTrial]': 'TrialType',
                     'FixDuration[SubTrial]': 'FixDuration'}, inplace=True)

# set params
id_code = 'Subject'                     # column with participant id
task = 'Task'                           # column with code for task type (StopSignal or Go-NoGo)
trial = 'SubTrial'                      # column with trial types (congruent, incongruent, etc.)
trial_type = 'TrialType'                # column with codes for trial type (Go, NoGo or StopSignal)
fix_dur = 'FixDuration'                 # column with fixation durations
target_RT = 'Target_RT'                 # column with reaction times
target_acc = 'Target_ACC'               # column with correct and incorrect responses

# if the children omit the pretarget (RT = 0), accuracy = 1
pretarget_RT = 'PreTarget_RT'           # column with reaction time for the pretarget
pretarget_acc = 'PreTarget_ACC'         # column with accuracy for the pretarget
pretarget_dur = 'PreTarget_Duration'    # column with pretarget duration
correct = 1                             # code for correct responses


cols_to_keep = [id_code, task, trial, trial_type, fix_dur,
                target_RT, target_acc, pretarget_RT, pretarget_acc, pretarget_dur]

# =============================================================================
#  PROCESS FILE FUNCTION
# =============================================================================


def process_file(dataframe, trial_type, cols_keep):

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

    def compute_mean(dataframe):
        # reduce data by subject and task type for Go trials and only in correct responses
        file_go = dataframe[(dataframe[trial_type] == 'Go') &
                            (dataframe[target_RT] > 200) &
                            (dataframe[target_acc] == 1)]
        file_results_go = pd.DataFrame(file_go.groupby([id_code, task, trial_type])[target_RT].mean())
        file_pivot_go = pd.pivot_table(file_results_go, index=[id_code],
                                       columns=[task, trial_type]).reset_index()
        file_pivot_go.columns = ['_'.join(col).strip() for col in file_pivot_go.columns.values]
        file_pivot_go.rename(columns={'Subject_': 'Subject',
                                      'Subject__': 'Subject',
                                      'Subject___': 'Subject'}, inplace=True)
        # reduce data by subject and task type for No-Go and StopSignal trials, and only in incorrect responses
        file_nogo_stopsignal = dataframe[(dataframe[trial_type] != 'Go') &
                                         (dataframe[target_acc] == 0)]
        file_results_nogo = pd.DataFrame(file_nogo_stopsignal.groupby([id_code, task, trial_type])[target_RT,
                                                                                                   pretarget_dur].mean())
        file_pivot_nogo = pd.pivot_table(file_results_nogo, index=[id_code],
                                         columns=[task, trial_type]).reset_index()
        file_pivot_nogo.columns = ['_'.join(col).strip() for col in file_pivot_nogo.columns.values]
        file_pivot_nogo.rename(columns={'Subject_': 'Subject',
                                        'Subject__': 'Subject',
                                        'Subject___': 'Subject'}, inplace=True)
        go_nogo_stopsignal = file_pivot_go.merge(file_pivot_nogo, how='inner', on=id_code)
        # we only need pretarget for StopSignal trials, in NoGo trials is fixed
        go_nogo_stopsignal.drop(columns='PreTarget_Duration_GoNoGo_NoGo', inplace=True)

        return go_nogo_stopsignal

    def compute_errors(dataframe):

        def compute(x):

            d = {}
            d['n_total_trials'] = x[target_acc].count()
            d['n_total_errors'] = x[x[target_acc] == 0][target_acc].count()
            # omission errors have a reaction time equal to zero (no response)
            d['n_omission_errors'] = x[(x[target_acc] == 0) &
                                       (x[target_RT] == 0)][target_RT].count()
            d['n_comission_errors'] = d['n_total_errors'] - d['n_omission_errors']
            d['pretarget_responses'] = x[(x[pretarget_acc] == 0)][pretarget_acc].count()

            return pd.Series(d)

        # compute errors grouping by subject and trial type
        results_errors = dataframe.groupby([id_code, task, trial_type]).apply(compute)
        # pivot table
        results_pivot = pd.pivot_table(results_errors, index=[id_code],
                                       columns=[task, trial_type]).reset_index(drop=False)
        results_pivot.columns = ['_'.join(col).strip() for col in results_pivot.columns.values]
        results_pivot.rename(columns={'Subject_': 'Subject',
                                      'Subject__': 'Subject',
                                      'Subject___': 'Subject'}, inplace=True)
        return results_pivot

    # condition to replace with 0 pretarget_RT != 0 in Go and NoGo trials
    condition_replace_non_stopsignal = (dataframe[trial_type] == 'StopSignal')
    # condition to replace target_RT with pretarget_RT if pretarget_RT is different than 0
    condition_pretarget_RT = (dataframe[pretarget_RT] == 0)
    # replace with zero pretarget_RT for Go and NoGo trials. If trial is StopSignal then keep original pretarget_RT
    dataframe[pretarget_RT].where(condition_replace_non_stopsignal, 0, inplace=True)
    # where pretarget_RT is different than 0, replace target_RT with pretarget_RT
    dataframe[target_RT].where(condition_pretarget_RT, dataframe[pretarget_RT], inplace=True)
    # where pretarget_acc is different than StopSignal, replace pretarget_acc with 1
    dataframe[pretarget_acc].where(condition_replace_non_stopsignal, 1, inplace=True)
    # select columns of interest and drop NaN values
    dataframe = dataframe[cols_keep]
    # change dtypes
    dataframe = dataframe.astype({id_code: int,
                                  fix_dur: int,
                                  pretarget_RT: int,
                                  pretarget_acc: int,
                                  target_RT: int,
                                  target_acc: int,
                                  trial: int})
    # compute mean
    df_RT_subject_go_nogo_stopsignal = compute_mean(dataframe)
    # compute errors
    df_errors_subject_go_nogo_stopsignal = compute_errors(dataframe)

    return df_RT_subject_go_nogo_stopsignal, df_errors_subject_go_nogo_stopsignal, dataframe


go_nogo_RT, go_nogo_errors, df = process_file(file, trial_type, cols_to_keep)

