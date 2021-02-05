# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 03:36:10 2020

@author: Sebastian Moyano
PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Center for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain

"""

# =============================================================================
#  IMPORT MODULES
# =============================================================================

import pandas as pd
import os

# =============================================================================
#  SET PARAMETERS
# =============================================================================

# path
directory = 'C:\\Users\\sebas\\ownCloud\\DATOS W5\\ANT Backup 18mayo2018\\Exported\\ANT_filtered_added.csv'
# load files (csv)
file = pd.read_csv(directory, sep=';')
# rename columns to remove dots
file.rename(columns={'Target.ACC': 'Target_ACC', 'Target.RT': 'Target_RT'}, inplace=True)

# set params
id_code = 'Subject'           # column with participant id
trial = 'Trial'               # column with trial types (congruent, incongruent, etc.)
block = 'BlockList'           # column with bock list
RT = 'Target_RT'              # column with reaction times
accuracy = 'Target_ACC'       # column with correct and incorrect responses
correct = 1                   # code for correct responses
cond1 = 'CueCond'             # code for cue condition
cond2 = 'Flanker'             # code for flanker condition

cols_to_keep = [id_code, block, trial, RT, accuracy, cond1, cond2]

# =============================================================================
#  PROCESS FILE FUNCTION
# =============================================================================


def process_file(dataframe, trial_type_col1, trial_type_col2, cols_keep):

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
        # by subject (cue and flanker type)
        results_mean_cue = pd.DataFrame(dataframe.groupby([id_code, trial_type_col1])[RT].mean())
        result_mean_flanker = pd.DataFrame(dataframe.groupby([id_code, trial_type_col2])[RT].mean())
        result_mean_flanker_cue = pd.DataFrame(dataframe.groupby([id_code, trial_type_col2,
                                                                  trial_type_col1])[RT].mean())
        # by block (cue and flanker type)
        result_mean_cue_block = pd.DataFrame(dataframe.groupby([id_code, block, trial_type_col1])[RT].mean())
        result_mean_flanker_block = pd.DataFrame(dataframe.groupby([id_code, block, trial_type_col2])[RT].mean())
        result_mean_flanker_cue_block = pd.DataFrame(dataframe.groupby([id_code, block, trial_type_col2,
                                                                        trial_type_col1])[RT].mean())
        # pivot
        results_mean_cue_pivot = pd.pivot_table(results_mean_cue, index=[id_code],
                                                columns=[trial_type_col1]).reset_index(drop=False)
        results_mean_flanker_pivot = pd.pivot_table(result_mean_flanker, index=[id_code],
                                                    columns=[trial_type_col2]).reset_index(drop=False)
        results_mean_flanker_cue_pivot = pd.pivot_table(result_mean_flanker_cue, index=[id_code],
                                                        columns=[trial_type_col2, trial_type_col1]).reset_index(drop=False)
        results_mean_block_cue_pivot = pd.pivot_table(result_mean_cue_block, index=[id_code],
                                                      columns=[block, trial_type_col1]).reset_index(drop=False)
        results_mean_block_flanker_pivot = pd.pivot_table(result_mean_flanker_block, index=[id_code],
                                                          columns=[block, trial_type_col2]).reset_index(drop=False)
        results_mean_block_flanker_cue_pivot = pd.pivot_table(result_mean_flanker_cue_block, index=[id_code],
                                                              columns=[block, trial_type_col2,
                                                                       trial_type_col1]).reset_index(drop=False)

        # flatten multiindex columns
        dfs = [results_mean_cue_pivot, results_mean_flanker_pivot, results_mean_flanker_cue_pivot,
               results_mean_block_cue_pivot, results_mean_block_flanker_pivot, results_mean_block_flanker_cue_pivot]

        for result in dfs:
            result.columns = ['_'.join(col).strip() for col in result.columns.values]
            result.rename(columns={'Subject_': 'Subject',
                                   'Subject__': 'Subject',
                                   'Subject___': 'Subject'}, inplace=True)

        results_mean_cue_pivot, results_mean_flanker_pivot, results_mean_flanker_cue_pivot, \
        results_mean_block_cue_pivot, results_mean_block_flanker_pivot, results_mean_block_flanker_cue_pivot = dfs

        results_mean_subject = results_mean_cue_pivot.merge(results_mean_flanker_pivot, how='inner', on='Subject').\
            merge(results_mean_flanker_cue_pivot, how='inner', on='Subject')
        results_mean_block = results_mean_block_cue_pivot.merge(results_mean_block_flanker_pivot, how='inner',
                                                                on='Subject').merge(results_mean_block_flanker_cue_pivot,
                                                                                    how='inner', on='Subject')

        return results_mean_subject, results_mean_block

    def compute_errors(dataframe):

        def compute(x):

            d = {}
            d['n_total_trials'] = x[accuracy].count()
            d['n_total_errors'] = x[x[accuracy] == 0][accuracy].count()
            # omission errors have a reaction time equal to zero (no response)
            d['n_omission_errors'] = x[x[RT] == 0][RT].count()
            d['n_comission_errors'] = d['n_total_errors'] - d['n_omission_errors']

            return pd.Series(d)

        # compute errors grouping by subject and trial type
        df_errors_cue = dataframe.groupby([id_code, trial_type_col1]).apply(compute)
        df_errors_flanker = dataframe.groupby([id_code, trial_type_col2]).apply(compute)
        df_errors_flanker_cue = dataframe.groupby([id_code, trial_type_col2, trial_type_col1]).apply(compute)

        # compute errors grouping by subject, block and trial type
        df_errors_cue_block = dataframe.groupby([id_code, block, trial_type_col1]).apply(compute)
        df_errors_flanker_block = dataframe.groupby([id_code, block, trial_type_col2]).apply(compute)
        df_errors_flanker_cue_block = dataframe.groupby([id_code, block, trial_type_col2,
                                                         trial_type_col1]).apply(compute)
        # pivot table
        results_errors_cue = pd.pivot_table(df_errors_cue, index=[id_code],
                                            columns=[trial_type_col1]).reset_index(drop=False)
        results_errors_flanker = pd.pivot_table(df_errors_flanker, index=[id_code],
                                                columns=[trial_type_col2]).reset_index(drop=False)
        results_errors_flanker_cue = pd.pivot_table(df_errors_flanker_cue, index=[id_code],
                                                    columns=[trial_type_col2,
                                                             trial_type_col1]).reset_index(drop=False)
        results_errors_cue_block = pd.pivot_table(df_errors_cue_block, index=[id_code],
                                                  columns=[block, trial_type_col1]).reset_index(drop=False)
        results_errors_flanker_block = pd.pivot_table(df_errors_flanker_block, index=[id_code],
                                                      columns=[block, trial_type_col2]).reset_index(drop=False)
        results_errors_flanker_cue_block = pd.pivot_table(df_errors_flanker_cue_block, index=[id_code],
                                                          columns=[block, trial_type_col2,
                                                                   trial_type_col1]).reset_index(drop=False)

        # flatten multiindex columns
        dfs = [results_errors_cue, results_errors_flanker, results_errors_flanker_cue,
               results_errors_cue_block, results_errors_flanker_block, results_errors_flanker_cue_block]

        for result in dfs:
            result.columns = ['_'.join(col).strip() for col in result.columns.values]
            result.rename(columns={'Subject_': 'Subject',
                                   'Subject__': 'Subject',
                                   'Subject___': 'Subject'}, inplace=True)

        results_errors_cue, results_errors_flanker, results_errors_flanker_cue, \
        results_errors_cue_block, results_errors_flanker_block, results_errors_flanker_cue_block = dfs

        results_errors_subject = results_errors_cue.merge(results_errors_flanker, how='inner',
                                                          on='Subject'). merge(results_errors_flanker_cue,
                                                                               how='inner', on='Subject')

        results_errors_block = results_errors_cue_block.merge(results_errors_flanker_block, how='inner',
                                                              on='Subject').merge(results_errors_flanker_cue_block,
                                                                                  how='inner', on='Subject')

        # compute total errors type
        results_errors_subject['n_total_comission_errors'] = results_errors_subject['n_comission_errors_congruent'] + \
                                                             results_errors_subject['n_comission_errors_incongruent']
        results_errors_subject['n_total_omission_errors'] = results_errors_subject['n_omission_errors_congruent'] + \
                                                             results_errors_subject['n_omission_errors_incongruent']
        results_errors_subject['n_total_errors'] = results_errors_subject['n_total_errors_congruent'] + \
                                                   results_errors_subject['n_total_errors_incongruent']

        comission_errors_block1 = ['n_comission_errors_block1_congruent', 'n_comission_errors_block1_incongruent']
        comission_errors_block2 = ['n_comission_errors_block2_congruent', 'n_comission_errors_block2_incongruent']
        comission_errors_block3 = ['n_comission_errors_block3_congruent', 'n_comission_errors_block3_incongruent']
        comission_errors_block4 = ['n_comission_errors_block4_congruent', 'n_comission_errors_block4_incongruent']

        omission_errors_block1 = ['n_omission_errors_block1_congruent', 'n_omission_errors_block1_incongruent']
        omission_errors_block2 = ['n_omission_errors_block2_congruent', 'n_omission_errors_block2_incongruent']
        omission_errors_block3 = ['n_omission_errors_block3_congruent', 'n_omission_errors_block3_incongruent']
        omission_errors_block4 = ['n_omission_errors_block4_congruent', 'n_omission_errors_block4_incongruent']

        total_errors_block1 = comission_errors_block1 + omission_errors_block1
        total_errors_block2 = comission_errors_block2 + omission_errors_block2
        total_errors_block3 = comission_errors_block3 + omission_errors_block3
        total_errors_block4 = comission_errors_block4 + omission_errors_block4

        results_errors_block['n_total_comission_errors_block1'] = results_errors_block[comission_errors_block1].sum(
            axis=1)
        results_errors_block['n_total_comission_errors_block2'] = results_errors_block[comission_errors_block2].sum(
            axis=1)
        results_errors_block['n_total_comission_errors_block3'] = results_errors_block[comission_errors_block3].sum(
            axis=1)
        results_errors_block['n_total_comission_errors_block4'] = results_errors_block[comission_errors_block4].sum(
            axis=1)
        results_errors_block['n_total_omission_errors_block1'] = results_errors_block[omission_errors_block1].sum(
            axis=1)
        results_errors_block['n_total_omission_errors_block2'] = results_errors_block[omission_errors_block2].sum(
            axis=1)
        results_errors_block['n_total_omission_errors_block3'] = results_errors_block[omission_errors_block3].sum(
            axis=1)
        results_errors_block['n_total_omission_errors_block4'] = results_errors_block[omission_errors_block4].sum(
            axis=1)
        results_errors_block['n_total_errors_block1'] = results_errors_block[total_errors_block1].sum(
            axis=1)
        results_errors_block['n_total_errors_block2'] = results_errors_block[total_errors_block2].sum(
            axis=1)
        results_errors_block['n_total_errors_block3'] = results_errors_block[total_errors_block3].sum(
            axis=1)
        results_errors_block['n_total_errors_block4'] = results_errors_block[total_errors_block4].sum(
            axis=1)

        return results_errors_subject, results_errors_block

    # select columns of interest and drop NaN values
    dataframe = dataframe[cols_keep].dropna()
    # change dtypes
    dataframe = dataframe.astype({accuracy: int, RT: int, id_code: int})
    # change block names
    blocks = {1: 'block1', 2: 'block2', 3: 'block3', 4: 'block4'}
    dataframe[block] = dataframe[block].map(blocks)
    # drop reaction times below 200 ms but keep reaction times equal to zero (omission errors, no response)
    df_without200ms = dataframe[(dataframe[RT] > 200) | (dataframe[RT] == 0)]
    # filter by correct responses and reset index
    df_corr_without200ms = df_without200ms.query(accuracy + ' == ' + str(correct)).reset_index(drop=True)
    # compute median
    df_RT_subject, df_RT_block = compute_mean(df_corr_without200ms)
    # compute errors
    df_errors_subject, df_errors_block = compute_errors(df_without200ms)
    # merge dfs
    results_subject = df_RT_subject.merge(df_errors_subject, how='inner', on=[id_code])
    results_block = df_RT_block.merge(df_errors_block, how='inner', on=[id_code])

    return results_subject, results_block

# =============================================================================
#  EXAMPLE
# =============================================================================

results_subject, results_subject_block = process_file(file, cond1, cond2, cols_to_keep)

directory_files = 'C:\\Users\\sebas\\ownCloud\\DATOS W5\\ANT Backup 18mayo2018\\Exported'
os.chdir(directory_files)
results_dir = os.path.join(directory_files, 'Analyzed_data')

# if the directory doesn't exist, this loop creates it
try:
    os.mkdir(results_dir)
except OSError:
    print ("Creation of the directory %s failed" % results_dir)
else:
    print ("Successfully created the directory %s " % results_dir)

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

results_subject.to_csv(os.path.join(results_dir, 'reduced_data_subject.csv'), sep=';', decimal=',')
results_subject_block.to_csv(os.path.join(results_dir, 'reduced_data_subject_block.csv'), sep=';', decimal=',')

