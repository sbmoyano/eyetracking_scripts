
import pandas as pd
import numpy as np


def position(df):

    # position 1
    incorrect_one = df['AOI Name'].str.contains('ANTICIPATION INCORRECT 1')
    correct_one = df['AOI Name'].str.contains('ANTICIPATION CORRECT 1')
    reactive_one = df['AOI Name'].str.contains('DOING IT 1')

    # position 2
    incorrect_two = df['AOI Name'].str.contains('ANTICIPATION INCORRECT 2')
    correct_two = df['AOI Name'].str.contains('ANTICIPATION CORRECT 2')
    reactive_two = df['AOI Name'].str.contains('DOING IT 2')

    # position 3
    incorrect_three = df['AOI Name'].str.contains('ANTICIPATION INCORRECT 3')
    correct_three = df['AOI Name'].str.contains('ANTICIPATION CORRECT 3')
    reactive_three = df['AOI Name'].str.contains('DOING IT 3')

    # masks
    mask_position_one = incorrect_one | correct_one | reactive_one
    mask_position_two = incorrect_two | correct_two | reactive_two
    mask_position_three = incorrect_three | correct_three | reactive_three

    df['position'] = 0
    df.loc[mask_position_one, 'position'] = 1
    df.loc[mask_position_two, 'position'] = 2
    df.loc[mask_position_three, 'position'] = 3

    return df


def trial_type(df):

    # if reactive fixation in position 2 or 3 (unambiguous) or any type of anticipation to position 1
    # (unambiguous)
    unambiguous_list = ['ANTICIPATION INCORRECT 1 (POSITION 3)', 'ANTICIPATION INCORRECT 1 (POSITION 2)',
                        'ANTICIPATION CORRECT 1 (POSITION 1)', 'DOING IT 2', 'DOING IT 3']

    unambiguous = df['AOI Name'].isin(unambiguous_list)

    # masks
    mask_unambiguous_trial = unambiguous
    mask_ambiguous_trial = ~unambiguous

    df['trial_type'] = ''
    df.loc[mask_unambiguous_trial, 'trial_type'] = 'unambiguous'
    df.loc[mask_ambiguous_trial, 'trial_type'] = 'ambiguous'

    return df



def trial_number(df):

    df['trial_recoded'] = 0
    start_array = np.arange(9, 385, 6)
    end_array = np.append(np.arange(14, 385, 6), 384)
    init_trial = 2

    for init, end in zip(start_array, end_array):

        # condition
        condition = (df['Trial'] >= init) & (df['Trial'] <= end)
        condition_first = df['Trial'] < 9
        # mask
        df.loc[condition, 'trial_recoded'] = init_trial
        df.loc[condition_first, 'trial_recoded'] = 1
        # update init_trial
        init_trial += 1

    return df


def reactive(df):

    df['reactive'] = 0
    # list of reactive AOIs
    reactive_list = ['DOING IT 1', 'DOING IT 2', 'DOING IT 3']
    # conditions reactive fix
    AOI_reactive = df['AOI Name'].isin(reactive_list)
    condition_entry_time = df['Entry Time [ms]'] > 0
    # mask
    condition_reactive_fix = AOI_reactive & condition_entry_time
    # apply
    df.loc[condition_reactive_fix, 'reactive'] = 1

    return df


def raw_trial(df):

    df['trial_raw'] = 0
    start_array = np.arange(6, 385, 6)
    end_array = np.append(np.arange(11, 385, 6), 384)
    init_trial = 2

    for init, end in zip(start_array, end_array):

        # condition
        condition = (df['Trial'] >= init) & (df['Trial'] <= end)
        condition_first = df['Trial'] < 6
        condition_last = df['Trial'] > 384
        # mask
        df.loc[condition, 'trial_raw'] = init_trial
        df.loc[condition_first, 'trial_raw'] = 1
        df.loc[condition_last, 'trial_raw'] = 65
        # update init_trial
        init_trial += 1

    return df


def subtrial(df):

    df['subtrial'] = 0

    # list of subtrials
    subtrial_one = np.append(np.arange(1, 65, 4), 65)
    subtrial_two = np.arange(2, 66, 4)
    subtrial_three = np.arange(3, 67, 4)
    subtrial_four = np.arange(4, 68 , 4)

    # conditions
    condition_subtrial_one = df['trial_raw'].isin(subtrial_one)
    condition_subtrial_two = df['trial_raw'].isin(subtrial_two)
    condition_subtrial_three = df['trial_raw'].isin(subtrial_three)
    condition_subtrial_four = df['trial_raw'].isin(subtrial_four)

    # masks
    df.loc[condition_subtrial_one, 'subtrial'] = 1
    df.loc[condition_subtrial_two, 'subtrial'] = 2
    df.loc[condition_subtrial_three, 'subtrial'] = 3
    df.loc[condition_subtrial_four, 'subtrial'] = 4

    return df


def previous_position(df):

    df['previous_position'] = 0

    # conditions
    previous_one_first = (df['position'] == 2) & (df['subtrial'] == 2)
    previous_one_second = (df['position'] == 3) & (df['subtrial'] == 4)
    previous_two = (df['position'] == 1) & (df['subtrial'] == 3)
    previous_three = (df['position'] == 1) & (df['subtrial'] == 1)
    previous_one = previous_one_first | previous_one_second

    # masks
    df.loc[previous_one, 'previous_position'] = 1
    df.loc[previous_two, 'previous_position'] = 2
    df.loc[previous_three, 'previous_position'] = 3

    return df


def correct_incorrect_anticipations(df):

    df['correct_anticipation'] = 0
    df['incorrect_anticipation'] = 0

    list_stim_bs = ['BlackScreen1.bmp', 'BlackScreen2.bmp', 'BlackScreen3.bmp']
    list_correct_AOI = ['ANTICIPATION CORRECT 1 (POSITION 1)', 'ANTICIPATION CORRECT 2 (POSITION 2)',
                        'ANTICIPATION CORRECT 3 (POSITION 3)']
    list_incorrect_AOI = ['ANTICIPATION INCORRECT 1 (POSITION 2)', 'ANTICIPATION INCORRECT 1 (POSITION 3)',
                          'ANTICIPATION INCORRECT 2 (POSITION 1)', 'ANTICIPATION INCORRECT 2 (POSITION 3)',
                          'ANTICIPATION INCORRECT 3 (POSITION 1)', 'ANTICIPATION INCORRECT 3 (POSITION 2)']

    # conditions
    corr_anticipation_stim_bs = (df['Stimulus'].isin(list_stim_bs)) & (df['AOI Name'].isin(list_correct_AOI))
    incorr_anticipation_stim_bs = (df['Stimulus'].isin(list_stim_bs)) & (df['AOI Name'].isin(list_incorrect_AOI))
    anticipation_time_bs = (df['Entry Time [ms]'] > 200) & (df['Entry Time [ms]'] < 1000)
    corr_anticipation_stim_150 = df['AOI Name'].isin(list_correct_AOI) & (df['AOI Group'] == '150ms')
    incorr_anticipation_stim_150 = df['AOI Name'].isin(list_incorrect_AOI) & (df['AOI Group'] == '150ms')
    anticipation_time_150 = (df['Entry Time [ms]'] > 0) & (df['Entry Time [ms]'] < 150)
    corr_anticipation_stim_50 = df['AOI Name'].isin(list_correct_AOI) & (df['AOI Group'] == '50ms')
    incorr_anticipation_stim_50 = df['AOI Name'].isin(list_incorrect_AOI) & (df['AOI Group'] == '50ms')
    anticipation_time_50 = (df['Entry Time [ms]'] > 0) & (df['Entry Time [ms]'] < 50)

    # masks
    correct_anticipation_bs = corr_anticipation_stim_bs & anticipation_time_bs
    correct_anticipation_150 = corr_anticipation_stim_150 & anticipation_time_150
    correct_anticipation_50 = corr_anticipation_stim_50 & anticipation_time_50
    incorrect_anticipation_bs = incorr_anticipation_stim_bs & anticipation_time_bs
    incorrect_anticipation_150 = incorr_anticipation_stim_150 & anticipation_time_150
    incorrect_anticipation_50 = incorr_anticipation_stim_50 & anticipation_time_50

    # apply masks
    df.loc[correct_anticipation_bs, 'correct_anticipation'] = 1
    df.loc[correct_anticipation_150, 'correct_anticipation'] = 1
    df.loc[correct_anticipation_50, 'correct_anticipation'] = 1
    df.loc[incorrect_anticipation_bs, 'incorrect_anticipation'] = 1
    df.loc[incorrect_anticipation_150, 'incorrect_anticipation'] = 1
    df.loc[incorrect_anticipation_50, 'incorrect_anticipation'] = 1

    # if visited position is equal to previous position is not considered an anticipation
    visited_position_equal_previous_position_one = (df['AOI Name'].str.contains('POSITION 1')) & \
                                                   (df['previous_position'] == 1)
    visited_position_equal_previous_position_two = (df['AOI Name'].str.contains('POSITION 2')) & \
                                                   (df['previous_position'] == 2)
    visited_position_equal_previous_position_three = (df['AOI Name'].str.contains('POSITION 3')) & \
                                                     (df['previous_position'] == 3)

    # remove coded incorrect anticipations that are registered in the same AOI as the previous position
    df.loc[visited_position_equal_previous_position_one, 'incorrect_anticipation'] = 0
    df.loc[visited_position_equal_previous_position_two, 'incorrect_anticipation'] = 0
    df.loc[visited_position_equal_previous_position_three, 'incorrect_anticipation'] = 0

    return df


def choose_correct_incorrect_anticipation(df):

    grouped = df.groupby(['Participant',
                          'trial_recoded',
                          'trial_type',
                          'correct_anticipation',
                          'incorrect_anticipation',
                          'reactive']).agg({'Fixation Time [ms]': 'sum'}).reset_index(drop=False)

    # pivot table
    pivot = grouped.pivot(index=['Participant', 'trial_recoded', 'trial_type'],
                           columns=['correct_anticipation', 'incorrect_anticipation', 'reactive'],
                           values='Fixation Time [ms]')

    # change level name (columns multiindex)
    pivot.columns.set_levels(['none', 'reactive_time'], level=2, inplace=True)
    pivot.columns.set_levels(['none', 'incorrect_anticipation_time'], level=1, inplace=True)
    pivot.columns.set_levels(['none', 'correct_anticipation_time'], level=0, inplace=True)
    pivot.reset_index(drop=False, inplace=True)
    pivot.columns = ['_'.join(col) for col in pivot.columns.values]
    pivot.rename(columns={'Participant__': 'Participant',
                          'trial_recoded__': 'trial_recoded',
                          'trial_type__': 'trial_type',
                          'none_none_none': 'none',
                          'none_none_reactive_time': 'reactive_time',
                          'none_incorrect_anticipation_time_none': 'incorrect_anticipation_time',
                          'correct_anticipation_time_none_none': 'correct_anticipation_time'}, inplace=True)
    # drop unwanted times (these are higher than 1000 ms, 150 ms or 50 ms because of recording)
    # replace np.nan with zero
    pivot.drop(columns='none', inplace=True)
    pivot.replace(to_replace=np.nan, value=0, inplace=True)

    return pivot


def valid_anticipations(df, time):

    df['reactive'] = 0
    df['incorrect_anticipation'] = 0
    df['correct_anticipation'] = 0
    df['valid_correct_anticipation'] = 0
    df['valid_incorrect_anticipation'] = 0
    df['block'] = ''
    df['difference_between_fixations'] = np.abs(df['correct_anticipation_time'] - df['incorrect_anticipation_time'])
    # both anticipations
    correct_anticipation = df['correct_anticipation_time'] > df['incorrect_anticipation_time']
    incorrect_anticipation = df['correct_anticipation_time'] < df['incorrect_anticipation_time']
    null_anticipation = df['difference_between_fixations'] < time
    reactive = df['reactive_time'] > time

    df.loc[correct_anticipation, 'correct_anticipation'] = 1
    df.loc[incorrect_anticipation, 'incorrect_anticipation'] = 1
    df.loc[null_anticipation, 'correct_anticipation'] = 0
    df.loc[null_anticipation, 'incorrect_anticipation'] = 0
    df.loc[reactive, 'reactive'] = 1

    # valid anticipations
    valid_correct_anticipation = (df['reactive'] == 1) & (df['correct_anticipation'] == 1)
    valid_incorrect_anticipation = (df['reactive'] == 1) & (df['incorrect_anticipation'] == 1)

    df.loc[valid_correct_anticipation, 'valid_correct_anticipation'] = 1
    df.loc[valid_incorrect_anticipation, 'valid_incorrect_anticipation'] = 1

    # add block
    block_one = (df['trial_recoded'] > 12) & (df['trial_recoded'] <= 38)
    block_two = (df['trial_recoded'] > 38) & (df['trial_recoded'] <= 65)

    df.loc[block_one, 'block'] = 'block1'
    df.loc[block_two, 'block'] = 'block2'

    return df


def reduce_data(df):

    drop = df.groupby(['Participant']).apply(lambda x: x.loc[x['trial_recoded'] > 12]).reset_index(drop=True)

    # count in general
    drop_count = drop.groupby(['Participant', 'trial_type']).agg({'valid_correct_anticipation': 'sum',
                                                                  'valid_incorrect_anticipation': 'sum',
                                                                  'reactive': 'sum'}).reset_index(drop=False)

    drop_pivot = drop_count.pivot(index=['Participant'], columns=['trial_type'])

    drop_pivot.columns = ['_'.join(col) for col in drop_pivot.columns.values]
    drop_pivot['valid_correct_anticipation_total'] = drop_pivot['valid_correct_anticipation_ambiguous'] + drop_pivot['valid_correct_anticipation_unambiguous']
    drop_pivot['valid_incorrect_anticipation_total'] = drop_pivot['valid_incorrect_anticipation_ambiguous'] + drop_pivot['valid_incorrect_anticipation_unambiguous']
    drop_pivot['reactive_total'] = drop_pivot['reactive_ambiguous'] + drop_pivot['reactive_unambiguous']

    # per block
    drop_block = df.groupby(['Participant']).apply(lambda x: x.loc[x['trial_recoded'] > 12]).reset_index(drop=True)
    drop_count_block = drop_block.groupby(['Participant', 'trial_type', 'block']).agg({'valid_correct_anticipation': 'sum',
                                                                                 'valid_incorrect_anticipation': 'sum',
                                                                                 'reactive': 'sum'}).reset_index(drop=False)
    drop_count_block_pivot = drop_count_block.pivot(index=['Participant'], columns=['trial_type', 'block'])
    drop_count_block_pivot.columns = ['_'.join(col) for col in drop_count_block_pivot.columns.values]
    drop_count_block_pivot['valid_correct_anticipation_total_block1'] = drop_count_block_pivot['valid_correct_anticipation_ambiguous_block1'] + \
                                                                        drop_count_block_pivot['valid_correct_anticipation_unambiguous_block1']
    drop_count_block_pivot['valid_correct_anticipation_total_block2'] = drop_count_block_pivot['valid_correct_anticipation_ambiguous_block2'] + \
                                                                        drop_count_block_pivot['valid_correct_anticipation_unambiguous_block2']
    drop_count_block_pivot['valid_incorrect_anticipation_total_block1'] = drop_count_block_pivot['valid_incorrect_anticipation_ambiguous_block1'] + \
                                                                          drop_count_block_pivot['valid_incorrect_anticipation_unambiguous_block1']
    drop_count_block_pivot['valid_incorrect_anticipation_total_block2'] = drop_count_block_pivot['valid_incorrect_anticipation_ambiguous_block2'] + \
                                                                          drop_count_block_pivot['valid_incorrect_anticipation_unambiguous_block2']
    drop_count_block_pivot['valid_correct_anticipation_total_block1'] = drop_count_block_pivot['valid_correct_anticipation_ambiguous_block1'] + \
                                                                        drop_count_block_pivot['valid_correct_anticipation_unambiguous_block1']
    drop_count_block_pivot['valid_correct_anticipation_total_block2'] = drop_count_block_pivot['valid_correct_anticipation_ambiguous_block2'] + \
                                                                        drop_count_block_pivot['valid_correct_anticipation_unambiguous_block2']
    drop_count_block_pivot['valid_incorrect_anticipation_total_block1'] = drop_count_block_pivot['valid_incorrect_anticipation_ambiguous_block1'] + \
                                                                          drop_count_block_pivot['valid_incorrect_anticipation_unambiguous_block1']
    drop_count_block_pivot['valid_incorrect_anticipation_total_block2'] = drop_count_block_pivot['valid_incorrect_anticipation_ambiguous_block2'] + \
                                                                          drop_count_block_pivot['valid_incorrect_anticipation_unambiguous_block2']
    drop_count_block_pivot['reactive_total_block1'] = drop_count_block_pivot['reactive_ambiguous_block1'] + \
                                                      drop_count_block_pivot['reactive_unambiguous_block1']
    drop_count_block_pivot['reactive_total_block2'] = drop_count_block_pivot['reactive_ambiguous_block2'] + \
                                                      drop_count_block_pivot['reactive_unambiguous_block2']

    drop_pivot.reset_index(drop=False, inplace=True)
    drop_count_block_pivot.reset_index(drop=False, inplace=True)

    reduced = drop_pivot.merge(drop_count_block_pivot, on='Participant', how='left')

    return reduced




path_one = 'C:/Users/sebas/ownCloud/ATTLENG_18_19/DATA_18_19/COPIA DE SEGURIDAD/EYE TRACKER/SMI_Data_Exported/VSL_modifiedAOIs_dic2019/AOI Statistics - Trial Summary (AOI) - 6dic2019 - AOIs modified 1.txt'
path_two = 'C:/Users/sebas/ownCloud/ATTLENG_18_19/DATA_18_19/COPIA DE SEGURIDAD/EYE TRACKER/SMI_Data_Exported/VSL_modifiedAOIs_dic2019/AOI Statistics - Trial Summary (AOI) - 6dic2019 - AOIs modified 2.txt'
path_three = 'C:/Users/sebas/ownCloud/ATTLENG_18_19/DATA_18_19/COPIA DE SEGURIDAD/EYE TRACKER/SMI_Data_Exported/VSL_modifiedAOIs_dic2019/AOI Statistics - Trial Summary (AOI) - 6dic2019 - AOIs modified 3 (2).txt'
path_four = 'C:/Users/sebas/ownCloud/ATTLENG_18_19/DATA_18_19/COPIA DE SEGURIDAD/EYE TRACKER/SMI_Data_Exported/VSL_modifiedAOIs_dic2019/AOI Statistics - Trial Summary (AOI) - 20feb2020 - AOIs modified 4.txt'

path_one_100ms = 'C:/Users/sebas/ownCloud/ATTLENG_18_19/DATA_18_19/COPIA DE SEGURIDAD/EYE TRACKER/SMI_Data_Exported/VSL_modifiedAOIs_dic2019/Min_fix_100ms/AOI Statistics - Trial Summary (AOI) - 1 - 27Feb2020 - 100msMinFix.txt'
path_two_100ms = 'C:/Users/sebas/ownCloud/ATTLENG_18_19/DATA_18_19/COPIA DE SEGURIDAD/EYE TRACKER/SMI_Data_Exported/VSL_modifiedAOIs_dic2019/Min_fix_100ms/AOI Statistics - Trial Summary (AOI) - 2 - 27Feb2020 - 100msMinFix.txt'
path_three_100ms = 'C:/Users/sebas/ownCloud/ATTLENG_18_19/DATA_18_19/COPIA DE SEGURIDAD/EYE TRACKER/SMI_Data_Exported/VSL_modifiedAOIs_dic2019/Min_fix_100ms/AOI Statistics - Trial Summary (AOI) - 3 - 27Feb2020 - 100msMinFix.txt'
path_four_100ms = 'C:/Users/sebas/ownCloud/ATTLENG_18_19/DATA_18_19/COPIA DE SEGURIDAD/EYE TRACKER/SMI_Data_Exported/VSL_modifiedAOIs_dic2019/Min_fix_100ms/AOI Statistics - Trial Summary (AOI) - 4 - 27Feb2020 - 100msMinFix.txt'


file = pd.read_csv(path_four_100ms, delimiter=',', na_values='-').sort_values(by=['Participant', 'Trial'])

keep_cols = ['Trial', 'Stimulus', 'Participant', 'Eye L/R', 'AOI Name', 'AOI Group',
             'Entry Time [ms]', 'Sequence', 'Dwell Time [ms]', 'Fixation Time [ms]']

file = file[keep_cols]
file['Trial'] = file['Trial'].str.slice(start=5).astype('int')


file_position = position(file)
file_position_trial = trial_type(file_position)
file_position_trial_recoded = trial_number(file_position_trial)
file_position_trial_recoded_reactive = reactive(file_position_trial_recoded)
file_position_trial_recoded_reactive_raw = raw_trial((file_position_trial_recoded_reactive))
file_position_trial_recoded_reactive_raw_subtrial = subtrial(file_position_trial_recoded_reactive_raw)
file_position_trial_recoded_reactive_raw_subtrial_previous = previous_position((file_position_trial_recoded_reactive_raw_subtrial))
file_position_trial_recoded_reactive_raw_subtrial_previous_corr_anticipation = correct_incorrect_anticipations(file_position_trial_recoded_reactive_raw_subtrial_previous)
# keep only right eye
processed = file_position_trial_recoded_reactive_raw_subtrial_previous_corr_anticipation[file_position_trial_recoded_reactive_raw_subtrial_previous_corr_anticipation['Eye L/R'] == 'Right']
pivot = choose_correct_incorrect_anticipation(processed)
valid_ant = valid_anticipations(pivot, 100)
reduced = reduce_data(valid_ant)

reduced.to_csv('C:/Users/sebas/ownCloud/ATTLENG_18_19/fourth_100.csv', sep=';', decimal='.')