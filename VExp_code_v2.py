# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 20:44 2023

@author: Sebastian Moyano

PhD Candidate at the Developmental Cognitive Neuroscience Lab (labNCd)
Research Centre for Mind, Brain and Behaviour (CIMCYC)
University of Granada (UGR)
Granada, Spain.

Description:
Modified version of the script to extract the coded behavioural data for the
Violation of Expectation paradigm. This task was employed as part of the
BEXAT project, a longitudinal study aimed at studying the development of
executive attention in infants from 6 months to 4 years of age.

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import os

# =============================================================================
# SET PATHS
# =============================================================================

directory_path = "set_path"
separator = ";"
saving_name = 'set_path'

directory_save = "set_path"


# =============================================================================
# FUNCTION TO LOAD FILES
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
        if filename.endswith(".csv") and filename not in files:
            files.append(filename)

    return files

# =============================================================================
# FUNCTION TO READ EACH FILE
# =============================================================================


def read_file(fname):
    """

    """

    loaded_file = pd.read_csv(fname, sep=separator, header=0, index_col=False)

    return loaded_file

# =============================================================================
# FUNCTION TO REDUCE THE CODE
# =============================================================================


def reduce_code(directory):
    """
    Ignore the following column: "Irruption.onset_irruption1_.

    """

    # load all file names
    all_files = load_files(directory)

    # instantiate empty list to store dfs for each participant
    dfs_out = []

    # iterate and process each participant's file
    for filename in all_files:
        print(f"Working on case {filename}")
        fname = directory_path + "/" + filename
        df = read_file(fname)
        dfs = []

        # ------------------------------------------------------------------------------------------------------------
        # MERGE IRRUPTIONS INTO ONE COLUMN
        # ------------------------------------------------------------------------------------------------------------
        # join irruption events into the same column
        df["Irruption.event1"].fillna(df["Irruption.event2"], inplace=True)
        df["Irruption.event1"].fillna(".", inplace=True)

        # ------------------------------------------------------------------------------------------------------------
        # GET COLUMNS NAMES FOR EACH TYPE OF EVENT
        # ------------------------------------------------------------------------------------------------------------
        # as the events in the original coding scheme are not sorted by time, create
        # one independent DataFrames for Irruption, Behaviour, Orienting and Toy.
        irruption_cols = [col for col in df.columns if "Irruption" in col]
        behaviour_cols = [col for col in df.columns if "Behaviour" in col]
        orienting_cols = [col for col in df.columns if "Orienting" in col]
        toy_cols = [col for col in df.columns if "Toy" in col]

        # ------------------------------------------------------------------------------------------------------------
        # SLICE DATAFRAMES (WE KEEP TOY COLUMNS WITH BEHAVIOUR AS WE WILL USE IT FOR APPROXIMATION EVENTS)
        # ------------------------------------------------------------------------------------------------------------
        # filter each DataFrame keeping only the columns of interest.
        df_irruption = df[irruption_cols]
        df_behaviour = df[behaviour_cols + toy_cols]
        df_orienting = df[orienting_cols]

        # ------------------------------------------------------------------------------------------------------------
        # ONSET OF THE IRRUPTION
        # ------------------------------------------------------------------------------------------------------------
        # get the onset of the first irruption to remove all Behaviour and Orienting
        # events after the first irruption onset. If there is no irruption, then set irruption
        # time to +inf, otherwise get the time of the first irruption
        if df_irruption["Irruption.onset"].isnull().all():
            first_irruption_onset = np.inf
        else:
            first_irruption_onset = df_irruption["Irruption.onset"].nsmallest(1)[0]

        # ------------------------------------------------------------------------------------------------------------
        # CLEAN DATAFRAMES REMOVING EVENTS THAT TAKE PLACE AFTER AN IRRUPTION EVENT
        # ------------------------------------------------------------------------------------------------------------
        # remove all behaviour, orienting and toy events whose onsets are greater than the
        # onset of the first irruption event.
        df_behaviour = df_behaviour[df_behaviour["Behaviour.onset"] < first_irruption_onset]
        df_orienting = df_orienting[df_orienting["Orienting.onset"] < first_irruption_onset]

        # if the irruption takes place during a certain behaviour, orienting or toy event,
        # then replace the offset of that event for the onset of the irruption. That is,
        # as the onset of the event is lower that the onset of the irruption, the event is
        # valid until the moment the irruption starts..
        df_behaviour.loc[
            df_behaviour["Behaviour.offset"] > first_irruption_onset, "Behaviour.offset"] = first_irruption_onset
        df_orienting.loc[
            df_orienting["Orienting.offset"] > first_irruption_onset, "Orienting.offset"] = first_irruption_onset

        # ------------------------------------------------------------------------------------------------------------
        # START PROCESSING STEPS
        # ------------------------------------------------------------------------------------------------------------
        # if the DataFrames are empty, skip all the subsequent steps
        if df_behaviour.empty & df_orienting.empty:
            pass
        else:
            # --------------------------------------------------------------------------------------------------------
            # GET DATA FOR BEHAVIOUR
            # --------------------------------------------------------------------------------------------------------
            # check if we have some data left in the DataFrame, otherwise skip the processing steps
            if df_behaviour.empty:
                pass
            else:
                # compute difference between offset and onset for each Behaviour event
                df_behaviour["Behaviour.time"] = df_behaviour["Behaviour.offset"] - df_behaviour["Behaviour.onset"]

                # get the onset of each behaviour
                df_onset_beh1 = df_behaviour.groupby(["Behaviour.beh1"], as_index=False)["Behaviour.onset"].first()
                df_onset_beh1 = pd.pivot_table(df_onset_beh1,
                                               values="Behaviour.onset",
                                               columns="Behaviour.beh1",
                                               aggfunc="sum").reset_index(drop=True)
                df_onset_beh1.columns = ["Behaviour1.onset_" + col for col in df_onset_beh1.columns]

                df_onset_beh2 = df_behaviour.groupby(["Behaviour.beh2"], as_index=False)["Behaviour.onset"].first()
                df_onset_beh2 = pd.pivot_table(df_onset_beh2,
                                               values="Behaviour.onset",
                                               columns="Behaviour.beh2",
                                               aggfunc="sum").reset_index(drop=True)
                df_onset_beh2.columns = ["Behaviour2.onset_" + col for col in df_onset_beh2.columns]

                # get total time for each behaviour
                df_time_beh1 = df_behaviour.groupby(["Behaviour.beh1"], as_index=False)["Behaviour.time"].sum()
                df_time_beh1 = pd.pivot_table(df_time_beh1, values="Behaviour.time",
                                              columns="Behaviour.beh1",
                                              aggfunc="sum").reset_index(drop=True)
                df_time_beh1.columns = ["Behaviour1.time_" + col for col in df_time_beh1.columns]

                df_time_beh2 = df_behaviour.groupby(["Behaviour.beh2"], as_index=False)["Behaviour.time"].sum()
                df_time_beh2 = pd.pivot_table(df_time_beh2, values="Behaviour.time",
                                              columns="Behaviour.beh2",
                                              aggfunc="sum").reset_index(drop=True)
                df_time_beh2.columns = ["Behaviour2.time_" + col for col in df_time_beh2.columns]

                # total behaviour time (adding values for all behaviours except 'A')
                df_time_beh1["Behaviour1.total_time"] = np.sum(
                    df_time_beh1[[col for col in df_time_beh1.columns if "Behaviour1.time_A" not in col]].values)

                df_time_beh2["Behaviour2.total_time"] = np.sum(
                    df_time_beh2[[col for col in df_time_beh2.columns if "Behaviour2.time_A" not in col]].values)

                # append to dfs list
                dfs.append(df_time_beh1)
                dfs.append(df_time_beh2)

                # get the onset of the first approximation for toy 1 and toy 2
                # if we have an approximation behaviour coded for toy 1, then get the onset of the first approximation
                # if we don't, then set it to np.inf. This would allow to choose the smallest one when comparing
                # onsets between toys
                if (df_behaviour["Behaviour.beh1"].eq("A")).any():
                    A_toy1 = df_behaviour[df_behaviour["Behaviour.beh1"] == "A"]["Behaviour.onset"].iloc[0]
                else:
                    A_toy1 = np.inf
                # if we have an approximation behaviour coded for toy 1, then get the onset of the first approximation
                if (df_behaviour["Behaviour.beh2"].eq("A")).any():
                    A_toy2 = df_behaviour[df_behaviour["Behaviour.beh2"] == "A"]["Behaviour.onset"].iloc[0]
                else:
                    A_toy2 = np.inf

            # --------------------------------------------------------------------------------------------------------
            # GET DATA FOR ORIENTING
            # --------------------------------------------------------------------------------------------------------
            if df_orienting.empty:
                pass
            else:
                # compute difference between offset and onset for each Orienting event
                df_orienting["Orienting.time"] = df_orienting["Orienting.offset"] - df_orienting["Orienting.onset"]

                # get total time of orienting for each toy
                df_orient_toy = df_orienting.groupby(["Orienting.object"], as_index=False)["Orienting.time"].sum()
                df_orient_toy = pd.pivot_table(df_orient_toy, values="Orienting.time",
                                               columns="Orienting.object",
                                               aggfunc="sum").reset_index(drop=True)
                df_orient_toy.columns = ["Orienting.time_" + col for col in df_orient_toy.columns]

                dfs.append(df_orient_toy)

            # --------------------------------------------------------------------------------------------------------
            # CONCATENATE DATA
            # --------------------------------------------------------------------------------------------------------
            if len(dfs) == 1:
                df_all = dfs[0]
            else:
                df_all = pd.concat(dfs, axis=1)

            # --------------------------------------------------------------------------------------------------------
            # GET THE IDENTITY OF EACH TOY
            # --------------------------------------------------------------------------------------------------------
            toy1 = df["Toy.toy1"].dropna().unique()
            toy2 = df["Toy.toy2"].dropna().unique()

            # --------------------------------------------------------------------------------------------------------
            # FIRST APPROXIMATION TO EACH TOY
            # --------------------------------------------------------------------------------------------------------
            # if we have an approximation behaviour coded for toy 1, then get the onset of the first approximation
            df_all["Toy1.A_onset"] = A_toy1
            # if we have an approximation behaviour coded for toy 1, then get the onset of the first approximation
            df_all["Toy2.A_onset"] = A_toy2

            # --------------------------------------------------------------------------------------------------------
            # LOOK FOR THE PREFERRED TOY
            # --------------------------------------------------------------------------------------------------------
            preferred_toy = [toy1 if df_all["Toy1.A_onset"].iloc[0] < df_all["Toy2.A_onset"].iloc[0] else toy2]

            # --------------------------------------------------------------------------------------------------------
            # ADD FINAL COLUMNS
            # --------------------------------------------------------------------------------------------------------
            df_all["Toy.preferred"] = preferred_toy
            df_all["first_irruption_onset"] = first_irruption_onset
            df_all["code"] = filename
            df_all.reset_index(drop=True, inplace=True)
            # --------------------------------------------------------------------------------------------------------
            # APPEND TO LIST OF SUBJECT DATAFRAMES TO CONCATENATE
            # --------------------------------------------------------------------------------------------------------
            dfs_out.append(df_all)

    out = pd.concat(dfs_out)
    sort_behaviour = [col for col in out.columns if "Behaviour" in col]
    sort_orienting = [col for col in out.columns if "Orienting" in col]
    sort_toy = [col for col in out.columns if "Toy" in col]

    return out[["code", "first_irruption_onset"] + sort_behaviour + sort_orienting + sort_toy]


# call function
resultados = reduce_code(directory_path)

# save data to csv file
resultados.to_csv(os.path.join(directory_save, saving_name), sep=';', decimal=',')
