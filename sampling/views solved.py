import log_filtering.transformation as trans
import log_filtering.utils as utils
import log_filtering.abstraction_support_functions as asf
import re
from pm4py.algo.discovery.dfg import algorithm as dfg_factory
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.objects.log.importer.xes import importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from wsgiref.util import FileWrapper
from django.http import HttpResponseRedirect, HttpResponse
from datetime import datetime
from os import path
from django.conf import settings
import json
import xml
import matplotlib.pyplot as plt
import time
import math
from scipy.spatial import distance
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics.pairwise import euclidean_distances
import functools
import itertools
import random
import multiprocessing
from multiprocessing import Process, Queue, Pool, cpu_count
from pm4py.objects.log.importer.xes import importer as xes_importer
from numpy.linalg import norm
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.statistics.traces.generic.pandas import case_statistics
from pm4py.statistics.traces.generic.log import case_statistics as csss

from pm4py.util import constants
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.filtering.pandas.variants import variants_filter
from pm4py.algo.filtering.log.end_activities import end_activities_filter
from pm4py.algo.filtering.log.start_activities import start_activities_filter
from pm4py.algo.filtering.pandas.timestamp import timestamp_filter
from pm4py.algo.filtering.log.cases import case_filter
import numpy as np
import pandas.api.types as ptypes
import decimal
import numbers
import pm4py
import pandas as pd
import os
from django.http.response import HttpResponse
from typing import AsyncContextManager
import csv
from tempfile import NamedTemporaryFile
import shutil
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

from pm4py.objects.conversion.log.variants import to_data_frame as log_to_data_frame
from os import listdir

from django.template import RequestContext


import shutil

from django.shortcuts import render
from django.conf import settings
import os
from os import path
from datetime import datetime
from django.http import HttpResponseRedirect, HttpResponse
from wsgiref.util import FileWrapper
from pm4py.objects.log.importer.xes import importer
import json
import re
import log_filtering.abstraction_support_functions as asf
import log_filtering.utils as utils
import log_filtering.transformation as trans
# Create your views here.

from django.views.decorators.csrf import csrf_exempt

case_id = "case:concept:name"
activity_name = "concept:name"
timestamp_time = "time:timestamp"
log_csv = "log"
sam = {}
exported_file = ""
ci = case_id
ac = activity_name

""" 
@csrf_exempt
def test(request):
    return HttpResponse("hello world")


# from pm4py.statistics.traces.pandas import case_statistics


# convert log file(in csv) to pandas dataframe
# read the csv file
log_csv = pd.read_csv('./eventlogs/ItalianHelpdeskFinal.csv', sep=',')
# fill the csv file with the most frequent values of each column
log_csv = log_csv.fillna(log_csv.mode().iloc[0])
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
param_keys = {constants.PARAMETER_CONSTANT_CASEID_KEY: case_id,
                  constants.PARAMETER_CONSTANT_ACTIVITY_KEY: activity_name,
                  constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: timestamp_time
                  }

event_log = log_converter.apply(log_csv, parameters=param_keys)


# import xes file
# q = xes_importer.apply('../eventlogs/repairExample.xes')

# convert xes to dataframe
# dataframe = log_converter.apply(q, variant=log_converter.Variants.TO_DATA_FRAME)

# ci is necessary for access the original case id of log file
# this should be differ from the "Case IDs" which contains the list of case ids
ci = 'Case ID'
ac = 'Activity'

# this method will preprocess the given log_csv
# it means it will organize the dataframe such that

            Case ID
variant
abc         ["Case 1", "Case 23", ...]
acc         ["Case 10", "Case 22", ...]
adc         ["Case 5", "Case 6", ...]
.
.
.
.
.
.

 """
# 1-Traversing the event log:


def preprocessing_csv(log_csv, caseid, activity):

    # organize the variant for each case
    # this csv has the index as the Case ID and the value variant
    variants = case_statistics.get_variants_df(log_csv,
                                               parameters={constants.PARAMETER_CONSTANT_CASEID_KEY: case_id,
                                                           constants.PARAMETER_CONSTANT_ACTIVITY_KEY: activity_name})

    variant_caseid = variants.copy()

    # a. duplicate the case ids as an additional column
    variant_caseid['Case IDs'] = variants.index

    # b. set variant(sequence of acitvities) as the index
    # -> exchange the index from case id to variant
    # duplicated Case Ids contain the case Ids, the case Ids in the index is deleted
    variant_caseid.set_index('variant', inplace=True)

    # c. duplicate variant and store them as an additional column
    variant_caseid['variant2'] = variant_caseid.index

    # d copy the dataframe
    variant_caseid_statistics = variant_caseid.copy()

    # e dataframe that contains collected list of case ids
    variant_caseid_statistics = variant_caseid.groupby(
        "variant2")["Case IDs"].apply(list).to_frame()

    # f. add to variant_caseid_statistics the length of each list Case IDs
    variant_caseid_statistics["casenums"] = variant_caseid_statistics["Case IDs"].str.len(
    )
    # reset index -> assign numbers to each variant
    # variant_caseid_statistics = variant_caseid_statistics.reset_index(drop=True)

    # g. collect every case ids sharing the same variant
    variant_caseid = list(variant_caseid.groupby(
        "variant2")["Case IDs"].apply(list))

    # h. vizualize the statistics
    # variant_caseid_statistics.plot(y = "casenums",  kind="bar", rot=5, fontsize=7, use_index=True, figsize=(50,10))

    # variant_caseid contains all the list of case ids sharing the same variant
    # statistics is the organized dataframe to represent the statistics
    return variant_caseid, variant_caseid_statistics


#variant_caseid, statistics = preprocessing_csv(log_csv, ci, ac)
# 2-Distribution Computation:
############################################################################################################
############################################################################################################
#############################################ATTENTION!!!!!!################################################
############################################################################################################
#############################################################################################################
"""
the [1:4 means just the unnecessary columns like unnecessary data
time, or something
in this case, it is the second, third, and fourth column
"""

# unnecessaries = [1,2,3]


def get_last_activities(logdf):

    # copy the dataframe
    
    log_csv2 = logdf.copy()

    # just keep the last activity record
    last_activities = log_csv2.drop_duplicates(subset=[case_id], keep="last")

    # drop unnecessary columns such as timestamp and resource
    # 1:4 means determined index that are not necessary
    last_activities = last_activities.drop(
        list(last_activities.columns)[1:4], axis=1)

    return last_activities


#last_activities = get_last_activities(log_csv)

# this function splits the given list a in n parts
# such that splitted sublists have roughly same length
# be used : a : caseids, n : # processes = 4 -> split the list of caseids


def split(a, n):

    # the function divmod takes two argument (a1, a2) and returns two results(r1, r2)
    # r1 = a1/a2, r2 = a1 mod a2
    """
    divmod(8, 3) =  (2, 2)
    divmod(7.5, 2.5) =  (3.0, 0.0)
    """
    k, m = divmod(len(a), n)
    return list((a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)))

# filtersingles filter out all the variant with just 1 cases
# be used : l -> list of caseids


def filtersingles(l):

    # variants with just one case
    singles = [i for i in l if len(i) == 1]

    # the left outcome is the list of case ids that share the variant
    # in other words, the collection of case ids of each variant
    return [x for x in l if x not in singles], singles

# this function uses multiprocessing to apply parallel programming paradigm


def filtersingles_mp(a_list):

    # multiprocessing with 4 processes
    PROCESSES = 4

    # starting multiprocessing
    with multiprocessing.Pool(PROCESSES) as pool:

        chunks = split(list(a_list), 4)

        results = [pool.apply_async(filtersingles, [chunk])
                   for chunk in chunks]

        # list comprehension is mostly used because it is a bit faster than
        # the same functionality of a single loop
        nonsingles = [r.get()[0] for r in results]
        singles = [r.get()[1] for r in results]

    return nonsingles, singles


#ns_vci, s_vci = filtersingles(variant_caseid)


def modifyindex(df):

    df = df.set_index(case_id)

    return df


def filter_catcols(logdf):

    # returns the categorical column index of each column
    ccols = [c for c in range(len(list(logdf.columns))) if ptypes.is_numeric_dtype(
        list(logdf.columns)[c]) != True]

    return ccols

# multiply 1 to whole column such that they can be numeric


def preprocessig_df(logdf):

    result_df = logdf.copy()

    for c in range(len(list(result_df.columns))):

        # check if the column is boolean
        if ptypes.is_bool_dtype(list(result_df.columns)[c]) == True:

            # then multiply 1 to whole column such that they can be numeric
            # used if else such that each column can be checked with on condition
            result_df.iloc[:, c].apply(lambda x: x*1)

    ccols = filter_catcols(result_df)

    return result_df, ccols


#last_activities, ccols = preprocessig_df(last_activities)
# this function returns the dataframe part regarding the given caseid
# be used : splits the log dataframe corresponding the given caseid


def filterdf(logdf, caseid):
    print("+++++++++++++++++++++++++++++")
    print(type(logdf))
    print(logdf)
    log = logdf.loc

    return logdf.loc[logdf[case_id].isin(caseid)]



def splitdf(logdf, caseids):

    dfs = [filterdf(logdf, ci) for ci in caseids]
    print("+0000000000000000000000000000000000")
    print(type(dfs))
    print(dfs)

    return dfs

# this function applies splitdf in multiprocessing
# logdf : dataframe to be splitted
# caseids : list of caseid (variant_caseid) will be used


def splitdf_mp(logdf, caseids):

    # the number of processes
    PROCESSES = 4

    # list of dataframes to be returned
    dfs = []

    with multiprocessing.Pool(PROCESSES) as pool:

        # split caseid into 4
        chunks = split(list(caseids), 4)

        results = [pool.apply_async(splitdf, (logdf, chunk[0]))
                   for chunk in chunks]
        dfs = [r.get() for r in results]

    return dfs

# subset of dfs corresponding their single variant
# each single element in the list is dataframe

# dfs_variant = splitdf_mp(last_activities, ns_vci)

# dfs_variant[0][0].iloc[:, 2]


# np.unique([1, 1, 2, 2, 3, 3])
# len(np.unique(dfs_variant[0][0].iloc[:, 3]))
# the list of all dataframe pieces
# dfs_variant_raw
#dfs_variant_raw = splitdf(last_activities, ns_vci)


# One hot encoding
# now apply one hot encoding using every categorical colum from the catcols
# check every list of case id of each variant

# this function applies one hot encoding to the given dataframe regarding the given column names
# df : dataframe to be the one hot encoding applied
# columns : columnnames(should be categorical columns)


def one_hot_encoding(df, columns):

    cols = []

    cols.append(df.iloc[:, 0])

    encoded = [pd.get_dummies(df.iloc[:, c], prefix=df.columns[c])
               for c in columns if c != 0 and len(np.unique(df.iloc[:, c])) != 1]

    cols = cols + encoded

    return pd.concat(cols, axis=1)


def map_one_hot_encoding(dfs, columns):

    one_hot_encoded = []

    # for i in range(len(dfs)):

    # one_hot_encoded.append(list(map(functools.partial(one_hot_encoding, columns=columns), dfs[i])))

    one_hot_encoded = [list(map(functools.partial(
        one_hot_encoding, columns=columns), dfs[i])) for i in range(len(dfs))]

    return one_hot_encoded


def map_one_hotenc(df, columns):

    dfs_variant_raw = splitdf(df, columns)
    ccols = filter_catcols(df)
    return [one_hot_encoding(df, ccols) for df in dfs_variant_raw]


# dfs_variant_encoded = map_one_hot_encoding(dfs_variant_raw, ccols)
# list(map(one_hot_encoding(dfs_variant_raw[0], ccols)
#q = map_one_hotenc(dfs_variant_raw, ccols)
# q


def euclidean_distance(df):

    # set the Case Id(first column) as index
    df = modifyindex(df)

    # add the dist column containing the euclidean distance to every rows
    df['dist'] = pd.Series(np.sum(distance.cdist(
        df.iloc[:, 1:], df.iloc[:, 1:], metric='euclidean'), axis=1), index=df.index)

    # sort this small dataframe by the dist
    df = df.sort_values(by=['dist'])

    # rank this dataframe by the distance
    # frame['sum_order'] = frame['sum'].rank()

    df['dist_order'] = df['dist'].rank()

    return df['dist_order']


def euclidean_distance_list(dfs):

    new_dfs = [euclidean_distance(df) for df in dfs]

    return new_dfs


# this function computes the euclidean distance using multiprocessing
def euclidean_distance_mp(dfs):

    # the number of processes
    PROCESSES = 4

    with multiprocessing.Pool(PROCESSES) as pool:

        results = pool.map(euclidean_distance_list, dfs)

        return results

# dists = list(map(euclidean_distance, q))


def euclidean_distance_map(dfs):

    return list(map(euclidean_distance, dfs))


#dists = euclidean_distance_map(q)
# dists
# returns the list of ones with length n
# whereby n means the number of dataframes in a list
# aims def final_caseids(tp):


def unique_selection_df(df):

    return [df, 1]


def unique_selection_list(dfs):

    return list(map(unique_selection_df, dfs))


def unique_selection(dfs):

    return list(map(unique_selection_list, dfs))


def unique_selection_map(dfs):

    return list(map(unique_selection_df, dfs))

# this function returns Log_2(the number of cases of each variant)


def logarithmic_selection_df(df):

    index = list(df.to_frame().index)
    return round(math.log2(len(index)))


def logarithmic_selection_df_final(df):

    index = list(df.to_frame().index)
    return [df, round(math.log2(len(index)))]


def logarithmic_selection_map(dfs):

    return list(map(logarithmic_selection_df_final, dfs))

# this function returns Log_2(every number of cases of each variant)


def logarithmic_selection_list(dfs):

    nums = []

    for df in dfs:

        nums.append(logarithmic_selection_df(df))

    return list(zip(dfs, nums))

# this function returns Log_2(every number of cases of every variant)


def logarithmic_selection(dfs):

    return list(map(logarithmic_selection_list, dfs))

# this function returns the number of cases of each variant / 2


def division_selection_df(df):

    index = list(df.to_frame().index)
    return math.ceil(len(index)/2)


def division_selection_df_cop(df):

    index = list(df.to_frame().index)
    return [df, math.ceil(len(index)/2)]


def division_selection_map(dfs):

    return list(map(division_selection_df_cop, dfs))

# this function returns the number of every cases of each variant / 2


def division_selection_list(dfs):

    nums = []

    for df in dfs:

        nums.append(division_selection_df(df))

    return list(zip(dfs, nums))

# this function returns all the number of cases of every variant / 2


def division_selection(dfs):

    return list(map(division_selection_list, dfs))


# dists
#after_dist = logarithmic_selection_map(dists)
# after_dist
# return the final case ids
# depending on the selection method, n has different values s.t
# unique selection n -> 1 (default setting)
# logarithmic selection n -> log2( number of traces of each variant)
# division selection n -> round(number of traces of each variant / 2)

# this function takes a tuple of
# (pandas.core.series.Series, n)
# n : the number of cases that should be returned


def final_caseids_single(tp):

    df = tp[0].to_frame()
    return list(df.head(tp[1]).index)


def flatten(t):
    return [item for sublist in t for item in sublist]


def final_caseids_mp(dfs, n=1):

    # the number of processes
    PROCESSES = 4

    # L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])

    with multiprocessing.Pool(PROCESSES) as pool:

        # .starmap(f_sum, data)
        results = pool.map(final_caseids_list, dfs)

    results = flatten(flatten(results))

    # results = [item for sublist in dfs for item in sublist]
    return results


def final_caseids(tps):

    return flatten(list(map(final_caseids_single, tps)))


#fcis = final_caseids(after_dist)


def export_file(df, caseids, s_vci, ci, ac, n=1):

    s_vci = flatten(flatten(s_vci))

    # 2 indicates that it is logarithmic
    # it is just for now!!!!
    if n != 2:

        caseids = caseids + s_vci

    export_csv = df[df[case_id].isin(caseids)]
    export_csv_cop = export_csv.copy()
    filename = str(time.strftime("%Y%m%d-%H%M%S")) + str(".csv")
    export_csv.to_csv(".\media\event_logs\SAMPLEDFILE" +
                      filename, index=True, header=True)
    exported_file= str("SAMPLEDFILE") + filename
    print(" exported file is")
    print(exported_file)

    # organize the given csv such that
    # index       Case IDs                     casenums
    # variant 1   [Case 1]                     1
    # variant 2   [Case 10, Case 11]           2
    # variant 3   [Case 51, Case 8, Case 90]   3
    #        .              .                 .
    #        .              .                 .
    #        .              .                 .
    #        .              .                 .

    caseid_list, statistics = preprocessing_csv(export_csv_cop, ci, ac)
    return export_csv_cop, caseid_list, statistics, exported_file


# this function shows statistics and saves an image file

def showstats(stat1, stat2):

    st1 = stat1.drop(columns=["Case IDs"])
    stat2.head(10)
    st2 = stat2.rename(columns={"casenums": "casenums2"}).drop(
        columns=["Case IDs"])
    stats = st1.join(st2).fillna(0)
    stats = stats.sort_values(by=['casenums'], ascending=False)
    stats = stats.rename(
        columns={'casenums': 'Original', 'casenums2': 'Sampled'})
    stats = stats.reset_index(drop=True)

    axes = stats.plot.bar(rot=0, subplots=True, figsize=(
        30, 20), fontsize=30, width=1.5, edgecolor='white')
    axes[1].legend(loc=1, prop={'size': 20})
    axes[1].set_title('Sampled', fontsize=40)
    axes[0].legend(loc=1, prop={'size': 20})
    axes[0].set_title('Original', fontsize=40)

    filename = str(time.strftime("%Y%m%d-%H%M%S"))

    plt.savefig(".\media\event_logs\STATISTICS_SAMPLED" + filename)
# How to execute all the functions?


# this function encompasses every function from above
# log_csv : the dataframe to be sampled; this should be entered from above
# ci : the column name of Case ID
# ac : the column name of activity
# n : indicates the selected method
# 1 : unique selection
# 2 : logarithmic
# 3 : division


def computation_sampling(log_csv, ci, ac, n):

    variant_caseid, stat1 = preprocessing_csv(log_csv, ci, ac)

    # separate variants between
    # ns_vci : NonSingle Variant Case ID
    # s_vci : Single Variant Case ID
    ns_vci, s_vci = filtersingles(variant_caseid)
    

    last_activities_raw = get_last_activities(log_csv)

    # extract just the very last row of each cases and separate Categorical Columns (ccols)
    last_activities, ccols = preprocessig_df(last_activities_raw)

    # subset of dfs corresponding their single variant
    # each single element in the list is dataframe
    dfs_variant = splitdf(last_activities, ns_vci)

    # One hot encoding
    dfs_variant_encoded = map_one_hotenc(dfs_variant, ccols)

    # euclidean distance is computed and ranked by this
    euclidean_dist = euclidean_distance_map(dfs_variant_encoded)

    if n == 1:

        # computed euclidean distance and return only one
        distances = unique_selection_map(euclidean_dist)

    elif n == 2:

        # computed euclidean distance, sort them and return the rank til based on log2
        distances = logarithmic_selection_map(euclidean_dist)

    else:

        # computed euclidean distance, sort them and return the rank til based on division / 2
        distances = division_selection_map(euclidean_dist)

    # extract final case ids that passed the euclidean_distance_mp
    final_caseid_list = final_caseids(distances)

    # export the csv file corresponding to the imported csv file
    sampled_csv, _, stat2, exported_file= export_file(
        log_csv, final_caseid_list, s_vci, ci, ac, n)

    # show statistic and saves the image file of it automatically
    showstats(stat1, stat2)

    return sampled_csv, exported_file


def go(request, cname, aname, sm):
    ci = cname
    ac = aname
    sm
# read the csv file
    log_csv = pd.read_csv('./eventlogs/ItalianHelpdeskFinal.csv', sep=',')
# fill the csv file with the most frequent values of each column
    log_csv = log_csv.fillna(log_csv.mode().iloc[0])
    log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
    param_keys = {constants.PARAMETER_CONSTANT_CASEID_KEY: 'Case ID',
                  constants.PARAMETER_CONSTANT_ACTIVITY_KEY: 'Activity',
                  constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "Complete Timestamp"
                  }
    event_log = log_converter.apply(log_csv, parameters=param_keys)


# import xes file
# q = xes_importer.apply('../eventlogs/repairExample.xes')

# convert xes to dataframe
# dataframe = log_converter.apply(q, variant=log_converter.Variants.TO_DATA_FRAME)

# ci is necessary for access the original case id of log file
# this should be differ from the "Case IDs" which contains the list of case ids
    ci = 'Case ID'
    ac = 'Activity'

# this method will preprocess the given log_csv
# it means it will organize the dataframe such that

    sam = computation_sampling(log_csv, ci, ac, sm)

    sam
    log_csv
    return render(request, 'upload.html')


"""
ci
CaseId


ac
ActivityName


concept:name
case:concept:name
"""


def tasty(request):
    if request.method == 'POST':
        if "StartSampling1" in request.POST:
            if "log_list" not in request.POST:
                return HttpResponseRedirect(request.path_info)
            mike = "mike"
            return render(request, 'upload.html',{'mike':mike})





""" import csv
with open('/ws2122-lspm/upload_eventlog/data.csv', newline='') as f:
    reader = csv.reader(f)
    row1 = next(reader)  # gets the first line
    for row in reader:
        print(row)       # prints rows 2 and onward
 """


""" def post(self, request, *args, **kwargs):
    selected_method = request.POST['selected_method']
 """


def run(request):
    Slog_attributes = {}
    event_logs_path = os.path.join(settings.MEDIA_ROOT, "event_logs")
    event_log = os.path.join(event_logs_path, settings.EVENT_LOG_NAME)
    n_event_logs_path = os.path.join(settings.MEDIA_ROOT, "none_event_logs")

    # log = importer.apply(event_log)


    if request.method == 'POST':
        selected_method = request.POST.get('selected_method', False)
        case_id = request.POST.get('case_id', False)
        activity_name = request.POST.get('activity_name', False)
        timestamp_time = request.POST.get('timestamp_time', False)




    # log = request.session['log']
    # print(log)
    # selected_method = request.Get.get("selected_method")
    # read the csv file


    #log_csv = pd.read_csv('./media/event_logs/ItalianHelpdeskFinal.csv', sep=',')

    log_csv = pd.read_csv(event_log, sep=',')
    # fill the csv file with the most frequent values of each column
    log_csv = log_csv.fillna(log_csv.mode().iloc[0])
    log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
    param_keys = {constants.PARAMETER_CONSTANT_CASEID_KEY: case_id,
                  constants.PARAMETER_CONSTANT_ACTIVITY_KEY: activity_name,
                  constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: timestamp_time
                  }

    event_log = log_converter.apply(log_csv, parameters=param_keys)

    #preprocessing_caseid(event_log, case_id)

    print("selected_method is " + selected_method)
    print("Selected Case id = " + case_id)
    print("Selected Activity name = " + activity_name)


    if selected_method == "Division":

        x = 0

    elif selected_method == "Unique-Selection":
        x = 1

    elif selected_method == "Logarithmic-distribution":
        x = 2




    # import xes file
    # q = xes_importer.apply('../eventlogs/repairExample.xes')

    # convert xes to dataframe
    # dataframe = log_converter.apply(q, variant=log_converter.Variants.TO_DATA_FRAME)

    # ci is necessary for access the original case id of log file
    # this should be differ from the "Case IDs" which contains the list of case ids
   
    ci = case_id
    ac = activity_name

    # case:concept:name

    # this method will preprocess the given log_csv
    # it means it will organize the dataframe such that

    samppled_file, exported_file = computation_sampling(log_csv, ci, ac, x)
    print("-----------------------------------")
    print(exported_file)

    #Slog_attributes = statistics(sam)
    return render(request, 'upload.html', {'Slog_attributes': Slog_attributes, 'exported_file':exported_file})





############################################################


#######################


def rename_caseid(x):
    
    x = str(x)
    
    if 'Case' in x:
        
        if x.startswith('Case') == False:
            
            #remove the word "Case" from x
            x = x.replace("Case", "")
            x = x.replace("case", "")
            
            #remove all the whitespace from x "".join(x.split())
            x = "".join(x.split())
            
            #remove all the slash from the x 
            x = x.replace("\\", "")
            x = x.replace("/", "")
        
        else :
            
            pass
            
    else :
        
        x = str("Case ") + str(x) 
    
    return str(x)


def preprocessing_caseid(df, col_name):
    
    df[col_name] = df[col_name].apply(rename_caseid)
    
    return df


        
####################################
def statistics(log):
    Slog_attributes = {}

    Slog_attributes['SColumnNamesValues'] = convert_eventlog_to_json(log)

#    eventlogs = [f for f in listdir(event_logs_path) if isfile(join(event_logs_path, f))]


                #Get all the log statistics
    Sno_cases, Sno_events, Sno_variants, Stotal_case_duration, Savg_case_duration, Smedian_case_duration= get_Log_Statistics(log)
    Slog_attributes['Sno_cases'] = Sno_cases
    Slog_attributes['Sno_events'] = Sno_events
    Slog_attributes['Sno_variants'] = Sno_variants
    Slog_attributes['Stotal_case_duration'] = Stotal_case_duration
    Slog_attributes['Savg_case_duration'] = Savg_case_duration
    Slog_attributes['Smedian_case_duration'] = Smedian_case_duration


    
   

    return {Slog_attributes}


########################################
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.conf import settings
import os
from os import listdir
from os.path import isfile, join
from django.http import HttpResponse
from mimetypes import guess_type
from wsgiref.util import FileWrapper
import json
import pandas as pd
from pm4py.objects.conversion.log import variants
from pm4py.objects.log.importer import xes
from pm4py.objects.log.importer.xes import importer as xes_importer_factory
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.log.variants import to_data_frame as log_to_data_frame
import heapq
import pm4py
from pm4py.algo.filtering.dfg import dfg_filtering
import datetime
import numpy as np
import pandas as pd


from array import array


# Create your views here.

filtered_logs = {}


def upload_page(request):
    log_attributes = {}
    event_logs_path = os.path.join(settings.MEDIA_ROOT, "event_logs")
    n_event_logs_path = os.path.join(settings.MEDIA_ROOT, "none_event_logs")

    if request.method == 'POST':
        if request.is_ajax():  # currently is not being used (get commented in html file)
            filename = request.POST["log_name"]
            print('filename = ', filename)
            file_dir = os.path.join(event_logs_path, filename)
            eventlogs = [f for f in listdir(
                event_logs_path) if isfile(join(event_logs_path, f))]

            log = xes_importer_factory.apply(file_dir)
            no_traces = len(log)
            no_events = sum([len(trace) for trace in log])
            log_attributes['no_traces'] = no_traces
            log_attributes['no_events'] = no_events
            print(log_attributes)
            json_respone = {'log_attributes': log_attributes,
                            'eventlog_list': eventlogs}
            return HttpResponse(json.dumps(json_respone), content_type='application/json')
            # return render(request, 'upload.html', {'log_attributes': log_attributes, 'eventlog_list':eventlogs})
        else:
            if "uploadButton" in request.POST:
                if "event_log" not in request.FILES:
                    return HttpResponseRedirect(request.path_info)

                log = request.FILES["event_log"]
                fs = FileSystemStorage(event_logs_path)
                filename = fs.save(log.name, log)
                uploaded_file_url = fs.url(filename)

                eventlogs = [f for f in listdir(
                    event_logs_path) if isfile(join(event_logs_path, f))]
                # eventlogs.append(filename)

                file_dir = os.path.join(event_logs_path, filename)

                # xes_log = xes_importer_factory.apply(file_dir)
                # no_traces = len(xes_log)
                # no_events = sum([len(trace) for trace in xes_log])
                # log_attributes['no_traces'] = no_traces
                # log_attributes['no_events'] = no_events

                return render(request, 'upload.html', {'eventlog_list': eventlogs})

            elif "deleteButton" in request.POST:  # for event logs
                if "log_list" not in request.POST:
                    return HttpResponseRedirect(request.path_info)

                filename = request.POST["log_list"]
                if settings.EVENT_LOG_NAME == filename:
                    settings.EVENT_LOG_NAME = ":notset:"

                eventlogs = [f for f in listdir(
                    event_logs_path) if isfile(join(event_logs_path, f))]
                n_eventlogs = [f for f in listdir(
                    n_event_logs_path) if isfile(join(n_event_logs_path, f))]

                eventlogs.remove(filename)
                file_dir = os.path.join(event_logs_path, filename)
                os.remove(file_dir)
                return render(request, 'upload.html', {'eventlog_list': eventlogs, 'n_eventlog_list': n_eventlogs})

            elif "n_deleteButton" in request.POST:  # for none event logs
                if "n_log_list" not in request.POST:
                    return HttpResponseRedirect(request.path_info)

                filename = request.POST["n_log_list"]

                n_eventlogs = [f for f in listdir(
                    n_event_logs_path) if isfile(join(n_event_logs_path, f))]
                eventlogs = [f for f in listdir(
                    event_logs_path) if isfile(join(event_logs_path, f))]

                n_eventlogs.remove(filename)
                file_dir = os.path.join(n_event_logs_path, filename)
                os.remove(file_dir)
                return render(request, 'upload.html', {'eventlog_list': eventlogs, 'n_eventlog_list': n_eventlogs})

            elif "setButtonN" in request.POST:
                if "log_list" not in request.POST:
                    return HttpResponseRedirect(request.path_info)

                filename = request.POST["log_list"]
                settings.EVENT_LOG_NAME = filename

                file_dir = os.path.join(event_logs_path, filename)
                print("File_dir --------------------" + file_dir)

                log, vars = convert_eventfile_to_log(file_dir)

               
                log_attributes['ColumnNamesValues'] = convert_eventlog_to_json(
                    log)

                eventlogs = [f for f in listdir(
                    event_logs_path) if isfile(join(event_logs_path, f))]

                # Get all the log statistics

                return render(request, 'upload.html',
                              {'eventlog_list': eventlogs, 'log_name': filename, 'vars': vars, 'log_attributes': log_attributes})

            elif "setButton" in request.POST:
                if "log_list" not in request.POST:
                    return HttpResponseRedirect(request.path_info)

                filename = request.POST["log_list"]
                settings.EVENT_LOG_NAME = filename

                file_dir = os.path.join(event_logs_path, filename)


                log, vars = convert_eventfile_to_log(file_dir)

                # Apply Filters on log
                # filters = {
                #     'concept:name': ['Test Repair']
                # }
                # log = filter_log(log, filters, True)

                dfg = log_to_dfg(log, 1, 'Frequency')

                g6, temp_file = dfg_to_g6(dfg)
                dfg_g6_json = json.dumps(g6)

                log_attributes['dfg'] = dfg_g6_json

                # Get all the column names and respective values
                log_attributes['ColumnNamesValues'] = convert_eventlog_to_json(log)

                eventlogs = [f for f in listdir(event_logs_path) if isfile(join(event_logs_path, f))]


                #Get all the log statistics
                no_cases, no_events, no_variants, total_case_duration, avg_case_duration, median_case_duration = get_Log_Statistics(log)
                log_attributes['no_cases'] = no_cases
                log_attributes['no_events'] = no_events
                log_attributes['no_variants'] = no_variants
                log_attributes['total_case_duration'] = total_case_duration
                log_attributes['avg_case_duration'] = avg_case_duration
                log_attributes['median_case_duration'] = median_case_duration


                return render(request, 'upload.html',
                              {'eventlog_list': eventlogs, 'log_name': filename, 'vars': vars, 'log_attributes': log_attributes})



            elif "downloadButton" in request.POST:  # for event logs
                if "log_list" not in request.POST:
                    return HttpResponseRedirect(request.path_info)

                filename = request.POST["log_list"]
                file_dir = os.path.join(event_logs_path, filename)

                try:
                    wrapper = FileWrapper(open(file_dir, 'rb'))
                    response = HttpResponse(
                        wrapper, content_type='application/force-download')
                    response['Content-Disposition'] = 'inline; filename=' + \
                        os.path.basename(file_dir)
                    return response
                except Exception as e:
                    return None

            elif "n_downloadButton" in request.POST:  # for none event logs
                if "n_log_list" not in request.POST:
                    return HttpResponseRedirect(request.path_info)

                filename = request.POST["n_log_list"]
                file_dir = os.path.join(n_event_logs_path, filename)

                try:
                    wrapper = FileWrapper(open(file_dir, 'rb'))
                    response = HttpResponse(
                        wrapper, content_type='application/force-download')
                    response['Content-Disposition'] = 'inline; filename=' + \
                        os.path.basename(file_dir)
                    return response
                except Exception as e:
                    return None

    else:

        # file_dir = os.path.join(settings.MEDIA_ROOT, "Privacy_P6uRPEd.xes")
        # xes_log = xes_importer_factory.apply(file_dir)
        # no_traces = len(xes_log)
        # no_events = sum([len(trace) for trace in xes_log])
        # log_attributes['no_traces'] = no_traces
        # log_attributes['no_events'] = no_events
        eventlogs = [f for f in listdir(
            event_logs_path) if isfile(join(event_logs_path, f))]
        n_eventlogs = [f for f in listdir(
            n_event_logs_path) if isfile(join(n_event_logs_path, f))]

        return render(request, 'upload.html', {'eventlog_list': eventlogs, 'n_eventlog_list': n_eventlogs})

        # return render(request, 'upload.html')


def log_to_dfg(log, percentage_most_freq_edges, type):
    # Discover DFG
    from pm4py.algo.discovery.dfg import algorithm as dfg_discovery

    if type == 'Frequency':
        dfg = dfg_discovery.apply(
            log, variant=dfg_discovery.Variants.FREQUENCY)
    else:
        dfg = dfg_discovery.apply(
            log, variant=dfg_discovery.Variants.PERFORMANCE)

    dfg1, sa, ea = pm4py.discover_directly_follows_graph(log)
    activities_count = pm4py.get_attribute_values(log, activity_name)

    # Filter Frequent Paths
    dfg, sa, ea, activities_count = dfg_filtering.filter_dfg_on_paths_percentage(
        dfg, sa, ea, activities_count, percentage_most_freq_edges)
    return dfg


def dfg_to_g6(dfg):
    unique_nodes = []
    print(dfg)
    for i in dfg:
        unique_nodes.extend(i)
    unique_nodes = list(set(unique_nodes))

    unique_nodes_dict = {}

    for index, node in enumerate(unique_nodes):
        unique_nodes_dict[node] = "node_" + str(index)

    nodes = [{'id': unique_nodes_dict[i], 'name': i, 'isUnique':False, 'conf': [
        {
            'label': 'Name',
            'value': i
        }
    ]} for i in unique_nodes_dict]
    freqList = [int(dfg[i]) for i in dfg]
    maxVal = max(freqList) if len(freqList) != 0 else 0
    minVal = min(freqList) if len(freqList) != 0 else 0

    edges = [{'source': unique_nodes_dict[i[0]], 'target': unique_nodes_dict[i[1]], 'label': round(dfg[i], 2),
              "style": {"lineWidth": ((int(dfg[i]) - minVal) / (maxVal - minVal) * (20 - 2) + 2), "endArrow": True}} for
             i in
             dfg]
    data = {
        "nodes": nodes,
        "edges": edges,
    }

    # Apply freq filtering on edges

    temp_path = os.path.join(settings.MEDIA_ROOT, "temp")
    temp_file = os.path.join(temp_path, 'data.json')
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data, temp_file


def highlight_uncommon_nodes(g61, g62):
    g6dict_1 = json.loads(g61)
    g6dict_2 = json.loads(g62)

    for node in g6dict_1['nodes']:
        if not find_node_in_g6(node['name'], g6dict_2):
            node['isUnique'] = 'True'
        else:
            node['isUnique'] = 'False'

    for node in g6dict_2['nodes']:
        if not find_node_in_g6(node['name'], g6dict_1):
            node['isUnique'] = 'True'
        else:
            node['isUnique'] = 'False'

    return g6dict_1, g6dict_2


def find_node_in_g6(node_name, g6_dict):
    for node in g6_dict['nodes']:
        if node['name'] == node_name:
            return True
    return False


def filter_log(log, filterItemList, isKeepOnlyThese=True):
    from pm4py.algo.filtering.log.attributes import attributes_filter
    filtered_log_events = log

    for key in filterItemList:
        list_of_values = filterItemList[key]
        if (type(list_of_values[0]).__name__ == 'int'):
            filtered_log_events = attributes_filter.apply_numeric_events(filtered_log_events, min(list_of_values),
                                                                         max(list_of_values), parameters={
                attributes_filter.Parameters.ATTRIBUTE_KEY: key,
                attributes_filter.Parameters.POSITIVE: isKeepOnlyThese})
        else:
            filtered_log_events = attributes_filter.apply(filtered_log_events, list_of_values,
                                                          parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: key,
                                                                      attributes_filter.Parameters.POSITIVE: isKeepOnlyThese})

    return filtered_log_events


def convert_eventfile_to_log(file_path):

    file_name, file_extension = os.path.splitext(file_path)

    if file_extension == '.csv':

        log = pd.read_csv(file_path, sep=',')
        charcteristic = list(log.columns.values)

        log = dataframe_utils.convert_timestamp_columns_in_df(log)  



        log = log_converter.apply(log)  # convert to eventlog

        vars = charcteristic

    else:

        log = xes_importer_factory.apply(file_path)
        df1 = log_to_data_frame.apply(log)

        charcteristic = list(df1.columns.values)
        vars = charcteristic

        """ list_of_column_names = []

        for row in log:
            list_of_column_names.append(row)
            break

        vars=[] 
        array = np.array(list_of_column_names)
        for column in array:
            for j in column:
                print(j)
                vars.append(j)
        vars = np.array(vars, dtype=object)
    # df = log_to_data_frame.apply(log) """

    return log, vars


def FilterDataToLogAttributes(FilterData, div_id):
    event_logs_path = os.path.join(settings.MEDIA_ROOT, "event_logs")
    print(FilterData)
    ColName = FilterData['ColumnName']
    if (ColName == "Choose Column"):
        ColName = ""
    ColValue = FilterData['ColumnValue']
    KeepAllExceptThese = FilterData['Checkbox']
    type = FilterData['Type']
    FileName = FilterData['FileName']
    Percentage_most_freq_edges = int(FilterData['FilterPercentage'])
    if (ColValue == "Choose Column Value"):
        ColValue = ""

    # dict = {'name':ColName,'colValue':ColValue,'Checkbox':Checkbox,'Type':Type,'FilterPercentage':FilterPercentage,'FileName':FileName}

    settings.EVENT_LOG_NAME = FileName
    file_dir = os.path.join(event_logs_path, FileName)
    log = convert_eventfile_to_log(file_dir)

    # Apply Filters on log
    if (ColName != ""):
        filters = {
            ColName: [ColValue]
        }
        log = filter_log(log, filters, not KeepAllExceptThese)

    filtered_logs[div_id] = log

    dfg = log_to_dfg(log, Percentage_most_freq_edges, type)

    g6, temp_file = dfg_to_g6(dfg)
    dfg_g6_json = json.dumps(g6)

    log_attributes = {}

    log_attributes['dfg'] = dfg_g6_json

    # Get all the column names and respective values
    log_attributes['ColumnNamesValues'] = convert_eventlog_to_json(log)

    # Get all the log statistics
    no_cases, no_events, no_variants, total_case_duration, avg_case_duration, median_case_duration = get_Log_Statistics(
        log)
    log_attributes['no_cases'] = no_cases
    log_attributes['no_events'] = no_events
    log_attributes['no_variants'] = no_variants
    log_attributes['total_case_duration'] = total_case_duration
    log_attributes['avg_case_duration'] = avg_case_duration
    log_attributes['median_case_duration'] = median_case_duration

    return log_attributes

def AjaxDownload(request):
    req = json.load(request)

    DivIds = {'Lid': req['Ldiv']}

    div_id = int(DivIds['Lid'])

    if div_id in filtered_logs:
        log = filtered_logs[div_id]
        from pm4py.objects.log.exporter.xes import exporter as xes_exporter
        file_dir = 'temp_log.xes'
        xes_exporter.apply(log, file_dir)
    else:
        event_logs_path = os.path.join(settings.MEDIA_ROOT, "event_logs")
        file_dir = os.path.join(event_logs_path, settings.EVENT_LOG_NAME)

    try:
        wrapper = FileWrapper(open(file_dir, 'rb'))
        response = HttpResponse(wrapper, content_type='application/force-download')
        response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_dir)
        return response
    except Exception as e:
        return None

def AjaxCall(request):
    req = json.load(request)

    FilterDataL = req['GraphL']
    FilterDataR = req['GraphR']

    ldiv_id = int(req['Ldiv'])
    rdiv_id = int(req['Rdiv'])

    log_attributes_L = FilterDataToLogAttributes(FilterDataL, ldiv_id)
    log_attributes_R = FilterDataToLogAttributes(FilterDataR, rdiv_id)

    g6L, g6R = highlight_uncommon_nodes(log_attributes_L['dfg'], log_attributes_R['dfg'])

    log_attributes_L['dfg'] = g6L
    log_attributes_R['dfg'] = g6R

    log_attributes_two_sided = {'log_attributes_L': log_attributes_L, 'log_attributes_R': log_attributes_R}

    return HttpResponse(json.dumps(log_attributes_two_sided), content_type="application/json")


def convert_eventlog_to_json(log):
    df = log_to_data_frame.apply(log)

    firstIteration = True
    jsonstr = "{ "
    for col in df:

        if not firstIteration:
            jsonstr += ", "
        else:
            firstIteration = False

        jsonstr += "\"" + col + "\"" + ": "

        uniqueSortedData = pd.Series(df[col].unique()).sort_values(ascending=True)
        uniqueSortedData = uniqueSortedData.reset_index(drop=True)
        jsonstr += uniqueSortedData.to_json(orient="columns", date_format='iso')

    jsonstr += " }"

    return jsonstr


def get_Log_Statistics(log):

    no_cases = len(log)

    no_events = sum([len(trace) for trace in log])
    
    variants = csss.variants_get.get_variants(log)
    no_variants = len(variants)

    all_case_durations = csss.get_all_casedurations(log, parameters={
    csss.Parameters.TIMESTAMP_KEY: timestamp_time})

    total_case_duration = (sum(all_case_durations))

    if no_cases <= 0:
        avg_case_duration = 0
    else:
        avg_case_duration = total_case_duration/no_cases

    median_case_duration = (csss.get_median_caseduration(log, parameters={
        csss.Parameters.TIMESTAMP_KEY: timestamp_time
    }))

    total_case_duration = days_hours_minutes(total_case_duration)

    avg_case_duration = days_hours_minutes(avg_case_duration)
    
    median_case_duration = days_hours_minutes(median_case_duration)

    print(no_cases, no_events, no_variants, total_case_duration, avg_case_duration, median_case_duration)

    return no_cases, no_events, no_variants, total_case_duration, avg_case_duration, median_case_duration


def days_hours_minutes(totalSeconds):
    
    td = datetime.timedelta(seconds = totalSeconds)

    days = td.days
    hours = td.seconds//3600
    minutes = (td.seconds//60)%60
    seconds = td.seconds - hours*3600 - minutes*60

    return str(days) + "d " + str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s"