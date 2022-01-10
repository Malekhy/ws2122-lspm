import os
import pandas as pd
import pm4py
import numbers
import decimal
import pandas.api.types as ptypes
import numpy as np
from pm4py.algo.filtering.log.cases import case_filter
from pm4py.algo.filtering.pandas.timestamp import timestamp_filter
from pm4py.algo.filtering.log.end_activities import end_activities_filter
from pm4py.algo.filtering.pandas.variants import variants_filter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util import constants
from pm4py.statistics.traces.generic.pandas import case_statistics

from pm4py.algo.filtering.log.attributes import attributes_filter
from numpy.linalg import norm
from pm4py.objects.log.importer.xes import importer as xes_importer
from multiprocessing import Process, Queue, Pool, cpu_count
import multiprocessing
import random
import itertools
import functools
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import distance
import math
import time
import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import render
from pm4py.objects.log.importer.xes import importer as xes_importer_factory
from pm4py.objects.conversion.log.variants import to_data_frame as log_to_data_frame
from django.http import HttpResponse, HttpResponseRedirect
import json
from pm4py.statistics.traces.generic.log import case_statistics as case_stat
import datetime










pd.set_option('display.max_columns', None)


#this function modifies all the case id column value s.t. every case id look like
#Case 1637838
#Case 6869797
#.
#.
case_id = "case:concept:name"
activity_name = "concept:name"
timestamp_time = "time:timestamp"
log_csv = "log"
sam = {}
exported_file = ""
ci = case_id
ac = activity_name



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

def abs_numcols(x):
    
    return abs(x)

#how to use this function:
#1. grab the column name of the case id(in this example, it is CustomerID)
#2. assing the applied column to the original column like below
#Example : cops["CustomerID"] = cops["CustomerID"].apply(rename_caseid)

def preprocessing_caseid(df, case_column):
    

    col_name  = case_column
    index_no  = df.columns.get_loc(col_name)

    df[index_no ] = df[index_no ].apply(rename_caseid)
    return df



def preprocessing_caseid1(df, col_name):
    
    df[col_name] = df[col_name].apply(rename_caseid)
    
    return df



#convert log file(in csv) to pandas dataframe

"""
QUESTIONS!
importing scheme

1.the major attributes : Case ID, Activity, Complete Timestamp, and org(resource or something)
They should be ordered such that the rest can be considered as case attributes
!!!!!!!this should be done either for xes or csv, also how to parse the given event log file such that
they can be managed in the form Case ID, Acitivity, Timestamp, and resource? is there any scheme to detect which colume
corresponds to which attribute?

2.importing method should deliver also the file path such that this program
can choose either xes or csv file converting

3. what is an empty value exists?

4. organize log file such that
caseid should be the first column
the others resource, timestamp

#It could make the whole process easy
5. allow user to select/organize columns by checking the name
??

Weakness:

1. still depending on the column orders
2. to often used the Case ID
3. def export_file(df, caseids, s_vci, n = 1): find the scheme s.t. 
n indicates unique selection, logarithmic, and division selection

"""

#read the csv file

#event_log = log_converter.apply(log_csv,parameters=param_keys)


#import xes file
#q = xes_importer.apply('../eventlogs/repairExample.xes')

#convert xes to dataframe
#dataframe = log_converter.apply(q, variant=log_converter.Variants.TO_DATA_FRAME)

#ci is necessary for access the original case id of log file
#this should be differ from the "Case IDs" which contains the list of case ids


#this method will preprocess the given log_csv
#it means it will organize the dataframe such that
"""
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
#log_csv
#mys.isnull().all()
#df.columns[df.isna().any()].tolist()

"""
Idea
drop columns thatsdfsd
contains nan
and # unique value =< 2
the others fill na with the most frequent value
"""
#df.name.mode() ECMO
def fill_mode(df):
    
    df = df.fillna(df.mode().iloc[0])
    
    return df

def preprocessing_caseid_nan(df, col_name):
    
    
    droppings = df.columns[df.isna().any()].tolist()
    fillnas = []
    
    for d in droppings:
        
        if len(list(df[d].unique())) > 2:
            
            droppings.remove(d)

        else :
            
            pass
    
    df = df.drop(droppings, axis=1)
    
    df[col_name] = df[col_name].apply(rename_caseid)
    df = df.apply(lambda col:fill_mode(col))
    
    return df
## 1-Traversing the event log:

def preprocessing_csv(log_csv, caseid, activity):
    
    #organize the variant for each case
    #this csv has the index as the Case ID and the value variant
    variants = case_statistics.get_variants_df(log_csv,
                                          parameters={constants.PARAMETER_CONSTANT_CASEID_KEY: caseid,
                                                      constants.PARAMETER_CONSTANT_ACTIVITY_KEY: activity})
    
    
    variant_caseid = variants.copy()
    
    #a. duplicate the case ids as an additional column
    variant_caseid['Case IDs'] = variants.index
    
    #b. set variant(sequence of acitvities) as the index
    #-> exchange the index from case id to variant
    # duplicated Case Ids contain the case Ids, the case Ids in the index is deleted
    variant_caseid.set_index('variant', inplace=True)
    
    #c. duplicate variant and store them as an additional column
    variant_caseid['variant2'] = variant_caseid.index
    
    #d copy the dataframe
    variant_caseid_statistics = variant_caseid.copy()
    
    #e dataframe that contains collected list of case ids
    variant_caseid_statistics = variant_caseid.groupby("variant2")["Case IDs"].apply(list).to_frame()
    
    #f. add to variant_caseid_statistics the length of each list Case IDs
    variant_caseid_statistics["casenums"] = variant_caseid_statistics["Case IDs"].str.len()
    # reset index -> assign numbers to each variant
    #variant_caseid_statistics = variant_caseid_statistics.reset_index(drop=True)
    
    #g. collect every case ids sharing the same variant
    variant_caseid = list(variant_caseid.groupby("variant2")["Case IDs"].apply(list))
    
    #h. vizualize the statistics
    #variant_caseid_statistics.plot(y = "casenums",  kind="bar", rot=5, fontsize=7, use_index=True, figsize=(50,10))
    
    #variant_caseid contains all the list of case ids sharing the same variant
    #statistics is the organized dataframe to represent the statistics
    return variant_caseid, variant_caseid_statistics
    
## 2-Distribution Computation:
############################################################################################################
############################################################################################################
#############################################ATTENTION!!!!!!################################################
############################################################################################################
#############################################################################################################
"""
the [1:4 means just the unnecessary columns like unnecessary data 
time, or something
in this case, it is the second, third, and fourth column

unnecessaries=["concept:name",
               "dismissal",
               "expense",
               "lastSent",
               "lifecycle:transition",
               "notificationType",
               "org:resource",
               "paymentAmount",
               "points",
               "time:timestamp"]
"""


#df.drop(columns, inplace=True, axis=1)
def intersection(lst1, lst2):
   

    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def get_last_activities1(logdf, ci, unnecessaries = []):
    
    #copy the dataframe
    log_csv2 = logdf.copy()
    
    #just keep the last activity record
    last_activities = log_csv2.drop_duplicates(subset=[str(ci)], keep = "last")
    
    #drop unnecessary columns such as timestamp and resource
    #1:4 means determined index that are not necessary
    #intersection(lst1, lst2)
    unnecessaries = intersection(unnecessaries, list(logdf.columns))
    last_activities = last_activities.drop(unnecessaries, axis = 1)
    
    return last_activities


def get_last_activities(logdf, ci):

    # copy the dataframe
    
    log_csv2 = logdf.copy()

    # just keep the last activity record
    last_activities = log_csv2.drop_duplicates(subset=[str(ci)], keep="last")

    # drop unnecessary columns such as timestamp and resource
    # 1:4 means determined index that are not necessary
    last_activities = last_activities.drop(
        list(last_activities.columns)[1:4], axis=1)

    return last_activities


#this function splits the given list a in n parts
#such that splitted sublists have roughly same length
#be used : a : caseids, n : # processes = 4 -> split the list of caseids
def split(a, n):
    
    #the function divmod takes two argument (a1, a2) and returns two results(r1, r2)
    #r1 = a1/a2, r2 = a1 mod a2
    """
    divmod(8, 3) =  (2, 2)
    divmod(7.5, 2.5) =  (3.0, 0.0)
    """
    k, m = divmod(len(a), n)
    return list((a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)))

#filtersingles filter out all the variant with just 1 cases
#be used : l -> list of caseids
def filtersingles(l):
    
    #variants with just one case
    singles = [i for i in l if len(i) == 1]
            
    #the left outcome is the list of case ids that share the variant
    # in other words, the collection of case ids of each variant
    return [x for x in l if x not in singles], singles

#this function uses multiprocessing to apply parallel programming paradigm 
def filtersingles_mp(a_list):
    
    #multiprocessing with 4 processes
    PROCESSES = 4
    
    #starting multiprocessing
    with multiprocessing.Pool(PROCESSES) as pool:
    
        chunks = split(list(a_list), 4)
    
        results = [pool.apply_async(filtersingles, [chunk]) for chunk in chunks]
        
        #list comprehension is mostly used because it is a bit faster than
        #the same functionality of a single loop
        nonsingles = [r.get()[0] for r in results]
        singles = [r.get()[1] for r in results]
            
    return nonsingles, singles


def modifyindex(df, ci): 
    df = df.set_index(ci)
    
    return df
def filter_catcols(logdf, ci):
    
    #returns the categorical column index of each column
    ccols = [c for c in range(len(list(logdf.columns))) if ptypes.is_numeric_dtype(logdf[list(logdf.columns)[c]]) != True]


    

    caseid_index = logdf.columns.get_loc(str(ci))
    #.columns.get_loc("case:concept:name")
    
    if isinstance(ccols, list):
        
        if len(ccols) >= 1:
    
            if caseid_index in ccols:
            
                ccols.remove(caseid_index)
    
    return ccols





def filter_catcols1(logdf, ci):

    # returns the categorical column index of each column
    ccols = [c for c in range(len(list(logdf.columns))) if ptypes.is_numeric_dtype(
        list(logdf.columns)[c]) != True]

  

    return ccols
#multiply 1 to whole column such that they can be numeric


def preprocessig_df(logdf, ci):
    
    result_df = logdf.copy()
    

    for c in range(len(list(result_df.columns))):
        
        #check if the column is boolean
        if ptypes.is_bool_dtype(list(result_df.columns)[c]) == True:
            
            #then multiply 1 to whole column such that they can be numeric
            #used if else such that each column can be checked with on condition 
            result_df.iloc[:, c].apply(lambda x: x*1)
        
        #check if the column is numeric
        elif ptypes.is_numeric_dtype(list(result_df.columns)[c]) == True:
            
            #if yes, convert it to the absolute value
            result_df.iloc[:, c].apply(abs_numcols)
                
        else :
            
            pass
            
    ccols = filter_catcols(result_df, ci)
            
    return result_df, ccols




def is_nested_list(l):
    
    try:
          next(x for x in l if isinstance(x,list))
    
    except StopIteration:
        return False
    
    return True

#this function returns the dataframe part regarding the given caseid
#be used : splits the log dataframe corresponding the given caseid
#col_name : name of the caseid
def filterdf1(logdf, caseid, col_name):

    return logdf.loc[logdf[col_name].isin(caseid)]


def filterdf222(logdf, caseid, col_name):

    log = logdf.loc[logdf[case_id].isin(caseid)]

    return log

def filterdf(logdf, caseid, col_name):

    return logdf.loc[logdf[col_name].isin(caseid)]

#logdf : dataframe to be splitted
#caseids : list of caseid (variant_caseid) will be used
def splitdf(logdf, caseids, col_name):
    
    dfs = [filterdf(logdf, ci, col_name) for ci in caseids]    
    
    return dfs
#One hot encoding
#now apply one hot encoding using every categorical colum from the catcols     
#check every list of case id of each variant

#this function applies one hot encoding to the given dataframe regarding the given column names
#df : dataframe to be the one hot encoding applied
#columns : columnnames(should be categorical columns)
def one_hot_encoding(df, columns):
    
    cols = []
    
    encoded = [pd.get_dummies(df.iloc[:, c], prefix = df.columns[c]) for c in columns if len(list(np.unique(df.iloc[:, c]))) != 1]
    
    if encoded is not None:
    
        cols = cols + encoded
    
    df.drop(df.columns[columns],axis=1,inplace=True)
    
    cols.append(df)
    
    if len(cols) > 1:
        
        new_df = pd.concat(cols, axis=1)
        
    return df

def map_one_hot_encoding(dfs, columns):
    
    one_hot_encoded = [list(map(functools.partial(one_hot_encoding, columns=columns), dfs[i])) for i in range(len(dfs))]
        
    return one_hot_encoded

def map_one_hotenc(dfs, ccols):
    
    return [one_hot_encoding(df, ccols) for df in dfs]


def euclidean_distance(df, ci):
    
    #set the Case Id(first column) as index
    df = modifyindex(df, ci)

    #add the dist column containing the euclidean distance to every rows
    df['dist']=pd.Series(np.sum(distance.cdist(df.iloc[:,1:], df.iloc[:,1:], metric='euclidean'), axis = 1), index = df.index)
    
    #sort this small dataframe by the dist
    df = df.sort_values(by=['dist'])
    
    #rank this dataframe by the distance
    #frame['sum_order'] = frame['sum'].rank()
    
    df['dist_order'] = df['dist'].rank()

    return df['dist_order']

def euclidean_distance_list(dfs, ci):
    


    new_dfs = [euclidean_distance(df, str(ci)) for df in dfs]
        
    return new_dfs

#dists = list(map(euclidean_distance, q))
def euclidean_distance_map(dfs, ci):
    
    new_dfs = [euclidean_distance_list(df_l, ci) for df_l in dfs]
    
    return new_dfs
#returns the list of ones with length n
#whereby n means the number of dataframes in a list
#aims def final_caseids(tp):
def unique_selection_df(df):
    
    return [df, 1]

def unique_selection_list(dfs):
    
    return list(map(unique_selection_df, dfs))

def unique_selection(dfs):
        
    return list(map(unique_selection_list, dfs))

def unique_selection_map(dfs):
    
    return list(map(unique_selection_df, dfs))

#this function returns Log_2(the number of cases of each variant)
def logarithmic_selection_df(df):
    
    index = list(df.to_frame().index)
    return round(math.log2(len(index)))

def logarithmic_selection_df_final(df):
    
    index = list(df.to_frame().index)
    return [df, round(math.log2(len(index)))]

def logarithmic_selection_map(dfs):
    
    return list(map(logarithmic_selection_df_final, dfs))

#this function returns Log_2(every number of cases of each variant)
def logarithmic_selection_list(dfs):
    
    nums = []
    
    for df in dfs:
        
        nums.append(logarithmic_selection_df(df))
        
    return list(zip(dfs, nums))

#this function returns Log_2(every number of cases of every variant)
def logarithmic_selection(dfs):
    
    return list(map(logarithmic_selection_list, dfs))

#this function returns the number of cases of each variant / 2
def division_selection_df(df):
    
    index = list(df.to_frame().index)
    return math.ceil(len(index)/2)

def division_selection_df_cop(df):
    
    index = list(df.to_frame().index)
    return [df, math.ceil(len(index)/2)]



#this function returns the number of every cases of each variant / 2
def division_selection_list(dfs):
    
    nums = []
    
    for df in dfs:
        
        nums.append(division_selection_df(df))
        
    return list(zip(dfs, nums))

def division_selection(dfs):
    
    return list(map(division_selection_df_cop, dfs))


#unique_selection_map
#logarithmic_selection
#division_selection
#return the final case ids
#depending on the selection method, n has different values s.t
# unique selection n -> 1 (default setting)
# logarithmic selection n -> log2( number of traces of each variant)
# division selection n -> round(number of traces of each variant / 2)

#this function takes a tuple of 
# (pandas.core.series.Series, n)
# n : the number of cases that should be returned
def final_caseids_single(tp):
    
    df = tp[0].to_frame()
    return list(df.head(tp[1]).index)
    

def flatten(t):
    return [item for sublist in t for item in sublist]

def final_caseids_mp(dfs, n = 1):
    
    # the number of processes
    PROCESSES = 4
    
    #L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
    
    with multiprocessing.Pool(PROCESSES) as pool:
        
        #.starmap(f_sum, data)
        results = pool.map(final_caseids_list, dfs)
        
    results = flatten(flatten(results))
    
    #results = [item for sublist in dfs for item in sublist]
    return results

def final_caseids(tps):
    
    return flatten(list(map(final_caseids_single, tps)))
def export_file(df, caseids, s_vci, ci, ac, n ,selected_method, log_name):
    
    if (s_vci) and (isinstance(s_vci[0], list)):
    
        s_vci = flatten(s_vci)
    
    if n != 2: 
        
        caseids = caseids + s_vci

    export_csv = df[df[case_id].isin(caseids)]
    export_csv_cop = export_csv.copy()
    filename = str(" - ") + str(log_name) + str(time.strftime("-%H%M%S")) + str(".csv")
    export_csv.to_csv(".\media\event_logs\LSPM - " + selected_method +
                      filename, index=True, header=True)




    exported_file=  str("LSPM - ") + selected_method + filename
 



    
    #organize the given csv such that 
    #index       Case IDs                     casenums
    #variant 1   [Case 1]                     1
    #variant 2   [Case 10, Case 11]           2
    #variant 3   [Case 51, Case 8, Case 90]   3   
    #        .              .                 .
    #        .              .                 .
    #        .              .                 .
    #        .              .                 .
    
    caseid_list, statistics = preprocessing_csv(export_csv_cop, ci, ac)
    return export_csv_cop, caseid_list, statistics, exported_file
    

#this function shows statistics and saves an image file

def showstats(stat1,stat2, selected_method, log_name):
    
    st1 = stat1.drop(columns=["Case IDs"])
    stat2.head(10)
    st2 = stat2.rename(columns={"casenums": "casenums2"}).drop(columns=["Case IDs"])
    stats = st1.join(st2).fillna(0)
    stats = stats.sort_values(by=['casenums'], ascending=False)
    stats = stats.rename(columns={'casenums': 'Original', 'casenums2': 'Sampled'})
    stats = stats.reset_index(drop=True)
    
    axes = stats.plot.bar(rot=0, subplots=True, figsize=(30,20), fontsize=30, width = 1.5, edgecolor='white')
    axes[1].legend(loc=1, prop={'size': 20})
    axes[1].set_title('Sampled', fontsize=40)
    axes[0].legend(loc=1, prop={'size': 20})
    axes[0].set_title('Original', fontsize=40)
    
    filename =str(" - ") +  str(log_name) + str(time.strftime("-%H%M%S")) + str(".png")

    plt.savefig(".\media\event_logs\LSPM - " + selected_method + filename)
## How to execute all the functions?

#this function encompasses every function from above
#log_csv : the dataframe to be sampled; this should be entered from above
#ci : the column name of Case ID
#ac : the column name of activity
#n : indicates the selected method
#1 : unique selection
#2 : logarithmic 
#3 : division

#samppled_file, exported_file = computation_sampling(log_csv, ci, ac, x)

#the unnecessaries is the list for the activity, timestamp and resources
def computation_sampling(log_csv, ci, ac,  n, selected_method, log_name):
    
    #log_csv_for_final = preprocessing_caseid(log_csv, ci)
    
    log_csv = preprocessing_caseid_nan(log_csv, ci)
    
    variant_caseid, stat1 = preprocessing_csv(log_csv, ci, ac)

    #separate variants between
    #ns_vci : NonSingle Variant Case ID
    #s_vci : Single Variant Case ID
    ns_vci, s_vci = filtersingles_mp(variant_caseid)
    
    last_activities_raw = get_last_activities(log_csv, ci)

    

    #extract just the very last row of each cases and separate Categorical Columns (ccols)
    last_activities, ccols = preprocessig_df(last_activities_raw, ci)
    
    #print(last_activities) ok!

    #subset of dfs corresponding their single variant
    #each single element in the list is dataframe
    
    
    dfs_variant = []
    #if ns_vci -> list(list(list))
    # is_nested_list(ns_vci[0]) true
    # else 
    # false
    

    if is_nested_list(ns_vci[0]):
        
        for i in range(len(ns_vci)):
            
            dfs_variant += splitdf(last_activities, ns_vci[i], ci)
            
    else :
        
        dfs_variant = splitdf(last_activities, ns_vci, ci)
    
    #print("Variant:")
    #print(dfs_variant) not ok

    #One hot encoding
    

    dfs_variant_encoded = map_one_hotenc(dfs_variant, ccols)

  

    
    #print(dfs_variant_encoded)

    #euclidean distance is computed and ranked by this
    euclidean_dist = euclidean_distance_list(dfs_variant_encoded, ci)
    
    #print(euclidean_dist)

    if n == 1:
        
        #computed euclidean distance and return only one
        distances = unique_selection_map(euclidean_dist)
        
    elif n == 2 :
        
        #computed euclidean distance, sort them and return the rank til based on log2
        distances = logarithmic_selection_map(euclidean_dist)
        
    else :
        
        #computed euclidean distance, sort them and return the rank til based on division / 2
        distances = division_selection(euclidean_dist) 
    
    
    #extract final case ids that passed the euclidean_distance_mp
    final_caseid_list = final_caseids(distances)
    
    print(final_caseid_list)

    #export the csv file corresponding to the imported csv file
    sampled_csv, _, stat2,exported_file = export_file(log_csv, final_caseid_list, s_vci, ci, ac, n, selected_method, log_name)

    #show statistic and saves the image file of it automatically
    showstats(stat1, stat2, selected_method, log_name)
    
    return sampled_csv, exported_file

unnecessaries = ["time:timestamp",
                 "org:resource"]



def run(request):
    Slog_attributes = {}
    event_logs_path = os.path.join(settings.MEDIA_ROOT, "event_logs")
    file_dir = os.path.join(event_logs_path, settings.EVENT_LOG_NAME)
    n_event_logs_path = os.path.join(settings.MEDIA_ROOT, "none_event_logs")

    # log = importer.apply(event_log)




    if request.method == 'POST':
        selected_method = request.POST.get('selected_method', False)
        case_id = request.POST.get('case_id', False)
        activity_name = request.POST.get('activity_name', False)
        timestamp_time = request.POST.get('timestamp_time', False)
        log_name = request.POST.get('log_name', False)
        



        param_keys = {constants.PARAMETER_CONSTANT_CASEID_KEY: case_id,
                    constants.PARAMETER_CONSTANT_ACTIVITY_KEY: activity_name,
                    constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: timestamp_time
                    }

        event_log = convert_eventfile_to_log(file_dir, timestamp_time, param_keys, activity_name, case_id)




        print("selected_method is " + selected_method)
        print("Selected Case id = " + case_id)
        print("Selected Activity name = " + activity_name)


        if selected_method == "Division":

            x = 0

        elif selected_method == "Unique-Selection":
            x = 1

        elif selected_method == "Logarithmic-distribution":
            x = 2



        ci = case_id
        ac = activity_name
        # case:concept:name

        # this method will preprocess the given log_csv
        # it means it will organize the dataframe such that

        samppled_file, exported_file = computation_sampling(event_log, ci, ac, x, selected_method, log_name)
        print("+++++++++++++++++++++++++++++exported_file")
        print(exported_file)



        file_dir_log = os.path.join(event_logs_path, exported_file)
       
       


        #log = xes_importer_factory.apply(file_dir)
        
        log = convert_exported_to_log(file_dir_log)


       


        exported_file = json.dumps(exported_file)

        Slog_attributes['exported_file'] = exported_file

        # Get all the column names and respective values
        #log_attributes['ColumnNamesValues'] = convert_eventlog_to_json(log)

        #eventlogs = [f for f in listdir(event_logs_path) if isfile(join(event_logs_path, f))]
        #xes_log = log

        #Get all the log statistics


        Sno_cases, Sno_events, Sno_variants, Stotal_case_duration, Savg_case_duration, Smedian_case_duration = get_Log_Statistics(log, timestamp_time)
        Slog_attributes['Sno_cases'] = Sno_cases
        Slog_attributes['Sno_events'] = Sno_events
        Slog_attributes['Sno_variants'] = Sno_variants
        Slog_attributes['Stotal_case_duration'] = Stotal_case_duration
        Slog_attributes['Savg_case_duration'] = Savg_case_duration
        Slog_attributes['Smedian_case_duration'] = Smedian_case_duration


        new_log_attributes = {'Slog_attributes': Slog_attributes}


        return HttpResponse(json.dumps(new_log_attributes), content_type="application/json")

    else:
        return render( 'upload.html', locals()) 





def convert_eventfile_to_log(file_path, timestamp_time_col, param_keys_h, activity_name1, case_id1):

    file_name, file_extension = os.path.splitext(file_path)

    if file_extension == '.csv':

        log = pd.read_csv(file_path, sep=',')
        log = dataframe_utils.convert_timestamp_columns_in_df(log) 
        log = log.sort_values(timestamp_time_col)
        log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME, parameters=param_keys_h)

    else:

        variant = xes_importer.Variants.ITERPARSE
        parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
        #log = xes_importer.apply(file_path, parameters=param_keys_h)

        log = xes_importer_factory.apply(file_path, parameters=param_keys_h)
        log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME , parameters=param_keys_h)
    return log


def convert_exported_to_log(file_path):

    file_name, file_extension = os.path.splitext(file_path)

    if file_extension == '.csv':

        log = pd.read_csv(file_path, sep=',')

        log = dataframe_utils.convert_timestamp_columns_in_df(log)  
        log = log_converter.apply(log) 

    else:

        log = xes_importer_factory.apply(file_path)
     
    return log





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


def get_Log_Statistics(log, timestamp_time):

    no_cases = len(log)

    no_events = sum([len(trace) for trace in log])

   

    
    variants = case_stat.variants_get.get_variants(log)
    no_variants = len(variants)

    all_case_durations = case_stat.get_all_casedurations(log, parameters={
    case_stat.Parameters.TIMESTAMP_KEY: timestamp_time})

    total_case_duration = (sum(all_case_durations))

    if no_cases <= 0:
        avg_case_duration = 0
    else:
        avg_case_duration = total_case_duration/no_cases

    median_case_duration = (case_stat.get_median_caseduration(log, parameters={
        case_stat.Parameters.TIMESTAMP_KEY: timestamp_time
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
