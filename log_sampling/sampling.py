import os
import pandas as pd
import pm4py
from pm4py.algo.filtering.log.cases import case_filter
from pm4py.algo.filtering.pandas.timestamp import timestamp_filter
from pm4py.algo.filtering.log.start_activities import start_activities_filter
from pm4py.algo.filtering.log.end_activities import end_activities_filter
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util import constants





#Import event log

log_csv = pd.read_csv('../eventlogs/ItalianHelpdeskFinal.csv', sep=',')
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
param_keys={constants.PARAMETER_CONSTANT_CASEID_KEY: 'Case ID', 
    constants.PARAMETER_CONSTANT_ACTIVITY_KEY: 'Activity', 
            constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "Complete Timestamp"
           }
log_csv




#Convert to dataframe

event_log = log_converter.apply(log_csv,parameters=param_keys)


#Insights to Variants

from pm4py.statistics.traces.pandas import case_statistics
variants = case_statistics.get_variants_df(log_csv,
                                          parameters={constants.PARAMETER_CONSTANT_CASEID_KEY: "Case ID",
                                                      constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "Activity"})
variants.describe(include='all')

stats = variants.describe(include='all')
print (stats)