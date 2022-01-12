from django.test import TestCase
import csv
import numpy as np

# Create your tests here.


def character():
    with open('/ws2122-lspm/eventlogs/ItalianHelpdeskFinal.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        list_of_column_names = []
        for row in csv_reader:
            list_of_column_names.append(row)
            break
    vars = []
    array = np.array(list_of_column_names)
    for column in array:

        for j in column:
            print(j)
            vars.append(j)
        
        return vars


x = character()
print(x)
