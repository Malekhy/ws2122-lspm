import csv
from tempfile import NamedTemporaryFile
import shutil
import numpy as np


""" import csv
with open('/ws2122-lspm/upload_eventlog/data.csv', newline='') as f:
    reader = csv.reader(f)
    row1 = next(reader)  # gets the first line
    for row in reader:
        print(row)       # prints rows 2 and onward
 """


with open('/ws2122-lspm/upload_eventlog/data.csv') as csv_file:
    # creating an object of csv reader
    # with the delimiter as ,
    csv_reader = csv.reader(csv_file, delimiter=',')

    # list to store the names of columns
    list_of_column_names = []

    # loop to iterate through the rows of csv
    for row in csv_reader:

        # adding the first row
        list_of_column_names.append(row)

        # breaking the loop after the
        # first iteration itself
        break

# printing the result
# print("List of column names : ",
    
vars = []
array = np.array(list_of_column_names)
for column in array:
    # hells[column]=list_of_column_names[0][column]
    for j in column:
        print(j)
        vars.append(j)
    print()

print(vars[3])
