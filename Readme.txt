
Instructions for using the Time Series Analysis Code:


1. Download the dataset file "TS_Final_Dataset_Six_Hourly.csv".

2. Open the Python file in PyCharm.

3. Locate line 64 in the code, and set the file path to the location where the downloaded file is saved using the pd.read_csv function. 
   PLEASE Be sure to include the arguments parse_dates=['DateTime'] and index_col='DateTime' in the pd.read_csv function, as shown in the example below:

for example:

import pandas as pd

df = pd.read_csv(r'C:\Users\your_username\Downloads\TS_Final_Dataset_Six_Hourly.csv', parse_dates=['DateTime'], index_col='DateTime')

4. Run the code. The results of the time series analysis will be presented. It will take some time to run the code. Please be patient!




I hope these instructions are helpful. Please let me know if you have any further questions or issues.

Thank you!

Sincerely,
HaeLee 