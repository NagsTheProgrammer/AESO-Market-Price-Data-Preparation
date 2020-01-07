import csv
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from random import seed
from random import random
from math import log

# variables
startTime = time(0, 0)
dateStart = datetime.combine(date(2019, 8, 10), startTime)   # october 1, 2016
dateTemp = dateStart
outputFilename = "RandomSMPTestSmall.csv"
numDays = 34

dateList = [dateStart + timedelta(minutes = x) for x in range(0, numDays*24*60)]
# for x in range(len(dateList)):
#     print(dateList[x])

with open("D:/1. Programming/Pycharm/AESO GRU Predictor/Test Data/"+outputFilename, mode='w', newline='') as SMP_file:
    SMPwriter = csv.writer(SMP_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    SMPwriter.writerow(["NRGSTREAM Data Extract for stream : AB - Real time SMP"])
    SMPwriter.writerow(["Generated at : 12/24/2019 12:00 with : 721 records"])
    SMPwriter.writerow(["Date criteria from : Dec 24 2019 to : Dec 25 2019"])
    SMPwriter.writerow(["Data Interval : 1 minutes"])
    SMPwriter.writerow([""])
    SMPwriter.writerow(["Date/Time", "Price", "Volume"])

    for x in dateList:
        # seed(1)
        chance = random()
        if chance > 0.96:
            mult = 20
        elif chance > 0.9:
            mult = 10
        elif chance > 0.5:
            mult = 6
        else:
            mult = 5

        temp = random()
        price = (log(temp*1000+1, 2) + temp - temp % 0.01)*mult
        print(price)
        SMPwriter.writerow(["{}/{}/{} {}:{}".format(x.month, x.day, x.year, x.hour, x.minute), price])