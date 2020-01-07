#Library Imports
import csv
import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext
import pandas as pd
import numpy as np
import re, math
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
import os
import winsound
import datetime
from datetime import date

#File generation constraints
eventFilter = "all"     # "online", "offline", "maintenance", "frequency division", "outage", or "all"
assetFilter = "all"
SMPsBack = 3*60         # 3 hours backwards
SMPsForward = 3*60      # 3 hours forwards
measToFile = 0          # write the event measurements (min, max SMP, etc) to each individual file before SMP list (can be turned off if files are to be used for separate purposes ie. files will only include header and SMPs)

# CSV import filenames
SMP_CSV = "D:/1. Programming/Pycharm/AESO GRU Predictor/Test Data/RandomSMPTestSmall.csv"        #Raw Data/SMPOct2016Oct2019.csv"
Event_CSV = "D:/1. Programming/Pycharm/AESO GRU Predictor/Raw Data/EventSmall0.csv"    #EventOct2016Oct2019.csv"

#Constants
stop_words = set(stopwords.words("english"))
include = ["out", "on", "off", "line", "in"]
eventTypeDF = pd.DataFrame({'offline':['offline', 'off', 'out', 'ofline', 'Off', 'Offline'], 'online':['online', 'On', 'on', 'in', 'In', 'Online'], 'maintenance':['maintenance', 'maintain', 'maint', 'maintance', 'Maintenance', 'Maintain'], 'frequency deviation':['frequency', 'deviation', 'Frequency', 'freq', 'Deviation', 'Freq'], 'outage':['outage', 'outtage', 'out tage', 'out age', 'Outage', 'Outtage']})
eventTypeList = [['offline', 'off', 'out', 'ofline', 'Off', 'Offline'], ['online', 'On', 'on', 'in', 'In', 'Online'], ['maintenance', 'maintain', 'maint', 'maintance', 'Maintenance', 'Maintain'], ['frequency deviation', 'deviation', 'Frequency', 'freq', 'Deviation', 'frequency'], ['outage', 'outtage', 'out tage', 'out age', 'Outage', 'Outtage']]
WORD = re.compile(r'\w+')
stemmer = PorterStemmer()

for w in include:
    if w in stop_words:
        stop_words.remove(w)

def main():
    print("AESO Event Data Prepper\n")
    # reads and parses data from SMP and grid event CSV files
    eventsCSV = readCSV(Event_CSV)
    smpCSV = readCSV(SMP_CSV)
    SMPList = parseSMP(smpCSV)
    eventList = parseEvent(eventsCSV)

    # prints all events parsed from files
    print("Printing events in eventList")
    for e in eventList:
        e.printEvent()

    # filters events into new list based on filters from the file generation constraints
    new_eventList = filterEvents(eventList)
    loadSMPList(new_eventList, SMPList)

    # prints the new list of events
    print("\nPrinting events in new_eventList")
    for e in new_eventList:
        e.printEvent()

    # writes the filtered event list to CSV files containing the SMPs from the constraints listed in the file generation constraints
    writeCSVFiles(new_eventList)

    # writes the filtered event data to a CSV file containing the event parameters
    writeEventData(new_eventList)

    # beeps to signal completion
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

    print("run complete")

# Event is a class that defines events produced from AESO AIES reports with date, time, and message
class Event:

    # constructor for Event
    def __init__(self, dateTime, message):
        self.dateTime = dateTime
        self.message = message
        self.eventType = self.findEventType()
        self.asset = self.findAsset()
        self.SMP = []
        self.AB_BC = []
        self.AB_MT = []
        self.AB_SK = []
        self.AB_Int = []
        self.eventSMP = 0
        self.minMaxDifTotal = 0
        self.maxSMPAfterEvent = 0
        self.minSMPAfterEvent = 0
        self.maxGrowth = 0
        self.minReduction = 0
        self.minTimeFromEvent = 0
        self.maxTimeFromEvent = 0
        self.priceEvent = "false"
        self.avgBeforeEvent = 0
        self.avgAfterEvent = 0

    # printEvent prints the Event info in a formatted string
    def printEvent(self):
        print("\nDate and time is: ", self.dateTime)
        print("Message is: ", self.message)
        print("Event is: ", self.eventType)
        print("Asset is: ", self.asset)
        # print("Printing SMP list")
        # # print(self.SMP)
        # for p in self.SMP:
        #     print(p[0], "\t", p[1])

    # appendMessage appends a string to the end of the message of an Event
    def appendMessage(self, appended):
        temp = self.message
        self.message = temp + appended

    # find event type from the message
    def findEventType(self):
        message = self.filterMessage(1)

        # for use with list type
        # print()
        # print("Message is: ", message)

        eventTypeLength = len(eventTypeList[0])
        tag = False
        for w in message:
            count = 0
            for x in eventTypeList:
                for y in x:
                    if y == w and not tag:
                        num = int((count / eventTypeLength))
                        num = int(num)
                        event = eventTypeList[num][0]
                        # print("Num is: ", num, "Count is: ", count, "Event is: ", event, "y is: ", y, "w is: ", w)
                        tag = True
                    count += 1

        if tag == False:
            event = "other"

        # print("event is: ", event)

        return event

    def findAsset(self):
        assetBool = [False, False]
        noun = ["NNP", "NN", "NNS", "NNPS"]
        dig = "CD"
        message = self.tagMessage()
        # print("Tagged message is: ", message)
        for w in message:
            if w[1] in noun and assetBool[0] == False:
                name = w[0]
                # print("Asset name is: ", name)
                assetBool[0] = True
            if w[1] == dig and assetBool[1] == False:
                num = w[0]
                # print("Asset num is: ", num)
                assetBool[1] = True

        if False in assetBool:
            asset = "other"
        else:
            asset = name + " " + num

        return asset

    def tagMessage(self, print_tag = 0):
        text = webtext.raw('overheard.txt')
        temp = PunktSentenceTokenizer(text)
        message = temp.tokenize(self.message)

        for w in message:
            words = nltk.word_tokenize(w)
            tagged = nltk.pos_tag(words)

        if print_tag:
            print("Message is: ", self.message)
            print("Tagged message is: ", tagged)

        return tagged

    def filterMessage(self, filter_stop_words = 0, stem = 0, printFilter = 0):
        ps = PorterStemmer()

        message_tokenized = word_tokenize(self.message)
        message = message_tokenized

        message_filtered = []
        message_stemmed = []

        if filter_stop_words:
            for w in message:
                if w not in stop_words:
                    message_filtered.append(w)
            message = message_filtered
        if stem:
            for w in message:
                message_stemmed = ps.stem(w)
            message = message_stemmed

        if printFilter:
            print("Message is: ", self.message)
            print("Tokenized message is: ", message_tokenized)
            if filter_stop_words:
                print("Filtered message is: ", message_filtered)
            if stem:
                print("Stemmed message is: ", message_stemmed)

        return message

    def getSMPList(self, SMPList):
        eventSMPList = []

        count = 0
        for d in SMPList:
            # print("SMPList date: ", d[0], "Self date: ", self.dateTime)
            # print("SMPList hour: ", d[0].hour, "Self hour: ", self.dateTime.hour)
            if d[0].year == self.dateTime.year and d[0].month == self.dateTime.month and d[0].day == self.dateTime.day and d[0].hour == self.dateTime.hour and d[0].minute == self.dateTime.minute:
                print("\nMATCH FOUND")
                print("Date match found is: ", self.dateTime)
                print("Count is: ", count)
                self.eventSMP = float(d[1])
                break
            count += 1

        if count < SMPsBack:
            rangeLow = 0
        else:
            rangeLow = count - SMPsBack
        if len(SMPList) - count < SMPsForward:
            rangeHigh = len(SMPList)
        else:
            rangeHigh = count + SMPsForward

        for p in range(rangeLow, rangeHigh):
            try:
                eventSMPList.append(SMPList[p])
            except:
                "SMP list out of range"

        self.SMP = eventSMPList

    def writeToCSV(self, dir = ""):
        if dir != "":
            dir = dir + "/"
        filename = dir + str(self.dateTime.year) + "-" + str(self.dateTime.month) + "-" + str(self.dateTime.day) + "_" + self.asset + "_" + self.eventType + ".csv"
        with open(filename, mode='w', newline='') as event_file:
            event_writer = csv.writer(event_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if (measToFile):
                event_writer.writerow(["Event Name", filename])
                event_writer.writerow(["Event Date & Time", self.dateTime])
                event_writer.writerow(["Event SMP", self.eventSMP])
                event_writer.writerow(["Minimum SMP after event", "{} ({})".format(self.minSMPAfterEvent[1], self.minSMPAfterEvent[0])])
                event_writer.writerow(["Maximum SMP after event", "{} ({})".format(self.maxSMPAfterEvent[1], self.maxSMPAfterEvent[0])])
                dif = self.minSMPAfterEvent[0] - self.dateTime
                event_writer.writerow(["Largest SMP reduction after event", "{}% ({} minutes after event)".format(self.minReduction, self.minTimeFromEvent)])
                dif = self.maxSMPAfterEvent[0] - self.dateTime
                event_writer.writerow(["Largest SMP growth after event", "{}% ({} minutes after event)".format(self.maxGrowth, self.maxTimeFromEvent)])
                event_writer.writerow(["Percent growth between smallest and largest SMP in file", "{}%".format(self.minMaxDifTotal)])
                event_writer.writerow([])
                event_writer.writerow(["Date Time", "System Marginal Price (SMP)"])
            else:
                event_writer.writerow(["Date Time", "System Marginal Price (SMP)", "Event occurred at {}".format(self.dateTime)])
            for p in self.SMP:
                event_writer.writerow([p[0], p[1]])

    def SMPMinMaxDif(self):
        minTotal = float(self.SMP[0][1])
        maxTotal = minTotal
        maxAfterEvent = [self.dateTime, self.eventSMP]
        minAfterEvent = [self.dateTime, self.eventSMP]
        avg = 0
        avgCount = 0

        for smp in self.SMP:
            smpPrice = float(smp[1])
            if smpPrice < minTotal:
                minTotal = smpPrice
            if smpPrice > maxTotal:
                maxTotal = smpPrice

            if smp[0] == self.dateTime:
                self.avgBeforeEvent = avg / avgCount    # avg SMP before event
                avg = 0
                avgCount = 0

            if smp[0] > self.dateTime:
                if smpPrice < float(minAfterEvent[1]):
                    minAfterEvent = smp
                if smpPrice > float(maxAfterEvent[1]):
                    maxAfterEvent = smp

            avg = avg + smpPrice
            avgCount += 1

        self.avgAfterEvent = avg / avgCount                                           # avg SMP after event
        self.minMaxDifTotal = ((maxTotal/minTotal)-1)*100                             # percent growth
        self.minSMPAfterEvent = minAfterEvent
        self.maxSMPAfterEvent = maxAfterEvent
        self.minReduction = (1-(self.minSMPAfterEvent[1] / self.eventSMP)) * 100      # percent reduction
        self.maxGrowth = ((self.maxSMPAfterEvent[1] / self.eventSMP)-1) * 100         # percent growth
        dif = self.minSMPAfterEvent[0] - self.dateTime
        self.maxTimeFromEvent = (dif.days * 24 * 60 * 60 + dif.seconds) / 60
        dif = self.maxSMPAfterEvent[0] - self.dateTime
        self.minTimeFromEvent = (dif.days * 24 * 60 * 60 + dif.seconds) / 60
        self.priceEvent = "true" if self.maxSMPAfterEvent[1] > 200 else "false"


# source: https://stackoverflow.com/questions/38365389/compare-similarity-between-names
def get_cosine(vec1, vec2):
    # print vec1, vec2
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

# source: https://stackoverflow.com/questions/38365389/compare-similarity-between-names
def text_to_vector(text):
    words = WORD.findall(text)
    a = []
    for i in words:
        for ss in wn.synsets(i):
            a.extend(ss.lemma_names())
    for i in words:
        if i not in a:
            a.append(i)
    a = set(a)
    w = [stemmer.stem(i) for i in a if i not in stop_words]
    return Counter(w)

# source: https://stackoverflow.com/questions/38365389/compare-similarity-between-names
def get_similarity(a, b):
    a = text_to_vector(a.strip().lower())
    b = text_to_vector(b.strip().lower())

    return get_cosine(a, b)

# source: https://stackoverflow.com/questions/38365389/compare-similarity-between-names
def get_char_wise_similarity(a, b):
    a = text_to_vector(a.strip().lower())
    b = text_to_vector(b.strip().lower())
    s = []

    for i in a:
        for j in b:
            s.append(get_similarity(str(i), str(j)))
    try:
        return sum(s) / float(len(s))
    except:  # len(s) == 0
        return 0

# readCSV reads a csv input from the AIES reports and separates it into a list of strings by row excluding some initial formatted lines
def readCSV(csvFile):
    with open(csvFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        row_count = 0
        csvList = []
        x = 0

        for row in csv_reader:

            if row_count > 5:
                csvList.append(f'{",".join(row)}')
                x += 1

            row_count += 1
    return csvList

# parseEvent parses the csvList into Events by separating date, time, message and checks for messages that were formatted improperly with alt+enter formatting in the cell
def parseEvent(csvList):
    events = []
    eventCount = 0

    for x in range(0, len(csvList)):
        dateSplit = csvList[x].split(" ", 1)  #splits "mm/dd/yyyy" and "hh:mm,message"
        date = dateSplit[0].split("/")          #splits "mm", "dd", and "yyyy"

        if len(date) == 3:
            timeSplit = dateSplit[1].split(",", 1)  # splits "hh:mm" and "message"
            message = timeSplit[1]  # message = "message"
            time = timeSplit[0].split(":")  # splits "hh" and "mm"

            dateTime = datetime.datetime(int(date[2]), int(date[0]), int(date[1]), int(time[0]), int(time[1]))
            events.append(Event(dateTime, message))
            eventCount += 1

        if len(date) != 3 and eventCount > 0:
            events[eventCount-1].appendMessage(", " + csvList[x])    #csvList[x]

    return events

# parseSMP parses the smpCSV into a list of dates / times and corresponding SMP
def parseSMP(SMP_CSV):
    SMPList = []

    for p in range(0, len(SMP_CSV)):
        dateSplit = SMP_CSV[p].split(" ", 1)  # splits "mm/dd/yyyy" and "hh:mm,price"
        date = dateSplit[0].split("/")          #splits "mm", "dd", and "yyyy"
        timeSplit = dateSplit[1].split(",", 1)  # splits "hh:mm" and "price"
        SMP = float(timeSplit[1])
        time = timeSplit[0].split(":")  # splits "hh" and "mm"

        dateTime = datetime.datetime(int(date[2]), int(date[0]), int(date[1]), int(time[0]), int(time[1]))
        list = [dateTime, SMP]
        SMPList.append(list)

    return SMPList

# printEvents prints all of the events in a eventList
def printEvents(eventList):
    for x in range(0, len(eventList)):
        eventList[x].printEvent()

def filterEvents(eventList, eventFilter = "all", assetFilter = "all"):
    new_eventList = []
    for e in eventList:
        if (e.eventType == eventFilter or eventFilter == "all") and (e.asset == assetFilter or assetFilter == "all"):
            new_eventList.append(e)

    return new_eventList

def loadSMPList(eventList, SMPList):
    count = 0
    for e in eventList:
        print("loadSMPList: ", count)
        e.getSMPList(SMPList)
        e.SMPMinMaxDif()
        count += 1

def writeCSVFiles(eventList):
    path = "/1. Programming/Pycharm/AESO GRU Predictor/Prepped Data"
    for e in eventList:
        e.writeToCSV(path)

def writeEventData(eventList):
    path = "/1. Programming/Pycharm/AESO GRU Predictor/Event Data/"

    today = date.today()
    filename = path + "EventData_Asset[" + assetFilter + "]_Event[" + eventFilter + "]_" + str(today.day) + str(today.month) + str(today.year) + ".csv"
    with open(filename, mode='w', newline='') as event_file:
        event_writer = csv.writer(event_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        event_writer.writerow(["Date", "Event Type", "Asset", "Message", "SMP", "Price Spike (>$200)", "Max SMP After Event", "SMP Growth from Event", "Max SMP Date", "Time from Event (minutes)", "Min SMP After Event", "SMP Reduction from Event", "Min SMP Date", "Time from Event (minutes)", "Avg SMP before Event", "Avg SMP After Event"])
        for e in eventList:
            event_writer.writerow(
                [e.dateTime, e.eventType, e.asset, e.message, e.eventSMP, e.priceEvent, e.maxSMPAfterEvent[1], e.maxGrowth, e.maxSMPAfterEvent[0],
                 e.maxTimeFromEvent, e.minSMPAfterEvent[1], e.minReduction, e.minSMPAfterEvent[0],
                 e.minTimeFromEvent, e.avgBeforeEvent, e.avgAfterEvent])

# defining main
if __name__ == "__main__":
    main()