# AESO Market Price Data Prep
#
# Developed by Austyn Nagribianko
# anagribianko@gmail.com

#Library Imports
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext
from nltk.corpus import stopwords
from nltk.stem.porter import *
import winsound
import datetime
from datetime import date
import time

#File generation constraints
eventFilter = "all"     # "online", "offline", "maintenance", "frequency division", "outage", or "all"
assetFilter = "all"
PPsBack = 3
PPsForward = 3
SMPsBack = PPsBack*60         # 3 hours backwards
SMPsForward = PPsForward*60      # 3 hours forwards
measToFile = 0          # write the event measurements (min, max SMP, etc) to each individual file before SMP list (can be turned off if files are to be used for separate purposes ie. files will only include header and SMPs)

# CSV import filenames
workingDir = "D:/1. Programming/Pycharm/AESO Market Price Preparation/"
SMP_CSV = workingDir + "Raw Data/SMPJan2017Aug2019.csv"
PP_CSV = workingDir + "Raw Data/PoolPriceJan2017Jan2020.csv"
Event_CSV = workingDir + "Raw Data/EventJan2017Aug2019.csv"                     # EventJan2017Aug2019.csv
AB_BC_Int_CSV = workingDir + "Raw Data/AB_BC_IntOct2016Oct2019.csv"
AB_MT_Int_CSV = workingDir + "Raw Data/AB_MT_IntOct2016Oct2019.csv"
AB_SK_Int_CSV = workingDir + "Raw Data/AB_SK_IntOct2016Oct2019.csv"
AB_Int_Total_CSV = workingDir + "Raw Data/AB_Int_TotalOct2016Oct2019.csv"
AB_Demand_CSV = workingDir + "Raw Data/AB_DemandOct2016Oct2019.csv"
AB_Net_Gen_CSV = workingDir + "Raw Data/AB_Net_GenJan2017Jan2020.csv"
eventDataPath = workingDir + "Event Data/"
CSVFilePath = workingDir + "Prepped Data/"
# ADDITIONAL FILE: add the file CSV directory here

#Constants
stop_words = set(stopwords.words("english"))
include = ["out", "on", "off", "line", "in"]
eventTypeList = [['offline', 'off', 'out', 'ofline', 'Off', 'Offline'], ['online', 'On', 'on', 'in', 'In', 'Online'], ['maintenance', 'maintain', 'maint', 'maintance', 'Maintenance', 'Maintain'], ['frequency deviation', 'deviation', 'Frequency', 'freq', 'Deviation', 'frequency'], ['outage', 'outtage', 'out tage', 'out age', 'Outage', 'Outtage']]
assetList = [["Calgary Energy Centre", "Calgary Energy Center"], ["HR Milner", "H. R. Milner Generating Station", "H.R. MIlner", "Milner"], ["MATL"], ["Shepard Energy Centre", "Shepard"], ["WECC"], ["MacKay River", "MKRC"], ["AESO", "planned system maintenance", "planned maintenance"]]
WORD = re.compile(r'\w+')
stemmer = PorterStemmer()

# removes key words from stopwords list
for w in include:
    if w in stop_words:
        stop_words.remove(w)

def main():
    # timer
    t0 = time.time()

    print("AESO Event Data Prepper\n")
    print("Event filter: {}\nAsset filter: {}\nSMPs before event: {}\nSMPs after event: {}\n".format(eventFilter, assetFilter, SMPsBack, SMPsForward))

    # reads and parses data from SMP and grid event CSV files
    print("Reading data files...")
    eventsCSV = readCSV(Event_CSV)
    smpCSV = readCSV(SMP_CSV)
    ppCSV = readCSV(PP_CSV)
    AB_BC_IntCSV = readCSV(AB_BC_Int_CSV)
    AB_MT_IntCSV = readCSV(AB_MT_Int_CSV)
    AB_SK_IntCSV = readCSV(AB_SK_Int_CSV)
    AB_Int_TotalCSV = readCSV(AB_Int_Total_CSV)
    AB_DemandCSV = readCSV(AB_Demand_CSV)
    AB_Net_GenCSV = readCSV(AB_Net_Gen_CSV)
    # ADDITIONAL FILE: mimic the above lines to read the CSV file

    # parse csv details
    print("Parsing data files...")
    print("Parsing file: {}".format(SMP_CSV))
    SMPList = parseFile(smpCSV)
    print("Parsing file: {}".format(PP_CSV))
    PPList = parseFile(ppCSV)
    print("Parsing file: {}".format(Event_CSV))
    eventList = parseEvent(eventsCSV)
    print("Parsing file: {}".format(AB_BC_Int_CSV))
    AB_BC_IntList = parseFile(AB_BC_IntCSV)
    print("Parsing file: {}".format(AB_MT_Int_CSV))
    AB_MT_IntList = parseFile(AB_MT_IntCSV)
    print("Parsing file: {}".format(AB_SK_Int_CSV))
    AB_SK_IntList = parseFile(AB_SK_IntCSV)
    print("Parsing file: {}".format(AB_Int_Total_CSV))
    AB_Int_TotalList = parseFile(AB_Int_TotalCSV)
    print("Parsing file: {}".format(AB_Demand_CSV))
    AB_DemandList = parseFile(AB_DemandCSV)
    print("Parsing file: {}".format(AB_Net_Gen_CSV))
    AB_Net_GenList = parseFile(AB_Net_GenCSV)
    # ADDITIONAL FILE: mimic the above lines to parse the list. Make sure to use parseFile and not parseEvent.
    # parseFile works for all files with date and unit columns (ie date and SMP) exported from NRGStream.
    # Other file formats may need to be formatted before they can be imported

    # checking to ensure that the SMP range exceeds the event list range
    # does not check ranges of intertie files
    if SMPList[0][0] > eventList[0].dateTime:
        print("SMP date range does not reach far enough. SMP list must exceed the events list")
        exit()
    elif SMPList[len(SMPList)-1][0] < eventList[len(eventList)-1].dateTime:
        print("SMP date range is not recent enough. SMP list must exceed the events list")
        exit()
    else:
        print("Files successfully read")

    # filters events into new list based on filters from the file generation constraints
    print("Filtering events list...")
    new_eventList = filterEvents(eventList)

    print("Loading event information...")
    loadLists(new_eventList, SMPList, PPList, AB_BC_IntList, AB_MT_IntList, AB_SK_IntList, AB_Int_TotalList, AB_DemandList, AB_Net_GenList)
    # ADDITIONAL FILE: add the list file to the end of the loadLists arguments (see loadLists function)

    # # prints the new list of events
    # print("\nPrinting events in new_eventList")
    # for e in new_eventList:
    #     e.printEvent()

    # writes the filtered event list to CSV files containing the SMPs from the constraints listed in the file generation constraints
    writeCSVFiles(new_eventList, CSVFilePath)

    # writes the filtered event data to a CSV file containing the event parameters
    writeEventData(new_eventList, eventDataPath)

    t1 = time.time()

    # beeps to signal completion
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

    print("Run complete")
    print("Time to run: {} seconds".format(t1-t0))

# Event is a class that defines events produced from AESO AIES reports with date, time, and message
class Event:

    # constructor for Event
    def __init__(self, dateTime, message):
        self.dateTime = dateTime
        self.message = message
        self.eventType = self.findEventType()
        self.asset = self.findAsset()
        self.SMP = []
        self.PP = []
        self.AB_BC_Int = 0
        self.AB_MT_Int = 0
        self.AB_SK_Int = 0
        self.AB_Int_Total = 0
        self.AB_Demand = 0
        self.AB_Net_Gen = 0
        self.eventSMP = 0
        self.eventPP = 0
        self.minMaxDifTotal = 0
        self.maxSMPAfterEvent = 0
        self.minSMPAfterEvent = 0
        self.maxGrowth = 0
        self.minReduction = 0
        self.minTimeFromEvent = 0
        self.maxTimeFromEvent = 0
        self.SMPPriceEvent = "NO"
        self.PPPriceEvent = "NO"
        self.avgBeforeEvent = 0
        self.avgAfterEvent = 0
        self.maxPP = 0

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

    # findEventType searches the event message to find the event type based on the key words in event
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
            for w in self.message.split():
                if assetBool[0] and assetBool[1]:
                    break
                for x in range(0, len(assetList)):
                    for y in assetList[x]:
                        if w == y:
                            asset = assetList[x][0]
                            assetBool = [True, True]
                            break
            asset = "other" if False in assetBool else asset
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

    def getPriceList(self, prices, numBack, numFor, type = "SMP", printResults = 0):
        priceList = []
        eventPrice = 0
        count = 0
        dt = self.dateTime if type == "SMP" else self.dateTime - datetime.timedelta(minutes = self.dateTime.minute)
        for d in prices:
            if d[0] == dt:              #d[0].year == self.dateTime.year and d[0].month == self.dateTime.month and d[0].day == self.dateTime.day and d[0].hour == self.dateTime.hour and d[0].minute == self.dateTime.minute:
                if printResults:
                    print("\nMATCH FOUND")
                    print("Date match found is: ", self.dateTime)
                    print("Count is: ", count)
                eventPrice = float(d[1])
                break
            count += 1

        if type == "PP":
            rangeLow = count
        elif count < numBack:
            rangeLow = 0
        else:
            rangeLow = count - numBack
        if len(prices) - count < numFor:
            rangeHigh = len(prices)
        else:
            rangeHigh = count + numFor

        for p in range(rangeLow, rangeHigh):
            try:
                priceList.append(prices[p])
            except:
                "Price list out of range"

        return priceList, eventPrice

    def loadDemand(self, AB_BC_IntList, AB_MT_IntList, AB_SK_IntList, AB_Int_TotalList, AB_DemandList, AB_Net_GenList):
        # ADDITIONAL FILE: add the list file to the end of the loadDemand arguments
        # depending on how you want to utilize the information you can either copy the below format for extracting information (copies value at the same hour as the event)
        # or alternatively you can develop your own method as well
        # the lists in this function fill the class variables therefore you may need to create another class variable to store the appropriate information
        # see writeEventData function
        for e in AB_BC_IntList:
            if e[0].year == self.dateTime.year and e[0].month == self.dateTime.month and e[0].day == self.dateTime.day and e[0].hour == self.dateTime.hour:
                self.AB_BC_Int = float(e[1])
        for e in AB_MT_IntList:
            if e[0].year == self.dateTime.year and e[0].month == self.dateTime.month and e[0].day == self.dateTime.day and e[0].hour == self.dateTime.hour:
                self.AB_MT_Int = float(e[1])
        for e in AB_SK_IntList:
            if e[0].year == self.dateTime.year and e[0].month == self.dateTime.month and e[0].day == self.dateTime.day and e[0].hour == self.dateTime.hour:
                self.AB_SK_Int = float(e[1])
        for e in AB_Int_TotalList:
            if e[0].year == self.dateTime.year and e[0].month == self.dateTime.month and e[0].day == self.dateTime.day and e[0].hour == self.dateTime.hour:
                self.AB_Int_Total = float(e[1])
        for e in AB_DemandList:
            if e[0].year == self.dateTime.year and e[0].month == self.dateTime.month and e[0].day == self.dateTime.day and e[0].hour == self.dateTime.hour:
                self.AB_Demand = float(e[1])
        for e in AB_Net_GenList:
            if e[0].year == self.dateTime.year and e[0].month == self.dateTime.month and e[0].day == self.dateTime.day and e[0].hour == self.dateTime.hour:
                self.AB_Net_Gen = float(e[1])

    def writeToCSV(self, dir = ""):
        if dir != "":
            dir = dir + "/"

        filename = dir + str(self.dateTime.year) + "-" + str(self.dateTime.month) + "-" + str(self.dateTime.day) + "_h" + str(self.dateTime.hour) + "m" + str(self.dateTime.minute) + "_" + self.asset + "_" + self.eventType + ".csv"

        with open(filename, mode='w', newline='') as event_file:
            event_writer = csv.writer(event_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # writes measurements to each csv file
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
                event_writer.writerow(["Date Time", "System Marginal Price (SMP)", "Event occurred at {}".format(self.dateTime), "Event Message: {}".format(self.message)])
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

        if minTotal != 0:
            self.minMaxDifTotal = ((maxTotal/minTotal)-1)*100                             # percent growth
        else:
            self.minMaxDifTotal = "INF"

        self.minSMPAfterEvent = minAfterEvent
        self.maxSMPAfterEvent = maxAfterEvent

        if self.eventSMP != 0:
            self.minReduction = (1-(self.minSMPAfterEvent[1] / self.eventSMP)) * 100     # percent reduction
            self.maxGrowth = ((self.maxSMPAfterEvent[1] / self.eventSMP)-1) * 100         # percent growth
        else:
            self.minReduction = 0
            self.maxGrowth = "INF"

        dif = self.maxSMPAfterEvent[0] - self.dateTime
        self.maxTimeFromEvent = (dif.days * 24 * 60 * 60 + dif.seconds) / 60
        dif = self.minSMPAfterEvent[0] - self.dateTime
        self.minTimeFromEvent = (dif.days * 24 * 60 * 60 + dif.seconds) / 60
        self.SMPPriceEvent = "YES" if self.maxSMPAfterEvent[1] > 200 else "NO"

    def PoolPriceEvent(self):
        temp = 0
        for e in self.PP:
            if e[1] > temp:
                temp = e[1]
        self.maxPP = temp
        self.PPPriceEvent = "YES" if self.maxPP > 200 else "NO"

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
def parseEvent(eventCSV):
    eventList = []
    eventCount = 0

    for x in range(0, len(eventCSV)):
        dateSplit = eventCSV[x].split(" ", 1)  #splits "mm/dd/yyyy" and "hh:mm,message"
        date = dateSplit[0].split("/")          #splits "mm", "dd", and "yyyy"

        if len(date) == 3:
            timeSplit = dateSplit[1].split(",", 1)  # splits "hh:mm" and "message"
            message = timeSplit[1]  # message = "message"
            time = timeSplit[0].split(":")  # splits "hh" and "mm"

            dateTime = datetime.datetime(int(date[2]), int(date[0]), int(date[1]), int(time[0]), int(time[1]))
            if x > 0 and dateTime > eventList[x-1].dateTime:
                eventList.append(Event(dateTime, message))
            else:
                eventList.insert(0, Event(dateTime, message))
            eventCount += 1

        if len(date) != 3 and eventCount > 0:
            eventList[eventCount-1].appendMessage(", " + eventCSV[x])    #eventCSV[x]

    return eventList

# parseSMP parses the smpCSV into a list of dates / times and corresponding SMP
def parseFile(fileCSV):
    itemList = []

    for p in range(0, len(fileCSV)):
        dateSplit = fileCSV[p].split(" ", 1)  # splits "mm/dd/yyyy" and "hh:mm,item"
        date = dateSplit[0].split("/")          #splits "mm", "dd", and "yyyy"
        timeSplit = dateSplit[1].split(",", 1)  # splits "hh:mm" and "price"
        item = float(timeSplit[1])
        time = timeSplit[0].split(":")  # splits "hh" and "mm"

        dateTime = datetime.datetime(int(date[2]), int(date[0]), int(date[1]), int(time[0]), int(time[1]))
        list = [dateTime, item]
        if p > 0 and list[0] > itemList[p-1][0]:
            itemList.append(list)
        else:
            itemList.insert(0, list)

    return itemList

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

def loadLists(eventList, SMPList, PPList, AB_BC_IntList, AB_MT_IntList, AB_SK_IntList, AB_Int_TotalList, AB_DemandList, AB_Net_GenList):
    # ADDITIONAL FILE: add the list file to the end of the loadLists arguments
    count = 0
    for e in eventList:
        # print("loadSMPList: ", count)
        e.SMP, e.eventSMP = e.getPriceList(SMPList, SMPsBack, SMPsForward)
        e.PP, e.eventPP = e.getPriceList(PPList, PPsBack, PPsForward, "PP")
        # print("Printing the pool price: {}".format(e.PP))
        e.loadDemand(AB_BC_IntList, AB_MT_IntList, AB_SK_IntList, AB_Int_TotalList, AB_DemandList, AB_Net_GenList)
        # ADDITIONAL FILE: add the list file to the end of the loadDemand arguments (see loadDemand function)
        e.SMPMinMaxDif()
        e.PoolPriceEvent()
        count += 1

def writeCSVFiles(eventList, path):
    for e in eventList:
        e.writeToCSV(path)

def writeEventData(eventList, path):
    today = date.today()
    filename = path + "EventData_Asset[" + assetFilter + "]_Event[" + eventFilter + "]_" + str(today.day) + str(today.month) + str(today.year) + ".csv"
    with open(filename, mode='w', newline='') as event_file:
        event_writer = csv.writer(event_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        event_writer.writerow(["Date", "Event Type", "Asset", "Message", "SMP Spike After Event (>$200)", "Pool Price Spike After Event (>$200)", "Event SMP ($/MW)", "Event Pool Price ($/MW)", "Max SMP After Event", "Max Pool Price After Event", "Max SMP Growth from Event", "Max SMP Date", "Max SMP Minutes from Event", "Min SMP After Event", "Min SMP Reduction from Event", "Min SMP Date", "Min SMP Minutes from Event", "Avg SMP before Event", "Avg SMP After Event", "Net Generation (MW)", "Total AB Demand (MW)", "AB Total Intertie Trade (MW)", "AB - BC Intertie (MW)", "AB - MT Intertie (MW)", "AB - SK Intertie (MW)"])
        # ADDITIONAL FILE: add new column header above
        for e in eventList:
            event_writer.writerow(
                [e.dateTime, e.eventType, e.asset, e.message, e.SMPPriceEvent, e.PPPriceEvent, e.eventSMP, e.eventPP, e.maxSMPAfterEvent[1], e.maxPP, e.maxGrowth, e.maxSMPAfterEvent[0],
                 e.maxTimeFromEvent, e.minSMPAfterEvent[1], e.minReduction, e.minSMPAfterEvent[0],
                 e.minTimeFromEvent, e.avgBeforeEvent, e.avgAfterEvent, e.AB_Net_Gen, e.AB_Demand, e.AB_Int_Total, e.AB_BC_Int, e.AB_MT_Int, e.AB_SK_Int])
            # ADDITIONAL FILE: add the column data in the abvoe line


# defining main
if __name__ == "__main__":
    main()