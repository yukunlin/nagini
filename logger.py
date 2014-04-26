class Logger:
    def __init__(self, filename):
        self.file = open(filename, 'w')

    def write(self, listOfStuff):
        strToWrite = ''
        for element in listOfStuff:
            strToWrite += str(element) + ', '

        self.file.write(strToWrite[:-2]+ '\n')

    def close(self):
        self.file.close()
