class Logger:
    def __init__(self, filename):
        self.file = open(filename, 'w')

    def write(self, listOfLines):
        for element in listOfLines:
            strToWrite = ''
            for line in element:
                strToWrite += str(line) + ', '

            self.file.write(strToWrite[:-2]+ '\n')

    def close(self):
        self.file.close()
