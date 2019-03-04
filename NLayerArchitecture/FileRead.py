def readFromFile(path):
    toread = open(path, "r")
    ret = toread.readlines()
    toread.close()
    return ret