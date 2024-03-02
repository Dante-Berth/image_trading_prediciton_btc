import os.path
def isReadableFile(file_path, file_name):
    full_path = file_path + "/" + file_name
    try:
        if not os.path.exists(file_path):
            print("File path is invalid.")
            return False
        elif not os.path.isfile(full_path):
            print("File does not exist.")
            return False
        elif not os.access(full_path, os.R_OK):
            print("File cannot be read.")
            return False
        else:
            print("File can be read.")
            return True
    except IOError as ex:
        print ("I/O error({0}): {1}".format(ex.errno, ex.strerror))
    return False