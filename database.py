import csv
import cv2
def loadDB(filename, images, labels):
    data = []
    with open(filename, "r") as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            data.append(row)
    csvFile.close()

    # Read images and labels
    for d in data:
        path, label = d[0].split(' ; ')
        # IMPORTANT TO CONVERT IMAGE TO GRAY COLOR
        img = cv2.imread(path)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(int(label))