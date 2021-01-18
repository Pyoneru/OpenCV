import database as db
import cv2
import numpy as np

# DB Faces: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
images = []
labels = []
db.loadDB('Database.csv', images, labels)

filename = input('Image path: ')
img = cv2.imread(filename, 0)

if not img is None:
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(images, np.array(labels))
    label, rect = model.predict(img)
    print('label: ', label)
    cv2.imshow('img', img)
    cv2.waitKey(0)
else:
    print("Image could not load.")

cv2.destroyAllWindows()
