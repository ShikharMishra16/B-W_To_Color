import numpy as np
import cv2 as cv


prototext = r"[ENTER THE LOCATION OF THE prototxt FILE HERE]"
points = r"[ENTER THE LOCATION OF THE pts_in_hull.npy FILE HERE]"
model = r"[ENTER THE LOCATION OF THE caffemodel FILE HERE]"
image_path=r"[ENTER THE LOCATION OF THE INPUT BLACK AND WHITE IMAGE FILE HERE]"


net = cv.dnn.readNetFromCaffe(prototext, model)
pts = np.load(points)


class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


image = cv.imread(image_path)
scaled = image.astype("float32") / 255.0
lab = cv.cvtColor(scaled, cv.COLOR_BGR2LAB)

resized = cv.resize(lab, (224, 224))
L = cv.split(resized)[0]
L -= 50

print("Colorizing the image")
net.setInput(cv.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv.resize(ab, (image.shape[1], image.shape[0]))

L = cv.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv.cvtColor(colorized, cv.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

cv.imshow("Original", image)
cv.imshow("Colorized", colorized)
cv.waitKey(0)
