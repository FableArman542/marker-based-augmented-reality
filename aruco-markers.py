import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt

# image = cv2.imread('./markers-tests/singlemarkersoriginal.jpg')
image = cv2.imread('./markers-tests/transferir.png')
# plt.figure()
# plt.imshow(image)
# plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()


# Print the dictionary
fig = plt.figure()
nx = 4
ny = 3
for i in range(1, nx*ny+1):
    ax = fig.add_subplot(ny,nx, i)
    img = aruco.drawMarker(dictionary,i, 700)
    plt.imshow(img, cmap =  'gray', interpolation = "nearest")
    ax.axis("off")

plt.show()

img = aruco.drawMarker(dictionary,23, 200)
plt.imshow(img, cmap = 'gray', interpolation = "nearest")
plt.show()

# Detect the markers in the image

corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)
# markers = aruco.detectMarkers(image, dictionary)
markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

plt.figure()
plt.imshow(markers)
for i in range(len(ids)):
    c = corners[i][0]
    plt.plot([c[:,0].mean()], [c[:,1].mean()], 'o', label='id={0}'.format(ids[i]))
plt.legend()
plt.show()

# plt.imshow(markers)
# plt.show()
