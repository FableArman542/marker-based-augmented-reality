import numpy as np
import cv2
import glob

class CameraCalibration:

    # termination criteria
    def __init__(self, images_path):
        self.__criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.__objp = np.zeros((6*9,3), np.float32)
        self.__objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.__objpoints = [] # 3d point in real world space
        self.__imgpoints = [] # 2d points in image plane.

        self.__images = glob.glob(images_path)

        self.__ret = None
        self.__mtx = None
        self.__dist = None
        self.__rvecs = None
        self.__tvecs = None


    def calibrate_camera(self, debug=False):

        for fnam in self.__images:
            img = cv2.imread(fnam)
            # cv2.imshow(fnam, img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # gray_shape = gray.shape[::-1]

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                self.__objpoints.append(self.__objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.__criteria)
                self.__imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)

                if debug:
                    cv2.imshow('img',img)
                    cv2.waitKey(0)

        cv2.destroyAllWindows()

        # Calibrate camera
        self.__ret, self.__mtx, self.__dist, self.__rvecs, self.__tvecs = cv2.calibrateCamera(self.__objpoints, self.__imgpoints, gray.shape[::-1],None,None)

        return self.__ret, self.__mtx, self.__dist, self.__rvecs, self.__tvecs

    # Save calibration data
    def save_data(self):

        assert self.__ret and self.__mtx and self.__dist and self.__rvecs and self.__tvecs, "\"calibrate_camera\" has to be called before to compute the needed parameters"
        np.savez('calibration_data.npz', mtx = self.__mtx, dist = self.__dist, rvecs = self.__rvecs, tvecs = self.__tvecs)


if __name__ == "__main__":
    camera_calibration = CameraCalibration(images_path='./chessboard-images/*.jpg')
    camera_calibration.calibrate_camera()
    camera_calibration.save_data()