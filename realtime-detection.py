import cv2
import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt



class ARMarkers:

    def __init__(self, marker_size=0.5):
        self.__wood_image = cv2.imread('./virtual-objects/wood.png')
        self.__mona_image = cv2.imread('./virtual-objects/monalisa.png')
        self.__roof_image = cv2.imread('./virtual-objects/roof.jpg')
        self.__painting_image = cv2.imread('./virtual-objects/painting.jpg')

        self.__marker_size = marker_size

        self.__mtx = None
        self.__dist = None
        self.__tvecs = None
        self.__rvecs = None

    # Load calibration data
    def load_calibration_data(self, filename):
        data = np.load(filename)
        self.__mtx = data['mtx']
        self.__dist = data['dist']
        self.__rvecs = data['rvecs']
        self.__tvecs = data['tvecs']

    # Create aruco dictionary
    def create_aruco_dictionary(self, aruco_dict):
        self.__dictionary = aruco.Dictionary_get(aruco_dict)
        self.__parameters = aruco.DetectorParameters_create()

    @staticmethod
    def world_to_image(axis, rvec, tvec, mtx, dist):
        points, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

        pp1 = points[0][0].ravel().astype(int)
        pp2 = points[1][0].ravel().astype(int)
        pp3 = points[2][0].ravel().astype(int)
        pp4 = points[3][0].ravel().astype(int)

        return pp1, pp2, pp3, pp4

    @staticmethod
    def project_point_to_image(point, rvec, tvec, mtx, dist):
        point, _ = cv2.projectPoints(np.array([point]), rvec, tvec, mtx, dist)
        return point[0][0].ravel().astype(int)

    @staticmethod
    def world_to_camera(point, rvec, tvec):
        # Convert rotation vector to rotation matrix
        rotation_matrix = cv2.Rodrigues(rvec.reshape(3, 1))[0].reshape(3, 3)
        translation_matrix = tvec.reshape(3, 1)

        # Get the transformation matrix
        transformation_matrix = np.concatenate((rotation_matrix, translation_matrix), axis=1)
        transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1])).reshape(4, 4)

        point = np.array([point[0], point[1], point[2], 1]).reshape(4, 1)

        # Transform the point
        transformed_point = np.dot(transformation_matrix, point)
        return transformed_point

    def __draw_image(self, points, virtual_image, dst, img):
        height, width, c = virtual_image.shape

        pp1, pp2, pp3, pp4 = points

        # cv2.circle(markers, pp1, 5, (255, 0, 0), -1)
        # cv2.circle(markers, pp2, 5, (0, 255, 0), -1)
        # cv2.circle(markers, pp3, 5, (0, 0, 255), -1)
        # cv2.circle(markers, pp4, 5, (255, 255, 0), -1)

        pts1 = np.array([pp1, pp2, pp3, pp4], dtype=np.float32)
        width_m = dst.shape[1]
        height_m = dst.shape[0]

        pts2 = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

        matrix, _ = cv2.findHomography(pts2, pts1)
        imgout = cv2.warpPerspective(virtual_image, matrix, (width_m, height_m))

        cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
        if imgout is not None:
            img = img + imgout

        return img

    def __draw_point(self, point, point_camera, z_buffer, markers, color=(255, 0, 0)):
        if point[0] >= 0 and point[0] < z_buffer.shape[0] and point[1] > 0 and point[1] < z_buffer.shape[1]:
            if z_buffer[point[0], point[1]] == 0 or z_buffer[point[0], point[1]] > point_camera[2]:
                cv2.circle(markers, point, 1, color, -1)
                z_buffer[point[0], point[1]] = point_camera[2]

    def run(self):
        assert self.__dictionary and self.__parameters, "Aruco dictionary is not created."
        assert self.__mtx is not None and self.__dist is not None and self.__rvecs is not None and self.__tvecs is not None, "Camera calibration was not loaded."

        cap = cv2.VideoCapture(0)

        row = np.arange(-self.__marker_size / 2, self.__marker_size / 2, .005)
        col = np.arange(-self.__marker_size / 2, self.__marker_size / 2, .005)

        while True:
            ret, frame = cap.read()
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.__mtx, self.__dist, (w, h), 1, (w, h))

            # undistort
            dst = cv2.undistort(frame, self.__mtx, self.__dist, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]

            # Create the z-buffer
            z_buffer = np.zeros((dst.shape[0], dst.shape[1], 1))

            gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.__dictionary, parameters=self.__parameters)
            markers = aruco.drawDetectedMarkers(dst.copy(), corners, ids)

            if ids is not None:
                for i in range(len(ids)):
                    c = corners[i][0]

                    # Draw axis on marker
                    self.__rvec, self.__tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], self.__marker_size, self.__mtx, self.__dist)

                    # aruco.drawAxis(markers, mtx, dist, rvec[0], tvec[0], 0.1)

                    # Get the edge points of the marker and show them
                    p1 = (int(c[0][0]), int(c[0][1]))
                    p2 = (int(c[1][0]), int(c[1][1]))
                    p3 = (int(c[2][0]), int(c[2][1]))
                    p4 = (int(c[3][0]), int(c[3][1]))

                    if ids[i][0] == 203:
                        # Draw an image
                        markers = self.__draw_image((p1, p2, p3, p4), self.__wood_image, dst, markers)

                        # Draw the walls
                        axis = np.float32(
                            [[-self.__marker_size / 2, self.__marker_size / 2, 0], [-self.__marker_size / 2, self.__marker_size / 2, self.__marker_size],
                             [self.__marker_size / 2, self.__marker_size / 2, 0], [self.__marker_size / 2, self.__marker_size / 2, self.__marker_size]]).reshape(
                            -1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__mona_image, dst, markers)

                        axis = np.float32([[-self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [-self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size],
                                           [self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size]]).reshape(-1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__mona_image, dst, markers)

                        axis = np.float32([[-self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [-self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size],
                                           [-self.__marker_size / 2, self.__marker_size / 2, 0],
                                           [-self.__marker_size / 2, self.__marker_size / 2, self.__marker_size]]).reshape(-1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__mona_image, dst, markers)

                        axis = np.float32([[-self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [-self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size],
                                           [self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size]]).reshape(-1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__mona_image, dst, markers)

                        axis = np.float32([[self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size],
                                           [self.__marker_size / 2, self.__marker_size / 2, 0],
                                           [self.__marker_size / 2, self.__marker_size / 2, self.__marker_size]]).reshape(-1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__mona_image, dst, markers)
                    elif ids[i][0] == 23:
                        # Draw the walls
                        axis = np.float32(
                            [[-self.__marker_size / 2, self.__marker_size / 2, 0], [-self.__marker_size / 2, self.__marker_size / 2, self.__marker_size],
                             [self.__marker_size / 2, self.__marker_size / 2, 0], [self.__marker_size / 2, self.__marker_size / 2, self.__marker_size]]).reshape(
                            -1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__wood_image, dst, markers)

                        axis = np.float32([[-self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [-self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size],
                                           [self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size]]).reshape(-1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__wood_image, dst, markers)

                        axis = np.float32([[-self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [-self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size],
                                           [-self.__marker_size / 2, self.__marker_size / 2, 0],
                                           [-self.__marker_size / 2, self.__marker_size / 2, self.__marker_size]]).reshape(-1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__wood_image, dst, markers)

                        axis = np.float32([[-self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [-self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size],
                                           [self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size]]).reshape(-1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__wood_image, dst, markers)

                        axis = np.float32([[self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size],
                                           [self.__marker_size / 2, self.__marker_size / 2, 0],
                                           [self.__marker_size / 2, self.__marker_size / 2, self.__marker_size]]).reshape(-1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__wood_image, dst, markers)

                        # Roof
                        axis = np.float32([[-self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size],
                                           [0, -self.__marker_size / 2, self.__marker_size + self.__marker_size/2],
                                           [-self.__marker_size / 2, self.__marker_size / 2, self.__marker_size],
                                           [0, self.__marker_size / 2, self.__marker_size + self.__marker_size/2]]).reshape(-1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__roof_image, dst, markers)

                        axis = np.float32([[self.__marker_size / 2, -self.__marker_size / 2, self.__marker_size],
                                           [0, -self.__marker_size / 2, self.__marker_size + self.__marker_size/2],
                                           [self.__marker_size / 2, self.__marker_size / 2, self.__marker_size],
                                           [0, self.__marker_size / 2, self.__marker_size + self.__marker_size/2]]).reshape(-1, 3)
                        pp1, pp2, pp3, pp4 = self.world_to_image(axis, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                        markers = self.__draw_image((pp1, pp2, pp4, pp3), self.__roof_image, dst, markers)
                    elif ids[i][0] == 62:
                        axis_p = np.float32([[-self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [-self.__marker_size / 2, self.__marker_size / 2, 0],
                                           [self.__marker_size / 2, -self.__marker_size / 2, 0],
                                           [self.__marker_size / 2, self.__marker_size / 2, 0]]).reshape(-1, 3)

                        for i in row:
                            for j in col:
                                point_t = np.float32([i, j, self.__marker_size/2])
                                p_camera = self.world_to_camera(point_t, self.__rvec, self.__tvec)
                                image_point = self.project_point_to_image(point_t, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                                self.__draw_point(image_point, p_camera, z_buffer, markers, color=(0, 0, 255))

                        for i in row:
                            for j in col:
                                point_t = np.float32([i, j, self.__marker_size])
                                p_camera = self.world_to_camera(point_t, self.__rvec, self.__tvec)
                                image_point = self.project_point_to_image(point_t, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                                self.__draw_point(image_point, p_camera, z_buffer, markers, color=(0, 255, 0))

                        for i in row:
                            for j in col:
                                point_t = np.float32([i, j, 0])
                                p_camera = self.world_to_camera(point_t, self.__rvec, self.__tvec)
                                image_point = self.project_point_to_image(point_t, self.__rvec, self.__tvec, self.__mtx, self.__dist)
                                self.__draw_point(image_point, p_camera, z_buffer, markers, color=(255, 0, 0))
                    else:
                        # Draw an image
                        markers = self.__draw_image((p1, p2, p3, p4), self.__painting_image, dst, markers)
                    # cv2.line(markers, pp1, pp4, (0, 255, 0), 3)


            cv2.imshow('markers', markers)

            # cv2.imshow('frame', dst)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    ar_markers = ARMarkers()
    ar_markers.load_calibration_data(filename='calibration_data.npz')
    ar_markers.create_aruco_dictionary(aruco_dict=aruco.DICT_6X6_250)
    ar_markers.run()