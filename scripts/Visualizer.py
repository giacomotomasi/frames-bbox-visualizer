# plotCoordinateFrame function taken from:
# https://github.com/ethz-asl/kalibr/blob/master/Schweizer-Messer/sm_python/python/sm/plotCoordinateFrame.py
# and edited to add some extra features.

import numpy as np
import matplotlib.pyplot as plt
from scripts.DetectionsList import DetectionList, Detection
from scripts.BoungingBox import BoundingBox


class Visualizer(object):
    def __init__(self, list_of_transformations=[], detection_list=DetectionList()):
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(111, projection='3d')
        # PURE ROTATION MATRICES
        self.rotation_x = np.array([[1.0, 0.0, 0.0],
                                    [0.0, 0.707107, -0.707107],
                                    [0.0, 0.707107, 0.707107]])
        self.rotation_y = np.array([[0.707107, 0.0, 0.707107],
                                    [0.0, 1.0, 0.0],
                                    [-0.707107, 0.0, 0.707107]])
        self.rotation_z = np.array([[0.707107, -0.707107, 0.0],
                                    [0.707107, 0.707107, 0.0],
                                    [0.0, 0.0, 1.0]])

        self.transformation_matrices = list_of_transformations
        self.detections = detection_list
        self.colors = ['k', 'm', 'c', 'y', 'b', 'r', 'g']


    def getMinMax(self, size=[0.4, 0.4, 0.4]):
        """
        Compute min and max of each bbox coordinate (wrt its center)
        :param size: 3D dimensions of the bbox
        :return: min and max of each coordinate
        """
        xmin = -size[0] / 2
        xmax = size[0] / 2
        ymin = -size[1] / 2
        ymax = size[1] / 2
        zmin = -size[2] / 2
        zmax = size[2] / 2
        return xmin, ymin, zmin, xmax, ymax, zmax


    def draw3DBBox(self, detection):
        """
        Define Bounding Box points, transform them and plot the line connecting the points.
        :param ax: axis of type matplotlib.axes.Axes3D.
        :param size: size of the bounding box.
        :param transform: affine transformation matrix from frame 0 to frame n. Full transformation.
        """
        size = [detection.bounding_box.size.x, detection.bounding_box.size.y, detection.bounding_box.size.z]
        xmin, ymin, zmin, xmax, ymax, zmax = self.getMinMax(size=size)
        p1 = np.array([xmin, ymin, zmin, 1.0])
        p2 = np.array([xmax, ymin, zmin, 1.0])
        p3 = np.array([xmax, ymax, zmin, 1.0])
        p4 = np.array([xmin, ymax, zmin, 1.0])
        p5 = np.array([xmin, ymin, zmax, 1.0])
        p6 = np.array([xmax, ymin, zmax, 1.0])
        p7 = np.array([xmax, ymax, zmax, 1.0])
        p8 = np.array([xmin, ymax, zmax, 1.0])
        points = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
        new_points = np.zeros(points.shape)
        transform = self.computeFullTransformation()
        for i in range(len(points)):
            new_points[i] = (transform@detection.affine_transform)@points[i]
        self.ax.plot([new_points[0,0], new_points[1,0]], [new_points[0,1], new_points[1,1]], [new_points[0,2], new_points[1,2]], color=self.colors[detection.id%len(self.colors)])  # P12
        self.ax.plot([new_points[1,0], new_points[2,0]], [new_points[1,1], new_points[2,1]], [new_points[1,2], new_points[2,2]], color=self.colors[detection.id%len(self.colors)])  # P23
        self.ax.plot([new_points[2,0], new_points[3,0]], [new_points[2,1], new_points[3,1]], [new_points[2,2], new_points[3,2]], color=self.colors[detection.id%len(self.colors)])  # P34
        self.ax.plot([new_points[3,0], new_points[0,0]], [new_points[3,1], new_points[0,1]], [new_points[3,2], new_points[0,2]], color=self.colors[detection.id%len(self.colors)])  # P41
        self.ax.plot([new_points[4,0], new_points[5,0]], [new_points[4,1], new_points[5,1]], [new_points[4,2], new_points[5,2]], color=self.colors[detection.id%len(self.colors)])  # P56
        self.ax.plot([new_points[5,0], new_points[6,0]], [new_points[5,1], new_points[6,1]], [new_points[5,2], new_points[6,2]], color=self.colors[detection.id%len(self.colors)])  # P67
        self.ax.plot([new_points[6,0], new_points[7,0]], [new_points[6,1], new_points[7,1]], [new_points[6,2], new_points[7,2]], color=self.colors[detection.id%len(self.colors)])  # P78
        self.ax.plot([new_points[7,0], new_points[4,0]], [new_points[7,1], new_points[4,1]], [new_points[7,2], new_points[4,2]], color=self.colors[detection.id%len(self.colors)])  # P85
        self.ax.plot([new_points[0,0], new_points[4,0]], [new_points[0,1], new_points[4,1]], [new_points[0,2], new_points[4,2]], color=self.colors[detection.id%len(self.colors)])  # P15
        self.ax.plot([new_points[1,0], new_points[5,0]], [new_points[1,1], new_points[5,1]], [new_points[1,2], new_points[5,2]], color=self.colors[detection.id%len(self.colors)])  # P26
        self.ax.plot([new_points[2,0], new_points[6,0]], [new_points[2,1], new_points[6,1]], [new_points[2,2], new_points[6,2]], color=self.colors[detection.id%len(self.colors)])  # P37
        self.ax.plot([new_points[3,0], new_points[7,0]], [new_points[3,1], new_points[7,1]], [new_points[3,2], new_points[7,2]], color=self.colors[detection.id%len(self.colors)])  # P48
        # draw center point
        self.ax.scatter((transform@detection.affine_transform)[0, 3], (transform@detection.affine_transform)[1, 3], (transform@detection.affine_transform)[2, 3], color=self.colors[detection.id%len(self.colors)], linewidth=2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')


    def drawDetections(self):
        if len(self.detections.detections) == 0:
            print("There are no detections provided!")
        else:
            for det in self.detections.detections:
                self.draw3DBBox(det)


    def plotCoordinateFrame(self, T_0f, n=-1, size=1, linewidth=3):
        """Plot a coordinate frame on a 3d axis. In the resulting plot,
        x = red, y = green, z = blue.
        plotCoordinateFrame(axis, T_0f, size=1, linewidth=3)
        Arguments:
        axis: an axis of type matplotlib.axes.Axes3D
        T_0f: The 4x4 affine transformation matrix that takes points from the frame of interest, to the plotting frame
        n: frame name/number to be visualized. -1 in None
        size: the length of each line in the coordinate frame
        linewidth: the width of each line in the coordinate frame
        """
        p_f = np.array([[0, 0, 0, 1], [size, 0, 0, 1], [0, size, 0, 1], [0, 0, size, 1]]).T;
        p_0 = np.dot(T_0f, p_f)
        X = np.append([p_0[:, 0].T], [p_0[:, 1].T], axis=0)
        Y = np.append([p_0[:, 0].T], [p_0[:, 2].T], axis=0)
        Z = np.append([p_0[:, 0].T], [p_0[:, 3].T], axis=0)
        self.ax.plot3D(X[:, 0], X[:, 1], X[:, 2], 'r-', linewidth=linewidth)
        self.ax.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], 'g-', linewidth=linewidth)
        self.ax.plot3D(Z[:, 0], Z[:, 1], Z[:, 2], 'b-', linewidth=linewidth)
        if n != -1:
            self.ax.text(X[1, 0], X[1, 1], X[1, 2], "x_"+n, color='red')
            self.ax.text(Y[1, 0], Y[1, 1], Y[1, 2], "y_"+n, color='green')
            self.ax.text(Z[1, 0], Z[1, 1], Z[1, 2], "z_"+n, color='blue')
        self.ax.set_xlim3d(left=-1, right=5)
        self.ax.set_ylim3d(bottom=-3, top=3)
        self.ax.set_zlim3d(bottom=-1, top=5)
        # axis.axis('equal')


    def computeFullTransformation(self):
        """
        Compute the transformation matrix from last to first frame
        :return: 4x4 transformation from last to first frame
        """
        n_frames = len(self.transformation_matrices)
        full_transformation = np.identity(4)
        for i in range(n_frames):
            full_transformation = full_transformation@self.transformation_matrices[i]
        return full_transformation
