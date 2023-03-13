from scripts.Visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
from scripts.DetectionsList import DetectionList, Detection
import copy

"""  DEFINE TRANSFORMATION MATRICES """
# frame 0 (main reference frame)
frame_0 = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])

frame_1 = np.array([[1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0]])

frame_2 = np.array([[0.707107, 0.0, 0.707107, 2.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-0.707107, 0.0, 0.707107, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])

""" DEFINE DETECTIONS """
d = DetectionList()
""" Detection 1 """
d1 = Detection()
d1.id = 0
d1.setBBoxPosition(position=[0.0, 0.0, 2.0])
rot_mat1 = np.array([[1.0, 0.0, 0.0],
                     [0.0, 0.707107, -0.707107],
                     [0.0, 0.707107, 0.707107]])
d1.setBBoxOrientation(rotation_mat=rot_mat1)
d1.setBBoxSize(size=[0.5, 0.5, 1.0])
""" Detection 2 """
# d2 = copy.deepcopy(d1)
d2 = Detection()  # TODO: check if it's possible to initialize with deepcopy in dataclass.
d2.id = 1
d2.setBBoxPosition(position=[1.5, 0.0, 2.0])
rot_mat2 = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])
d2.setBBoxOrientation(rotation_mat=rot_mat2)
d2.setBBoxSize(size=[0.5, 0.5, 1.0])

# print("\nd1 id:", id(d1))
# print("d2 id:", id(d2))
# print("\n d1 pos:", d1.bounding_box.position, "\n d2 pos:", d2.bounding_box.position)
# print("\n d1 affine_transformation id:", id(d1.affine_transform), "\n d2 affine_transformation id:", id(d2.affine_transform))

d.detections.append(d1)
d.detections.append(d2)
visualizer = Visualizer(list_of_transformations=[frame_0, frame_1, frame_2], detection_list=d)
visualizer.plotCoordinateFrame(T_0f=frame_0, n="0")
visualizer.plotCoordinateFrame(T_0f=frame_1, n="1")
visualizer.plotCoordinateFrame(T_0f=(frame_1 @ frame_2), n="2")
visualizer.drawDetections()
plt.show()
