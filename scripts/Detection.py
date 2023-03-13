from dataclasses import dataclass, field
import numpy as np
from scripts.BoungingBox import BoundingBox
import copy


# @dataclass
# class Detection:
#     id: int = 0
#     bounding_box: BoundingBox = BoundingBox()
#     # affine_transform: np.array = np.identity(4)
#     affine_transform: np.ndarray = copy.deepcopy(np.identity(4))
#
#     def computeAffineTransformation(self):
#         self.affine_transform[:3, :3] = self.bounding_box.orientation.rotation_matrix
#         self.affine_transform[:, 3] = np.array(
#             [self.bounding_box.position.x, self.bounding_box.position.y, self.bounding_box.position.z, 1.0])
#
#     def setBBoxPosition(self, position=[0.0, 0.0, 0.0]):
#         self.bounding_box.position.x = position[0]
#         self.bounding_box.position.y = position[1]
#         self.bounding_box.position.z = position[2]
#
#     def setBBoxOrientation(self, rotation_mat=np.identity(3)):
#         self.bounding_box.orientation.rotation_matrix = rotation_mat
#         self.computeAffineTransformation()
#
#     def setBBoxSize(self, size=[1.0, 1.0, 1.0]):
#         self.bounding_box.size.x = size[0]
#         self.bounding_box.size.y = size[1]
#         self.bounding_box.size.z = size[2]


class Detection:
    def __init__(self, _id=0, _bounding_box=BoundingBox(), affine_mat=np.identity(4)):
        self.id = _id
        self.bounding_box = copy.deepcopy(_bounding_box)
        self.affine_transform = copy.deepcopy(affine_mat)

    def computeAffineTransformation(self):
        self.affine_transform[:3, :3] = self.bounding_box.orientation.rotation_matrix
        self.affine_transform[:, 3] = np.array([self.bounding_box.position.x, self.bounding_box.position.y, self.bounding_box.position.z, 1.0])

    def setBBoxSize(self, size=[1.0, 1.0, 1.0]):
        self.bounding_box.size.x = size[0]
        self.bounding_box.size.y = size[1]
        self.bounding_box.size.z = size[2]

    def setBBoxPosition(self, position=[0.0, 0.0, 0.0]):
        self.bounding_box.position.x = position[0]
        self.bounding_box.position.y = position[1]
        self.bounding_box.position.z = position[2]

    def setBBoxOrientation(self, rotation_mat=np.identity(3)):
        self.bounding_box.orientation.rotation_matrix = rotation_mat
        self.computeAffineTransformation()






