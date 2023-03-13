from dataclasses import dataclass
import numpy as np


@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Orientation:
    rotation_matrix: np.array = np.identity(3)


@dataclass
class Size:
    x: float = 1.0
    y: float = 1.0
    z: float = 1.0


@dataclass
class BoundingBox:
    size: Size = Size()
    position: Position = Position()
    orientation: Orientation = Orientation()

######################################################################################################

# class Position():
#     def __init__(self):
#         self.x = 0.0
#         self.y = 0.0
#         self.z = 0.0
#
#
# class Orientation():
#     def __init__(self):
#         self.rotation_matrix = np.identity(3)
#
#
# class Size():
#     def __init__(self):
#         self.x = 0.0
#         self.y = 0.0
#         self.z = 0.0
#
#
# class BoundingBox():
#     def __init__(self):
#         self.size = Size()
#         self.position = Position()
#         self.orientation = Orientation()
