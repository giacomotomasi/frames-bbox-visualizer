from dataclasses import dataclass, field
from scripts.Detection import Detection


@dataclass
class DetectionList:
    detections: list = field(default_factory=list)

