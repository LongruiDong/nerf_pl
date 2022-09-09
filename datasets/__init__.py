from .blender import BlenderDataset
from .llff import LLFFDataset
from .phototourism import PhototourismDataset
from .Replica import ReplicaDataset
from .Replica_gt import ReplicaGTDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'phototourism': PhototourismDataset,
                'replica': ReplicaDataset,
                'replicagt': ReplicaGTDataset} # use gt pose depth instead from colmap