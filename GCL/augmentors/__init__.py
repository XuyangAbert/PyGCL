from .augmentor import Graph, Augmentor, Compose, RandomChoice
from .identity import Identity
from .rw_sampling import RWSampling
from .ppr_diffusion import PPRDiffusion
from .markov_diffusion import MarkovDiffusion
from .edge_adding import EdgeAdding
from .edge_removing import EdgeRemoving
from .node_dropping import NodeDropping
from .node_shuffling import NodeShuffling
from .feature_masking import FeatureMasking
from .feature_dropout import FeatureDropout
from .edge_attr_masking import EdgeAttrMasking
from .fosr import FOSR
from .sdrf import SDRF

__all__ = [
    'Graph',
    'Augmentor',
    'Compose',
    'RandomChoice',
    'EdgeAdding',
    'EdgeRemoving',
    'EdgeAttrMasking',
    'FeatureMasking',
    'FeatureDropout',
    'Identity',
    'PPRDiffusion',
    'MarkovDiffusion',
    'NodeDropping',
    'NodeShuffling',
    'RWSampling'
    'FOSR',
    'SDRF',
]

classes = __all__
