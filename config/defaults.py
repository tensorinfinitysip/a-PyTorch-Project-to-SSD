from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 21
_C.MODEL.META_ARCHITECTURE = 'ssd'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Channel for input image
_C.INPUT.IN_CHANNEL = 3
# Size of the image during training
_C.INPUT.SIZE_TRAIN = 300
# Size of the image during test
_C.INPUT.SIZE_TEST = 300
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [104, 117, 123]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = 'VOC'
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
# If keep difficult
_C.DATASETS.USE_DIFFICULT = False
# Dataset root
_C.DATASETS.ROOT = '/mnt/truenas/scratch/xingyu.liao/DATA/VOCdevkit/VOC2012'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.CONV_BODY = 'vgg16'
_C.MODEL.BACKBONE.ARCHITECTURE = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
_C.MODEL.BACKBONE.WEIGHT = '/mnt/truenas/scratch/xingyu.liao/model_zoo/vgg16_reducedfc.pth'

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.IN_CHANNEL = 1024
_C.MODEL.RPN.ARCHITECTURE = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]

# ---------------------------------------------------------------------------- #
# Anchor options
# ---------------------------------------------------------------------------- #
_C.MODEL.ANCHOR = CN()
_C.MODEL.ANCHOR.VARIANCE = [0.1, 0.2]
_C.MODEL.ANCHOR.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
_C.MODEL.ANCHOR.MIN_SIZES = [30, 60, 111, 162, 213, 264]
_C.MODEL.ANCHOR.MAX_SIZES = [60, 111, 162, 213, 264, 315]
_C.MODEL.ANCHOR.STRIDES = [8, 16, 32, 64, 100, 300]
_C.MODEL.ANCHOR.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
_C.MODEL.ANCHOR.NUM_PER_FEAT = [4, 6, 6, 6, 4, 4]  # number of anchors per feature map location
_C.MODEL.ANCHOR.CLIP = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
