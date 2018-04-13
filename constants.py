LEARNING_SUNRISE = True # False for sunset
CLUSTER = True
DAYS_PER_MONTH = 2 # 'MAX'
SIZE = ['small'] # 'large'
DATA_SOURCES = ['roundshot'] # 'panomax'
IMAGES_PER_DAY = 32
PATCH_SIZE = (128, 128) # height, width
NUM_CHANNELS = 3 # RGB
BATCH_SIZE = 1500
NUM_LOADER_WORKERS = 8
EPOCHS = 2
FIRST_FC_LAYER_SIZE = 32 * 26 * 26 # Output dimensionality of final convolution layer.
LOG_INTERVAL = 100
SPLIT_TOTAL = 100
SPLIT_TRAIN = 89 # 90% train
SPLIT_TEST = 94 # 5% test
SPLIT_VALID = 99 # 5% validation



