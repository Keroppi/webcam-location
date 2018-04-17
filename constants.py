CLUSTER = True
LEARNING_SUNRISE = True # False for sunset
DAYS_PER_MONTH = 3 # 'MAX'
SIZE = ['small'] # 'large'
DATA_SOURCES = ['roundshot'] # 'panomax'
IMAGES_PER_DAY = 32
PATCH_SIZE = (128, 128) # height, width
NUM_CHANNELS = 3 # RGB
BATCH_SIZE = 70
NUM_LOADER_WORKERS = 8
EPOCHS = 5
FIRST_FC_LAYER_SIZE = 16 * 15 * 8 * 8 # Output dimensionality of final convolution layer.
LOG_INTERVAL = 100
SPLIT_TOTAL = 100
SPLIT_TRAIN = 89 # 90% train
SPLIT_TEST = 94 # 5% test
SPLIT_VALID = 99 # 5% validation



