LEARNING_SUNRISE = True # False for sunset
CLUSTER = False
SMALL_DATASET = (30, 2, 2) # Train, test, validation
DAYS_PER_MONTH = 5 # 'MAX'
SIZE = ['small'] # 'large'
DATA_SOURCES = ['roundshot'] # 'panomax'
IMAGES_PER_DAY = 32
PATCH_SIZE = (28, 28) # height, width
NUM_CHANNELS = 3 # RGB
BATCH_SIZE = 2
NUM_LOADER_WORKERS = 8
EPOCHS = 1
FIRST_FC_LAYER_SIZE = 32 * 26 * 26 # Output dimensionality of final convolution layer.
LOG_INTERVAL = 1
SPLIT_TOTAL = 100
SPLIT_TRAIN = 79 # 80% train
SPLIT_TEST = 89 # 10% test
SPLIT_VALID = 99 # 10% validation



