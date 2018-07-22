CLUSTER = True
LEARNING_SUNRISE = False # True for sunrise, False for sunset
DAYS_PER_MONTH = 'MAX'
SIZE = ['small'] # 'large'
#DATA_SOURCES = ['roundshot'] # 'panomax'
IMAGES_PER_DAY = 32
PATCH_SIZE = (128, 128) # height, width
NUM_CHANNELS = 3 # RGB or YCbCr
BATCH_SIZE = 140
NUM_LOADER_WORKERS = 4
EPOCHS = 200
LOG_INTERVAL = 20
SPLIT_TOTAL = 100
SPLIT_TRAIN = 75 # 75% train
SPLIT_TEST = 100 # 25% test, 0% validation
BANDWIDTH = 0.125
            # 0.2 - 453
            # 0.175 - 450
            # 0.15 - 436
            # 0.125 - BEST 433
            # 0.1 - 438
            # 0.05 - 490
CENTER = False
LAMBDA = 1 # REGULARIZER
INLIER_THRESHOLD = 400 # km
                   # 500 - 499
                   # 450 - 490
                   # 400 - BEST 468
                   # 350 - 479
                   # 300 - 499
                   # 200 - 524
AZIMUTHAL_INLIER_THRESHOLD = 0.06283185307179587 # 400 km - ?
                             # 0.05497787143782138 # 350 km - 455
                             # 0.07068583470577036 # 450 km
MAHALANOBIS_INLIER_THRESHOLD = 1 # 1 stdev away
BIGM = 100
