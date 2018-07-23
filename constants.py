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
BANDWIDTH = 0.05
            # 0.2 - 453
            # 0.175 - 450
            # 0.15 - 436
            # 0.125 - BEST 433, 335 in fixed outside
            # 0.1 - 438
            # 0.05 - 490

            ## 0.3 - 416
            ## 0.2 - 383
            ## 0.125 - 396
CENTER = False
LAMBDA = 1 # REGULARIZER
INLIER_THRESHOLD = 600 # km
                   # 500 - 499
                   # 450 - 490
                   # 400 - BEST 468, 387
                   # 350 - 479
                   # 300 - 499
                   # 200 - 524

                   ## 400 - 469
                   ## 500 - 402
                   ## 600 -
AZIMUTHAL_INLIER_THRESHOLD = 0.09407324668249435 # 600 km
                             # 0.07853981633974483 # 500 km
                             # 0.0470366233412471 # 300 km
                             # 0.05497787143782138 # 350 km - 455
                             # 0.06283185307179587 # 400 km - ?, 356
                             # 0.07068583470577036 # 450 km

                             ## 300 - 466
                             ## 400 - 420
                             ## 500 - 391
                             ## 600 -
MAHALANOBIS_INLIER_THRESHOLD = 0.5
                               ## 1 - 426 without Bayesian with 2 BIC clusters, 372 with 1 cluster, 426 with 2 clusters (AIC and BIC)
BIGM = 100
