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
BANDWIDTH = 0.4
            # 0.8 - 332
            # 0.5 - 292
            # 0.3 - 307
            # 0.2 - 320
            # 0.175 -
            # 0.15 -
            # 0.125 - 324
            # 0.1 -
            # 0.075 -
            # 0.05 - 345

            ## 0.3 -
            ## 0.25 -
            ## 0.2 -
            ## 0.17 -
            ## 0.125 -
            ## 0.05 -
CENTER = False
LAMBDA = 1 # REGULARIZER
INLIER_THRESHOLD = 500
                   # 1500 - 290
                   # 1000 - 295
                   # 800 - 301
                   # 600 - 320
                   # 500 -
                   # 450 -
                   # 400 - 350
                   # 350 -
                   # 300 -
                   # 200 - 392

                   ## 400 -
                   ## 500 -
                   ## 600 -
                   ## 700 -
                   ## 800 -
AZIMUTHAL_INLIER_THRESHOLD = 0.10978120995044331 # 700 km -
# 0.23561944901923448 # 1500 km - 296
# 0.15707963267948966 # 1000 km - 290
# 0.1411098700237413 # 900 km - 298
# 0.12566370614359174 # 800 km - 297

                             # 0.09407324668249435 # 600 km - 319
                             # 0.07853981633974483 # 500 km -
                             # 0.07068583470577036 # 450 km -
# 0.06283185307179587 # 400 km - 354
                             # 0.05497787143782138 # 350 km -
                             # 0.0470366233412471 # 300 km -

                             ## 300 -
                             ## 400 -
                             ## 500 -
                             ## 600 -
                             ## 700 -
                             ## 800 -
MAHALANOBIS_INLIER_THRESHOLD = 1
                               # 2.5 - 311
                               # 2 - 302
                               # 1.75 - 305
                               # 1.5 - 311
                               # 1.25 - 304
                               # 1 -
                               # 0.5 - 341

                               ## 2 -
                               ## 1.5 -
                               ## 1 -
                               ## 0.5 -
AZIMUTHAL_MAHALANOBIS_INLIER_THRESHOLD = 1.5
                                         # 2 - 422
                                         # 1 - 283
                                         # 0.9 - 297
                                         # 0.75 - 292
                                         # 0.5 - 312
                                         # 0.2 - 378
BIGM = 100

# 1 pass
# Mean - 749 km
# Median - 332 km
