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
            # 0.3 - ?, 428
            # 0.2 - 453, 398
            # 0.175 - 450, 395
            # 0.15 - 436, 357
            # 0.125 - BEST 433, 335/302
            # 0.1 - 438, 337
            # 0.075 - ?, 355
            # 0.05 - 490, 410

            ## 0.3 - 416
            ## 0.25 - 411
            ## 0.2 - 383
            ## 0.17 - 399
            ## 0.125 - 396
            ## 0.05 - 547
CENTER = False
LAMBDA = 1 # REGULARIZER
INLIER_THRESHOLD = 600 # km
                   # 800 - ?, 343
                   # 600 - ?, 340/341
                   # 500 - 499, 362
                   # 450 - 490, 358
                   # 400 - BEST 468, 387
                   # 350 - 479
                   # 300 - 499, 382
                   # 200 - 524, 391

                   ## 400 - 469
                   ## 500 - 402
                   ## 600 - 368
                   ## 700 - 347
                   ## 800 - 365
AZIMUTHAL_INLIER_THRESHOLD = 0.09407324668249435 # 600 km - ?, 331
                             # 0.1411098700237413 # 900 km - ?, 342
                             # 0.10978120995044331 # 700 km - ?, 340
                             # 0.12566370614359174 # 800 km - ?, 337
                             # 0.07853981633974483 # 500 km - ?, 349
                             # 0.06283185307179587 # 400 km - ?, 356
                             # 0.0470366233412471 # 300 km - ?, 377
                             # 0.07068583470577036 # 450 km
                             # 0.05497787143782138 # 350 km - 455

                             ## 300 - 466
                             ## 400 - 420
                             ## 500 - 391
                             ## 600 - 353
                             ## 700 - 338
                             ## 800 - 338
AZIMUTHAL_MAHALANOBIS_INLIER_THRESHOLD = 1
MAHALANOBIS_INLIER_THRESHOLD = 1.25
                               # 2 - ?, 461 1 cluster
                               # 1.75 - ?, 451
                               # 1.5 - ?, 435
                               # 1.25 - ?, 400
                               # 1 - ?, 404 1 cluster
                               # 0.5 - ?, 522 with 1 cluster

                               ## 2 - 435 with 1 cluster
                               ## 1.5 - 409 with 1 cluster
                               ## 1 - 426 without Bayesian with 2 BIC clusters, 372 with 1 cluster, 426 with 2 clusters (AIC and BIC)
                               ## 0.5 - 488 with 1 cluster
BIGM = 100
