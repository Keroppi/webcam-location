import datetime, time

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
BANDWIDTH = 0.2
            # 0.8 - 332
            # 0.7 - 306*
            # 0.6 - 294*
            # 0.5 - 285*
            # 0.4 - 276*
            # 0.3 - 307
            # 0.2 - 255* -
            # 0.175 -
            # 0.15 -
            # 0.125 - 324
            # 0.1 -
            # 0.075 -
            # 0.05 - 292*

            ## 1 - 466*
            ## 0.9 -
            ## 0.8 - 418*
            ## 0.6 - 374*
            ## 0.5 -
            ## 0.4 - 346*
            ## 0.3 -
            ## 0.25 -
            ## 0.2 - 333* -
            ## 0.17 -
            ## 0.125 -
            ## 0.1 - 410
            ## 0.05 -

            ### 0.5 - 84
            ### 0.3 - 76
            ### 0.2 - 74
            ### 0.1 - 71
            ### 0.05 - 68 -
            ### 0.01 - 132
CENTER = False
LAMBDA = 1 # REGULARIZER
INLIER_THRESHOLD = 1000
                   # 1500 - 290
                   # 1400 - 271*
                   # 1200 - 272*
                   # 1000 - 258* -
                   # 800 - 272*
                   # 600 - 284*
                   # 500 - 341
                   # 450 -
                   # 400 - 303*
                   # 350 -
                   # 300 -
                   # 200 - 392

                   ## 400 - 401*
                   ## 500 -
                   ## 600 - 352*
                   ## 700 -
                   ## 800 - 333*
                   ## 1000 - 330* -
                   ## 1200 - 345*


                   ### 80 - 141
                   ### 65 - 135
                   ### 50 - 128
                   ### 35 - 121
                   ### 20 - 113*
                   ### 10 - 113

# PARTICLE
AZIMUTHAL_INLIER_THRESHOLD = 0.12566370614359174 # 800 km - 273*

# 0.23561944901923448 # 1500 km - 296
# 0.1884955592153876 # 1200 km - 272*
# 0.15707963267948966 # 1000 km - 259* -
# 0.1411098700237413 # 900 km - 298
# 0.12566370614359174 # 800 km - 273*
# 0.10978120995044331 # 700 km - 312
# 0.09407324668249435 # 600 km - 279*
# 0.07853981633974483 # 500 km -
# 0.07068583470577036 # 450 km -
# 0.06283185307179587 # 400 km - 302*
# 0.05497787143782138 # 350 km -
# 0.0470366233412471 # 300 km -
# 0.007839437223541183 # 50 km -
# 0.0031357748894164732 # 20 km -

                             ## 300 -
                             ## 400 - 399*
                             ## 500 -
                             ## 600 - 338*
                             ## 700 -
                             ## 800 - 326* -
                             ## 1000 - 329*
                             ## 1200 - 372*
                             ## 1500 -

                             ### 50 - ?
                             ### 20 - ?

# GMM
MAHALANOBIS_INLIER_THRESHOLD = 1.5
                               # 2.5 - 311
                               # 2.25 - 299*
                               # 2 - 298*
                               # 1.75 - 287* -
                               # 1.5 - 311
                               # 1.25 - 298*
                               # 1 - 316
                               # 0.75 - 312*
                               # 0.5 - 341
                               # 0.25 - 323*

                               ## 2.5 - 387*
                               ## 2 - 361*
                               ## 1.5 - 361* -
                               ## 1 - 404*
                               ## 0.75 -
                               ## 1.25 -
                               ## 0.5 - 407*
                               ## 0.2 -

                               ### 2 - 76
                               ### 1.5 - 76 -
                               ### 1 - 83
                               ### 0.5 - 114
                               ### 0.1 - 85

# Particle Mahalanobis
AZIMUTHAL_MAHALANOBIS_INLIER_THRESHOLD = 1
                                         # 2 - 422
                                         # 2.25 - 441*
                                         # 1.75 - 348*
                                         # 1.5 - 307*
                                         # 1.25 - 293*
                                         # 1 -
                                         # 0.9 - 297
                                         # 0.75 - 267* -
                                         # 0.5 - 312
                                         # 0.25 - 336*
                                         # 0.2 - 378

                                         ## 2.5 - 647*
                                         ## 2 - 528*
                                         ## 1.5 - 413*
                                         ## 1 - 346* -
                                         ## 0.75 -
                                         ## 0.5 - 348*
                                         ## 0.2 -

                                         ### 2 - ?
                                         ### 1 - ?
                                         ### 0.1 - ?
BIGM = 100

# UTC times below
WINTER_SOLSTICE_2016 = datetime.datetime(2016, 12, 21, 10, 44)
VERNAL_EQUINOX_2017 = datetime.datetime(2017, 3, 20, 10, 29)
SUMMER_SOLSTICE_2017 = datetime.datetime(2017, 6, 21, 4, 24) # start of dataset
AUTUMNAL_EQUINOX_2017 = datetime.datetime(2017, 9, 22, 20, 2)
WINTER_SOLSTICE_2017 = datetime.datetime(2017, 12, 21, 16, 28)
VERNAL_EQUINOX_2018 = datetime.datetime(2018, 3, 20, 16, 15) # end of dataset
SUMMER_SOLSTICE_2018 = datetime.datetime(2018, 6, 21, 10, 7)
AUTUMNAL_EQUINOX_2018 = datetime.datetime(2018, 9, 23, 1, 54)

# 1 pass
# Mean - 713 km
# Median - 302 km

# 4 pass
# Mean - 889.869959
# Median - 450.728295

# Actual
# Mean - 250
# Median - 113