import datetime, time

CLUSTER = True
LEARNING_SUNRISE = True # True for sunrise, False for sunset
DAYS_PER_MONTH = 'MAX'
SIZE = ['small'] # 'large'
#DATA_SOURCES = ['roundshot'] # 'panomax'
IMAGES_PER_DAY = 32
PATCH_SIZE = (128, 128) # height, width
NUM_CHANNELS = 3 # RGB or YCbCr
BATCH_SIZE = 120
NUM_LOADER_WORKERS = 4
EPOCHS = 200
LOG_INTERVAL = 20
SPLIT_TOTAL = 100
SPLIT_TRAIN = 75 # 75% train
SPLIT_TEST = 100 # 25% test, 0% validation
BANDWIDTH = 0.05
            # 0.8 - 332
            # 0.7 - 306, 184*
            # 0.6 - 294, 175*
            # 0.5 - 285, 174*
            # 0.4 - 276, 171*
            # 0.3 - 307
            # 0.2 - 255, 153* -
            # 0.175 -
            # 0.15 -
            # 0.125 - 324
            # 0.1 -
            # 0.075 -
            # 0.05 - 292, 190*

            ## 1 - 466, 275*
            ## 0.9 -
            ## 0.8 - 418, 231*
            ## 0.6 - 374, 206*
            ## 0.5 -
            ## 0.4 - 346, 182*
            ## 0.3 -
            ## 0.25 -
            ## 0.2 - 333, 185* -
            ## 0.17 -
            ## 0.125 -
            ## 0.1 - 410
            ## 0.05 -

            ### 0.5 - 84, 52
            ### 0.3 - 76, 45
            ### 0.2 - 74, 38
            ### 0.1 - 71, 34
            ### 0.05 - 68, 31 -
            ### 0.01 - 132, 132
CENTER = False
LAMBDA = 1 # REGULARIZER
INLIER_THRESHOLD = 10
                   # 1500 - 290
                   # 1400 - 271, 170*
                   # 1200 - 272, 171*
                   # 1000 - 258, 162* -
                   # 800 - 272, 169*
                   # 600 - 284, 177*
                   # 500 - 341
                   # 450 -
                   # 400 - 303, 203*
                   # 350 -
                   # 300 -
                   # 200 - 392

                   ## 400 - 401, 336*
                   ## 500 -
                   ## 600 - 352, 220*
                   ## 700 -
                   ## 800 - 333, 192* -
                   ## 1000 - 330, 202*
                   ## 1200 - 345, 198*


                   ### 80 - 141, 140
                   ### 65 - 135, 135
                   ### 50 - 128, 128
                   ### 35 - 121, 119
                   ### 20 - 113, 110
                   ### 10 - 113, 107 -

# PARTICLE
AZIMUTHAL_INLIER_THRESHOLD = 0.15707963267948966 # 1000 km - 259, 154* -

# 0.23561944901923448 # 1500 km - 296
# 0.1884955592153876 # 1200 km - 272, 169*
# 0.15707963267948966 # 1000 km - 259, 154* -
# 0.1411098700237413 # 900 km - 298
# 0.12566370614359174 # 800 km - 273, 162*
# 0.10978120995044331 # 700 km - 312
# 0.09407324668249435 # 600 km - 279, 177*
# 0.07853981633974483 # 500 km -
# 0.07068583470577036 # 450 km -
# 0.06283185307179587 # 400 km - 302, 198*
# 0.05497787143782138 # 350 km -
# 0.0470366233412471 # 300 km -
# 0.007839437223541183 # 50 km -
# 0.0031357748894164732 # 20 km -

                             ## 300 -
                             ## 400 - 399, 323*
                             ## 500 -
                             ## 600 - 338, 204*
                             ## 700 -
                             ## 800 - 326, 175* -
                             ## 1000 - 329, 183*
                             ## 1200 - 372, 187*
                             ## 1500 -

                             ### 50 - ?
                             ### 20 - ?

# GMM
MAHALANOBIS_INLIER_THRESHOLD = 1.5
                               # 2.5 - 311
                               # 2.25 - 299, 178*
                               # 2 - 298, 174*
                               # 1.75 - 287, 160* -
                               # 1.5 - 311
                               # 1.25 - 298, 171*
                               # 1 - 316
                               # 0.75 - 312, 193*
                               # 0.5 - 341
                               # 0.25 - 323, 196*

                               ## 2.5 - 387, 202*
                               ## 2 - 361, 206* -
                               ## 1.75 - 362, 213*
                               ## 1.5 - 361, 229*
                               ## 1 - 404, 255*
                               ## 0.75 -
                               ## 1.25 -
                               ## 0.5 - 407, 279*
                               ## 0.2 -

                               ### 2 - 76, 42
                               ### 1.5 - 76, 41 -
                               ### 1 - 83, 49
                               ### 0.5 - 114, 109
                               ### 0.1 - 85, 57

# Particle Mahalanobis
AZIMUTHAL_MAHALANOBIS_INLIER_THRESHOLD = 0.75
                                         # 2.25 - 441, 199*
                                         # 2 - 422
                                         # 1.75 - 348, 190*
                                         # 1.5 - 307, 181*
                                         # 1.25 - 293, 191*
                                         # 1 -
                                         # 0.9 - 297
                                         # 0.75 - 267, 184* -
                                         # 0.5 - 312
                                         # 0.25 - 336, 251*
                                         # 0.2 - 378

                                         ## 2.5 - 647, 325*
                                         ## 2 - 528, 264*
                                         ## 1.5 - 413, 223*
                                         ## 1 - 346, 194*
                                         ## 0.75 - 340, 194* -
                                         ## 0.5 - 348, 213*
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

MALI_LATITUDE = 17.5707

EQUINOX_DISCARD_DAYS = 35

# 1 pass
# Mean - 713 km, 504
# Median - 302 km, 169

# 4 pass
# Mean - 889.869959, 631
# Median - 450.728295, 255

# Actual
# Mean - 250, 187
# Median - 113, 103