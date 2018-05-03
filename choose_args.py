import sys, random, pickle
import constants

def RandomizeArgs(SGE_TASK_ID):
    conv_num_layers = random.randint(3, 6)

    kernel_sizes = [(1, random.randint(2, 6), random.randint(2, 6))] + [(random.randint(1, 4), random.randint(2, 6), random.randint(2, 6)) for x in range(conv_num_layers - 1)]

    if constants.CLUSTER:
        output_channels = [random.randint(12, 100) for x in range(conv_num_layers)]
    else:
        output_channels = [random.randint(1, 1) for x in range(conv_num_layers)]

    paddings = [(0, random.randint(0, 2), random.randint(0, 2)) for x in range(conv_num_layers)]
    strides = [(1, random.randint(1, 2), random.randint(1, 2))] + [(random.randint(1, 3), random.randint(1, 2), random.randint(1, 2)) for x in range(conv_num_layers - 1)]
    max_poolings = [(random.randint(1, 2), random.randint(2, 4), random.randint(2, 4))
                    if random.randint(1, conv_num_layers) >= 3 else None # On average 2 layers have no max pooling.
                    for x in range(conv_num_layers)]
    max_pooling_strides = [(1, random.randint(1, 2), random.randint(1, 2))] + [(random.randint(1, 3), random.randint(1, 2), random.randint(1, 2)) for x in range(conv_num_layers - 1)]
    conv_relus = [True if random.randint(0, 1) == 1 else False for x in range(conv_num_layers)]

    if constants.CLUSTER:
        num_hidden_fc_layers = random.randint(2, 6)
    else:
        num_hidden_fc_layers = random.randint(1, 1)

    fc_sizes = [random.randint(20 + x * 1000, 1000 + x * 1000) for x in range(num_hidden_fc_layers - 1, -1, -1)]

    #for i in range(1, num_hidden_fc_layers): # Decreasing width of layers.
    #    fc_sizes[i] = int(fc_sizes[i - 1] * 0.8)

    fc_relus = [True if random.randint(0, 1) == 1 else False for x in range(num_hidden_fc_layers)]

    parameters = (conv_num_layers, output_channels, kernel_sizes, paddings, strides, max_poolings, max_pooling_strides, conv_relus,
                  num_hidden_fc_layers, fc_sizes, fc_relus,
                  (constants.NUM_CHANNELS, constants.IMAGES_PER_DAY, constants.PATCH_SIZE[0], constants.PATCH_SIZE[1]))

    # Pickle to find batch size later.
    if constants.CLUSTER:
        dir = '/srv/glusterfs/vli/pickle/'
    else:
        dir = '/home/vli/pickle/'
    with open(dir + 'model_structure' + SGE_TASK_ID + '.pkl', 'wb') as f:
        pickle.dump(parameters, f)

    return parameters

# Fill out by hand if trying a specific configuration.
def ManualArgs(SGE_TASK_ID):
    conv_num_layers = 4
    kernel_sizes = [(1, 5, 5),
                    (4, 2, 2),  # Look at 4 frames at once.
                    (1, 2, 2),
                    (1, 2, 2)]
    output_channels = [16, 32, 48, 16]
    paddings = [(0, 2, 2), (0, 1, 1), (0, 1, 1), (0, 0, 0)]
    strides = [(1, 2, 2), (2, 1, 1), (1, 1, 1), (1, 1, 1)]
    max_poolings = [(1, 2, 2), (1, 2, 2), None, (1, 2, 2)]
    max_pooling_strides = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
    conv_relus = [True, True, False, True]

    num_hidden_fc_layers = 2
    fc_sizes = [2000, 200]
    fc_relus = [True, True]
    parameters = (conv_num_layers, output_channels, kernel_sizes, paddings, strides, max_poolings, max_pooling_strides, conv_relus,
                  num_hidden_fc_layers, fc_sizes, fc_relus,
                  (constants.NUM_CHANNELS, constants.IMAGES_PER_DAY, constants.PATCH_SIZE[0], constants.PATCH_SIZE[1]))

    # Pickle to find batch size later.
    if constants.CLUSTER:
        dir = '/srv/glusterfs/vli/pickle/'
    else:
        dir = '/home/vli/pickle/'
    with open(dir + 'model_structure' + SGE_TASK_ID + '.pkl', 'wb') as f:
        pickle.dump(parameters, f)

    return parameters