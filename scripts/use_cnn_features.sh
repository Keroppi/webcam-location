export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=$SGE_GPU /srv/glusterfs/vli/.pyenv/shims/python ~/webcam-location/use_cnn_features.py --sunrise_path=/srv/glusterfs/vli/models/best/sunrise1/ --sunset_path=/srv/glusterfs/vli/models/best/sunset1/
