export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=$SGE_GPU /srv/glusterfs/vli/.pyenv/shims/python ~/webcam-location/save_reverse_pkl.py --sunrise_model=/srv/glusterfs/vli/models/best/sunrise2/ --sunset_model=/srv/glusterfs/vli/models/best/sunset2/
