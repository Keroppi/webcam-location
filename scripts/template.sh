export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=$SGE_GPU /srv/glusterfs/vli/.pyenv/shims/python ~/webcam-location/template.py --pickle_file=/srv/glusterfs/vli/models/best/sunrise1/predictions/pred.pkl
