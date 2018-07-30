export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=$SGE_GPU /srv/glusterfs/vli/.pyenv/shims/python ~/webcam-location/use_pred_pkl.py --pickle_file=/srv/glusterfs/vli/models/best/sunrise2/predictions/pred.pkl
