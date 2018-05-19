#grep -h $(whoami) /tmp/lock-gpu*/info.txt
#qsub -l gpu -l h_vmem=8G -q gpu.short.q@* gpu.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=$SGE_GPU /srv/glusterfs/vli/.pyenv/shims/python ~/webcam-location/main.py --load_model_args=/srv/glusterfs/vli/models/best/sunset_model_structure2.pkl --resume=/srv/glusterfs/vli/models/best/sunset_checkpoint2.pth.tar
