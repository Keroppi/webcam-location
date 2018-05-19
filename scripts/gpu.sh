#grep -h $(whoami) /tmp/lock-gpu*/info.txt
#qsub -l gpu -l h_vmem=8G -q gpu.short.q@* gpu.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=$SGE_GPU ~/webcam-location/main.py
