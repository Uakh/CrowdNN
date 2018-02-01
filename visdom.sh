python -m visdom.server
firefox 127.0.0.1:8097
nvidia-smi -l 1

#How to install
pip install visdom
source create --name crowdnn
source activate crowdnn
conda install pytorch torchvision cuda80 -c soumith
conda install scikit-image matplotlib
