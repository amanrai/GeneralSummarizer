#Needed because the A4000 needs a specific install of Conda. The pip install doesnt work, so install from the Conda Forge Repo. 
#This is a one time install, so it is not in the requirements.txt file.

#conda create -n liaon python=3.9
#conda activate liaon
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

