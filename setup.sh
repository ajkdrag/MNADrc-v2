apt-get install libglib2.0-0 -y
apt install libgl1-mesa-glx -y
apt install wget
apt install nano
pip install --upgrade pip 
pip install matplotlib --ignore-installed certifi
pip install opencv-python
pip install scikit-learn
pip install wandb
pip install gdown


gdown https://drive.google.com/uc?id=1KNuNJTyRi2F5A3Mwg1Egp6QyHwtDcROr
tar -xzvf shanghaitech.tar.gz
rm shanghaitech.tar.gz
mv ./data/frame_labels_shanghai.npy ./data/frame_labels_shanghaitech.npy

#gdown https://drive.google.com/uc?id=17dvOb-t3fIfV1--BW83z1fYaATC9L7hF
#tar -xzvf avenue.tar.gz
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_oHxf3VaI1Xt1aOUzAnCa180Ha-uYSpl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_oHxf3VaI1Xt1aOUzAnCa180Ha-uYSpl" -O ped2.tar.gz && rm -rf /tmp/cookies.txt
#tar -xzvf ped2.tar.gz
