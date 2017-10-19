ch python | xargs dirname | xargs dirname`/envs/HyperSphere"
mv HyperSphere HyperSphere_TBR
cp -a HyperSphere_TBR/. ./
rm -rf HyperSphere_TBR
source activate HyperSphere
conda install --yes pytorch torchvision -c soumith -n HyperSphere
conda install --yes --file requirements.txt -n HyperSphere

