#

VENV_ROOT_DIR = "`which python | xargs dirname | xargs dirname`/envs/HyperSphere"
cd "$VENV_ROOT_DIR"
if [ ! -d "$VENV_ROOT_DIR/.git" ]; then
  mv HyperSphere HyperSphere_TBR
  cp -a HyperSphere_TBR/. ./
  rm -rf HyperSphere_TBR
fi
source activate HyperSphere 
conda install --yes pytorch torchvision -c soumith -n HyperSphere
conda install --yes --file requirements.txt -n HyperSphere

