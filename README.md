# HyperSphere

To set up this repo, go:

** Virtual Environment _Without conda_ **

```
git clone https://github.com/ChangYong-Oh/HyperSphere.git
cd HyperSphere
source setup_pip.sh
```

** Virtual Environment _With conda_ **

```
conda create -n HyperSphere python=2.7.14 anaconda --yes
cd "`which python | xargs dirname | xargs dirname`/envs/HyperSphere"
git clone https://github.com/ChangYong-Oh/HyperSphere.git
cd HyperSphere
source setup_conda.sh
```

Default python should be the anaconda python.

Different python version is possible. For avaialbe version search
```
conda search "^python$"
```

** Import in existing Python environment **


Or to be able to import this code in an existing Python environment, go:

```
pip install -e git+https://github.com/ChangYong-Oh/HyperSphere.git#egg=HyperSphere
```
