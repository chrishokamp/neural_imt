## Interactive Neural Machine Translation

### Installation
```
mkdir $HOME/projects
cd $HOME/projects
git clone https://bitbucket.org/chris_hokamp/neural_imt
git clone https://bitbucket.org/chris_hokamp/neural_mt

export PYTHONPATH=$HOME/projects/neural_mt:$PYTHONPATH
export PYTHONPATH=$HOME/projects/neural_imt:$PYTHONPATH

# Install Anconda
wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
bash Anaconda2-4.2.0-Linux-x86_64.sh 

echo 'export PATH=$PATH:$HOME/anaconda2/bin' >> $HOME/.bashrc
source $HOME/.bashrc

# make an Anaconda Theano environment
conda create --name theano python=2.7 anaconda
source activate theano

source activate theano

# Install blocks and fuel
# stable
pip install git+git://github.com/mila-udem/blocks.git \
    -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt

# now upgrade
pip install git+git://github.com/mila-udem/blocks.git \
    -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt --upgrade
    
# Install blocks-extras
cd ~/projects
git clone https://github.com/mila-udem/blocks-extras.git
cd blocks-extras
echo "export PYTHONPATH=\$PYTHONPATH:`pwd`" >> ~/.bashrc
source ~/.bashrc


# Install Theano -- note it's important to do this after Blocks
git clone git://github.com/Theano/Theano.git
cd Theano
python setup.py develop --user

# setup ~.theanorc
echo -e "\n[global]\ndevice=gpu\nfloatX=float32\noptimizer=fast_run\n[cuda]\nroot=/usr/local/cuda-8.0\n[blas]\nldflags=-lblas -lgfortran\n[nvcc]\nfastmath=True" >> ~/.theanorc



# Make required patches in Blocks(?) -- TODO: fix this(!)
# currently This will break if we are not on the new branch
# the `prefix_attention` branch fixes this because we provide a subclass of `AttentionRecurrent`

```


### Running Experiments


```
source activate theano
cd ~/projects/neural_imt

git checkout origin/prefix_attention
git checkout -b prefix_attention
python -m nn_imt -m train experiments/configs/en-de/prefix_attention/en-de_interactive_prefix_attention_baseline.yaml
```
- Get the data you need

- prep the data

- create a configuration file

- run the experiments!

