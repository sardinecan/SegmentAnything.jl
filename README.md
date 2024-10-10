# SegmentAnything.jl

## Installation
### Dependencies
#### Python
`SegmentAnything.jl` requires `python 3.8.6` with the following packages:
- `segment anything`
- `cv2`
- `matplotlib.pyplot`
- `numpy`
- `torch`

You can use [`pyenv`](https://github.com/pyenv/pyenv) to handle multiple versions of python on your computer. Once `pyenv` installed, run `pyenv install 3.8.6` and `pyenv global 3.8.6` to add and to activate the required version on your machine.

There is a `requirements.txt` file into `SegmentAnything.jl/deps/python/`. To install the python environment, you just need to create a virtual env and add the packages with the following commands
```python
# change directory 
cd SegmentAnything.jl/deps/python

# create venv
python3 -m venv .venv

# activate venv
source .venv/bin/activate

# install python packages from requirements.txt 
pip3 install -r requirements.txt
```
#### SAM
`SegmeAnything.jl` needs the [Segment Anything Model Checkpoints](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints). To download the 3 checkpoints, just run the `download_sam.jl` file located at `SegmentAnything/deps/sam`.

### SegmentAnything.jl
- Clone the repo
- In a terminal emulator, change the working directory: `cd path/to/SegmentAnything.jl`
- Activate the julia env in the julia REPL: `pkg> activate.`
- Instantiate the env to install deps: `pkg> instantiate`
- Then, load the module: `julia> include("SegmentAnything.jl")`

## Getting Started