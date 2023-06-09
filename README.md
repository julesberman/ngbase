# Repo For Neural Schemes

## Setup

To install the ngbase package, navigate to the base directory `.../ngbase` (where this file resides) and run:

```bash
pip install --editable .
```

Since the package is under active development, the editable installation will allow any changes to the original package to reflect directly in the environment. 

To install all required packages run:

```bash
 pip install -r requirements.txt
```

## Current Working Test Runs

These configurations have been tested and should produce reasonable results within 1-10 min running on a laptop

```bash
# normal opt_dis
python ngbase/run.py -p ac -m opt_dis -t 1e-2 -T 2 
python ngbase/run.py -p ac -m opt_dis -t 1e-2 -T 2 --save_init True   # automatically save init condition 
python ngbase/run.py -p ac -m opt_dis -t 1e-2 -T 2 --load_init 'auto' # automatically load init condition 

# opt_dis with random sparse subsampling
python ngbase/run.py -p burgers -m opt_dis_sub -t 5e-3 -T 2 --depth 4 --sub_params 200

# opt_dis on multiple quantity bz problem
python ngbase/run.py -p bz -m opt_dis_sub -t 1e-2 -T 2

# solve 2D wave equations [may take long time without GPU]
 python ngbase/run.py -p wavebc -m opt_dis_sub -t 5e-3 -T 1 --batch_size 25_000 --depth 6 --sub_params 500

```

