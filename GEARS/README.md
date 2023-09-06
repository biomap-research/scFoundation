## Data
We now provided demo data in the `data` folder. The demo data is a subset of the Dixit dataset. Then you need to  download the `go.csv.zip` from https://www.dropbox.com/s/wl0uxiz5z9dbliv/go.csv.zip?dl=0 and unzip it to the `data/demo/` folder.

## Installation
Install `PyG`, and then do `pip install cell-gears`. Actually we do not use the gears PyPI packages but installing it will install all the dependencies.

## Demo Usage
```
# cd to the GEARS folder
## no embedding
bash run_sh/run_singlecell_maeautobin-demo-baseline.sh

## with embedding
bash run_sh/run_singlecell_maeautobin-demo-emb.sh
```

Run `bash run_sh/run_singlecell_maeautobin-demo-baseline.sh` and `bash run_sh/run_singlecell_maeautobin-demo-emb.sh` to get the baseline and scFoundation results of the Demo data, respectively. 

The output text in the terminal will be redirected into the `train.log` file.

And the results will be saved in the `results` folder.  

Now the code will generate randomly embeddings for training. You can change the code in `gears/model.py` Line 130 to call the scFoundation API for model training. You need to set the API arguments `--output_type gene_batch` and `--pre_normalized A`. Then you will get the gene context embeddings for each training batch with shape batch\*19264\*hidden.

## Expected output
You will get a folder named `results/demo/0.75/xxx` with the following files:
```
config.pkl
model.pt
params.csv
train.log
```

The `Plot.ipynb` is the jupyter notebook of reproducing the figures. Since the raw data is about more than 10GB and we didn't include these data. If you want to execlute the code by yourself, please contact us.