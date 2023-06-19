

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

The method of using the API to call scFoundation is under construction. Now the code will generate randomly embeddings for training. In the future, we will update a version with calling the scFoundation API for model training.

## Expected output
You will get a folder named `results/demo/0.75/xxx` with the following files:
```
config.pkl
model.pt
params.csv
train.log
```