import os.path
import pandas as pd

os.chdir('../SCAD/')

## GDSC drug log IC50 and binarized repsonse information extraction. The following processing step refer to:
## https://github.com/hosseinshn/MOLI/blob/master/preprocessing_scr/annotations.ipynb

# Download the following three files in data response folder
## https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/suppData/TableS1E.xlsx
## https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources///Data/suppData/TableS5C.xlsx
## https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/suppData/TableS4A.xlsx

COSMIC_ids = pd.read_excel("./data/response/TableS1E.xlsx")
COSMIC_ids = COSMIC_ids.iloc[3:,[1,2,8]]
COSMIC_ids = COSMIC_ids.iloc[:-1,]
COSMIC_ids.columns = ["name",'COSMIC','Tissur_descriptor1']
COSMIC_ids.set_index("name",inplace=True,drop=True)   ## 1001 * 2
names2COSMIC = dict(COSMIC_ids["COSMIC"])

df = pd.read_excel("./data/response/TableS5C.xlsx")
df.drop([0,1,2,3],inplace=True)
df = df.iloc[:,1:]

df.columns = df.iloc[0][:]
df = df.iloc[1:,:]
df.index = df.iloc[:,0]
df.index.name = "cell_line"
df = df.iloc[:,1:]

IC50_thr = df.iloc[0,:]
IC50_thr.name = "logIC50_threshold"
df = df.iloc[1:,:]
df.rename(names2COSMIC,axis="index",inplace=True)
df.head()
df.sort_values(by="cell_line",inplace=True)
df.to_csv("./data/response/GDSC_response."+"all_drugs"+".tsv",sep="\t")

#### extract all drugs log(IC50) information ####
df_ic50 = pd.read_excel("./data/response/TableS4A.xlsx",'TableS4A-IC50s')
df_ic50 = df_ic50.iloc[3:,:]
df_ic50.drop("Unnamed: 1",axis=1,inplace=True)

df_ic50.columns = df_ic50.iloc[1,:].values
df_ic50 = df_ic50.iloc[2:,:]

df_ic50.index = df_ic50.iloc[:,0].astype('Int32')
df_ic50.index.name = "cell_line"
df_ic50 = df_ic50.iloc[:,1:]
df_ic50.sort_values(by="cell_line",inplace=True)
df_ic50.to_csv("./data/response/"+"GDSC_response."+"logIC50.all_drugs"+".tsv",sep="\t")
