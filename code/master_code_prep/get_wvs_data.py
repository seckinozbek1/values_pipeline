import os
import sys
import pandas as pd
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter

# paths
RDATA_PATH = "../../data/WVS_Cross-National_Wave_7_Rdata_v6_0.rdata"
CSV_OUT    = "../../output/master_code_prep_output/wvs7_full_data.csv"

# ensure file exists
if not os.path.isfile(RDATA_PATH):
    print(f"File not found: {RDATA_PATH}", file=sys.stderr)
    sys.exit(1)

# load RData and get object names
objs = list(r["load"](RDATA_PATH))
if not objs:
    print(f"No objects found in {RDATA_PATH}", file=sys.stderr)
    sys.exit(1)

# select the second object
if len(objs) < 2:
    print("Less than two objects in the RData; cannot select objs[2].", file=sys.stderr)
    sys.exit(1)

target_name = str(objs[1])

# convert the selected R object to pandas DataFrame
with localconverter(default_converter + pandas2ri.converter):
    wvs7 = r[target_name]

if not isinstance(wvs7, pd.DataFrame):
    print(f"Selected object '{target_name}' is not an R data.frame (after conversion).", file=sys.stderr)
    sys.exit(1)

# save to csv 
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
wvs7.to_csv(CSV_OUT, index=False, encoding="utf-8")
print(f"Saved output to: {CSV_OUT}")
