
#%%
import os
import gzip
from cedalion import io, units
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from cedalion.vis.anatomy import scalp_plot

ROOT_DIR = os.path.join("/projectnb", "nphfnirs", "s", "users", "lcarlton", "ANALYSIS_CODE", "cedalion_regression_test", "tests", "data")

TASK = "BS"
NOISE_MODEL = "ar_irls"
REC_STR = 'conc_o'
TRIAL_TYPES = ["right", "left"]
optional_flag = ''
N_RUNS = 3
PROBE_DIR = os.path.join(ROOT_DIR, 'probe')

dirs = os.listdir(ROOT_DIR)
subject_list = [d for d in dirs if "sub" in d]


filepath = os.path.join(
    ROOT_DIR,
    f"task-{TASK}_channel_hrf_{NOISE_MODEL}{optional_flag}.pkl.gz",
)

with gzip.open(filepath, 'rb') as f:
    results = pickle.load(f)

group_avg = results['X_hrf_ts_weighted']
channels_motor = group_avg['channel'].str.contains('S55') | group_avg['channel'].str.contains('S53') #| group_avg['channel'].str.contains('S50')

#%%
for ss, subject in enumerate(subject_list):

    print(f"Processing subject {ss+1} of {len(subject_list)}")
    SAVE_DIR = os.path.join(ROOT_DIR, subject, 'processed_data')
    os.makedirs(SAVE_DIR, exist_ok=True)

    for file_idx in range(N_RUNS):

        filenm = f"{subject}_task-BS_run-0{file_idx+1}_nirs"
        print(f"\t\tProcessing  {file_idx+1} of {N_RUNS} files : {filenm}")

        subDir = os.path.join(ROOT_DIR, subject, "nirs")

        file_path = os.path.join(subDir, filenm)
        rec = io.read_snirf(file_path)[0]

        data_sel = rec['amp'].sel(channel=channels_motor)
        data_sel = data_sel.pint.dequantify()

        data_sel.to_netcdf(file_path[:-5] + '_sample.nc')


#%%
# channels_motor = group_avg['channel'].str.contains('S55') | group_avg['channel'].str.contains('S53') #| group_avg['channel'].str.contains('S50')
hrf = X_hrf_ts_mean_weighted.sel(chromo='HbO', time=slice(5,8)).mean('time')
# hrf[~channels_motor] = 0 * units.micromolar
fig, ax = plt.subplots(figsize=[10,10])
scalp_plot(amp,geo3d_motor, hrf, ax, optode_labels=True, vmin=-max(hrf).values, vmax=max(hrf).values)
# %% save geo3d

recs = io.read_snirf('/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS_bids_v2/sub-618/nirs/sub-618_task-BS_run-01_nirs.snirf')
geo3d_orig = recs[0].geo3d

landmarks = ['Nz', 'LPA', 'RPA', 'Cz', 'Iz']
labels = list(np.unique(motor.source.values)) + list(np.unique(motor.detector.values)) + landmarks
geo3d_motor = geo3d_orig.sel(label=labels)



# %%
