"""
This script executes many simulations
in CFM simultaneously for the
temperature and precipitation sensitivity
test at multiple sites. 

See cfm_parallel for a simpler parallel
script.
"""

import cfm_claire as sim
import argparse
from multiprocessing import Pool
import copy
import socket 
import contextlib
import os
import traceback
import xarray as xr

# Parse command-line args
parser = argparse.ArgumentParser()
parser.add_argument('-n','--n_simultaneous_processes',action='store',type=int,default=1,
                    help='number of simultaneous processes')
parser.add_argument('-new','--new_spin',action='store',type=bool,default=True)
parser.add_argument('-site',action='store',type=str,default='Z')
parser.add_argument('-dt',action='store',type=str,default='1d')
parser.add_argument('-tag',action='store',type=str,default=None)
parser.add_argument('-physrho',nargs='+',default=['GSFC2020'],
                    help='Provide one or more densification schemes')
parser.add_argument('-config_name',default='CFMconfig.json',type=str,help='name of config file')
parser.add_argument('-spin_name',default='CFMspin.hdf5', type=str, help='name of spinup file')
parser.add_argument('-result_name',default='CFMresults.hdf5', type=str, help='name of spinup file')
parser.add_argument('-input_srho',default=0,
                    help='0 for default, 1 for variable, any number for other constant (e.g. 250)')
args_base = parser.parse_args()

# Set base filepaths for forcings and output
fp_forcings = '../../Firn/Forcings/'
fp_out = '../../Firn/Output/'

# Set the sites, variables and values to loop through in sensitivity test
sites = ['EC','T','Z','KQU','KPS']
base_fp = '/trace/group/rounce/cvwilson/Output/paper2/{g}{s}_subset/'
n_runs = 0
for site in sites:
    glacier = 'wolverine' if site == 'EC' else 'kahiltna' if 'K' in site else 'gulkana'
    for fn in os.listdir(base_fp.format(g=glacier, s=site)):
        n_runs += 1

# Define number of runs and number of simultaneous processers
# to enable doing more runs than you have processers
n_runs = len(sites) * n_runs
n_processes = args_base.n_simultaneous_processes
print(f'Beginning {n_runs} runs on {n_processes} processes')
if n_runs <= n_processes:
    n_runs_per_process = 1
    n_process_with_extra = 0
else:
    n_runs_per_process = n_runs // n_processes  # Base number of runs per CPU
    n_process_with_extra = n_runs % n_processes    # Number of CPUs with one extra run

# Create output directory
if 'trace' in socket.gethostname():
    fp_out = '/trace/group/rounce/cvwilson/Firn/Output/'
    fp_forcings = '/trace/group/rounce/cvwilson/Firn/Forcings/'

# Parse list for inputs to Pool function
packed_vars = [[] for _ in range(n_processes)]
run_no = 0  # Counter for runs added to each set
set_no = 0  # Index for the parallel process

# Loop through sites
for site in sites:
    # Copy args for this site
    args = copy.deepcopy(args_base)
    args.site = site
    args.glacier = 'wolverine' if site == 'EC' else 'kahiltna' if 'K' in site else 'gulkana'
    fp = base_fp.format(g=args.glacier, s=args.site)

    # Create the dataframe containing the PEBSI data for CFM forcing
    for fn in os.listdir(fp):
        param_fn = os.path.join(fp, fn)
        kp = fn.split('kp')[-1].split('_')[0]
        lapserate = fn.split('lapse_rate')[-1].split('_')[0]
        var_str = f'kp{kp}_lr{lapserate}'

        # Add glacier/site to the filepaths
        fn_data = fp_forcings + args.glacier + args.site +'/'
        fn_out = fp_out + args.glacier + args.site +'/'

        # Get filenames
        fn_data += f'{args.glacier}{site}_1d_{var_str}_forcings_recalibrate.csv' # 
        fn_out += f'{args.glacier}{args.site}_{var_str}_recalibrate/'

        # Load the dataset
        ds = xr.open_dataset(param_fn)
        timeres='1d'

        # get sublimation from any negative vaporsolid mass fluxes in m w.e.
        ds['vaporsolid'][ds['vaporsolid'] > 0] = 0
        ds['sublim'] = ds['vaporsolid']

        # change units of surftemp
        ds['surftemp'] += 273.15

        # resample to the specified resolution with sum (mass balance terms) and mean (surface temp)
        ds_mb = ds[['melt','accum','rainfall','sublim']].resample(time=timeres).sum()
        ds_mb *= 1000   # convert m w.e. to kg m-2
        ds_other = ds[['surftemp']].resample(time=timeres).mean()

        # merge datasets and rename
        data_in = xr.merge([ds_mb, ds_other])
        data_in = data_in.rename_vars({'melt':'SMELT', 'rainfall':'RAIN', 
                                        'surftemp':'TS', 'accum':'BDOT',
                                        'sublim':'SUBLIM'}) # , 'surfdens':'RHOS'

        # store data as a .csv       
        df = data_in[['BDOT','RAIN','TS','SMELT','SUBLIM']].to_dataframe()
        df.to_csv(fn_data)

        # Copy args for this run
        args_run = copy.deepcopy(args)

        # RERUN SPINUP????
        args_run.new_spin = False
        if args_run.new_spin:
            fn_out = fn_out[:-1] + '_respun/'

        # Pack vars
        packed_vars[set_no].append((fn_out, args_run, fn_data))

        # Check if moving to the next set of runs
        n_runs_set = n_runs_per_process + (1 if set_no < n_process_with_extra else 0)
        if run_no == n_runs_set - 1:
            set_no += 1
            run_no = -1

        # Advance counter
        run_no += 1

def run_cfm_parallel(list_inputs):
    """
    Executes the model on a single
    processor, called from Pool below.
    """
    # Storage for failed runs
    failed = []
    
    for inputs in list_inputs:
        # Unpack inputs
        fn_out, args, fn_data = inputs

        # Run the model
        print('Beginning',fn_out+args.result_name,'with',fn_data, flush=True)
        try:
            # Hush prints form CFM
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                sim.run_cfm(fn_out, args, fn_data, physRho='Crocus')
        except Exception as e:
            # If failed, print the error message
            failed.append(fn_out)
            print('FAILED IN', fn_out)
            traceback.print_exc()

        # Remove configuration file to clean up folders
        if os.path.exists(fn_out + args.config_name):
            os.remove(fn_out + args.config_name)

    # Print failed runs on this processor
    n_failed = len(failed)
    print(f'Finished process with {n_failed} failed')
    print(failed)
    print()

# Run model in parallel
with Pool(n_processes) as processes_pool:
    processes_pool.map(run_cfm_parallel,packed_vars)