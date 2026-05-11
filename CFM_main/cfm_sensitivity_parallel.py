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
sites = ['EC'] # ,'T','Z','KQU','KPS']
runs_dict = {
            'temp':[0, 0.5, 1, 2], # 
             'precip':[1, 1.05, 1.1, 1.2], # 
             }

# Define number of runs and number of simultaneous processers
# to enable doing more runs than you have processers
n_runs = len(sites) * sum([len(runs_dict[n]) for n in runs_dict])
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

    # Loop through temperature and precip changes
    for run_type in runs_dict:
        # Loop through values in either temperature or precip changes
        list_runs = runs_dict[run_type]
        for value in list_runs:
            # Add glacier/site to the filepaths
            fn_data = fp_forcings + args.glacier + args.site +'/'
            fn_out = fp_out + args.glacier + args.site +'/'

            # Specify string to add to filename
            if run_type == 'temp':
                var_str = 'temp+'+str(value) if value >= 0 else 'temp'+str(value)
            elif run_type == 'precip':
                var_str = 'tpx'+str(value)

            # Get filenames including the sensitivity run information
            fn_data += f'{args.glacier}{site}_1d_{var_str}_forcings.csv' # 
            fn_out += args.glacier + args.site + '_' + var_str + '_redo/' 

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