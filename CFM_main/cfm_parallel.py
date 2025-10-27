"""
This script executes a few parallel runs
of CFM with the exact same arguments.
It is intended to quickly run the five
sites in my project but can also be
easily edited to run more parallel
simulations.

See cfm_sensitivity_parallel for the 
sensitivity simulations where tp and 
temp and perturbed.
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
parser.add_argument('-spin_name',default='CFMspin.hdf5', type=str, help='name of spinup file')
parser.add_argument('-result_name',default='CFMresults.hdf5', type=str, help='name of results file')
parser.add_argument('-config_name',default='CFMconfig.json',type=str,help='name of config file')
parser.add_argument('-physrho',nargs='+',default=['GSFC2020'],
                    help='Provide one or more densification schemes')
parser.add_argument('-input_srho',default=0,
                    help='0 for default, 1 for variable, any number for other constant (e.g. 250)')
args_base = parser.parse_args()

# Define filepaths to forcings and output folders
fp_forcings = '../../Firn/Forcings/'
fp_out = '../../Firn/Output/'

# Sites to loop through
sites = ['T','Z','EC','KPS','KQU']

# Define number of runs and number of simultaneous processers
# in case you want to do more runs than you have processers
n_runs = len(sites)
n_processes = args_base.n_simultaneous_processes
print(f'Beginning {n_runs} runs on {n_processes} processes')
if n_runs <= n_processes:
    n_runs_per_process = 1
    n_process_with_extra = 0
else:
    n_runs_per_process = n_runs // n_processes  # Base number of runs per CPU
    n_process_with_extra = n_runs % n_processes    # Number of CPUs with one extra run

# Edit base filepaths if in supercomputer, trace
if 'trace' in socket.gethostname():
    fp_out = '/trace/group/rounce/cvwilson/Firn/Output/'
    fp_forcings = '/trace/group/rounce/cvwilson/Firn/Forcings/'

# Parse list for inputs to Pool function
packed_vars = [[] for _ in range(n_processes)]
run_no = 0  # Counter for runs added to each set
set_no = 0  # Index for the parallel process

# Loop through sites and generate arguments for the CFM function
for site in sites:
    # Copy args for this site
    args = copy.deepcopy(args_base)
    args.site = site
    args.glacier = 'wolverine' if site == 'EC' else 'kahiltna' if 'K' in site else 'gulkana'

    # Set forcings filename (absolute filepath)
    fn_data = fp_forcings + f'{args.glacier}{args.site}/{args.glacier}{args.site}_{args.dt}_tpx1_forcings.csv'
    # fn_data = fp_forcings + f'{args.glacier}{args.site}/{args.glacier}{args.site}_1d_forcings.csv'
    
    # Set output filepath (absolute filepath to a folder)
    fn_out = fp_out + args.glacier + args.site +'/' + args.glacier + args.site
    if not os.path.exists(fp_out + args.glacier + args.site):
        # Make the site folder if it doesn't exist
        os.mkdir(fp_out + args.glacier + args.site)

    # Pack vars
    packed_vars[set_no].append((fn_out, args, fn_data))

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
    # Storage for failed runs on this processor
    failed = []
    
    for inputs in list_inputs:
        # Unpack inputs
        fn_out, args, fn_data = inputs

        # Run the model
        print('Beginning',fn_out,'with',fn_data, flush=True)
        try:
            # Hush prints from the model
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                sim.run_cfm(fn_out, args, fn_data, physRho='Crocus')
        except Exception as e:
            # Model failed: print the error message
            failed.append(fn_out)
            print()
            print('FAILED IN', fn_out)
            traceback.print_exc()
            print()

    # Print how many failed on this processor
    n_failed = len(failed)
    print(f'Finished process with {n_failed} failed')
    print(failed)
    print()

# Run model in parallel
with Pool(n_processes) as processes_pool:
    processes_pool.map(run_cfm_parallel,packed_vars)