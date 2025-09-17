import cfm_claire as sim
import argparse
from multiprocessing import Pool
import copy
import socket 
import contextlib
import os
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('-n','--n_simultaneous_processes',action='store',type=int,default=1,
                    help='number of simultaneous processes')
parser.add_argument('-new','--new_spin',action='store',type=bool,default=True)
parser.add_argument('-site',action='store',type=str,default='Z')
parser.add_argument('-dt',action='store',type=str,default='1d')
parser.add_argument('-tag',action='store',type=str,default=None)
parser.add_argument('-physrho',nargs='+',default=['GSFC2020'],
                    help='Provide one or more densification schemes')
parser.add_argument('-input_srho',default=0,
                    help='0 for default, 1 for variable, any number for other constant (e.g. 250)')
args_base = parser.parse_args()

failed =[]
fp_forcings = '../../Firn/Forcings/'
fp_out = '../../Firn/Output/'
runs_dict = {'temp':[-5,-2,-1,0,1,2,5],
             'temp_sameacc':[-5,-2,-1,0,1,2,5],
             'precip':[0.5,0.667,0.9,1,1.1,1.5,2]}
sites = ['T','Z','EC','KPS']

n_runs = len(sites) * sum([len(runs_dict[n]) for n in runs_dict])
n_processes = args_base.n_simultaneous_processes
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

for site in sites:
    args = copy.deepcopy(args_base)
    args.site = site
    args.glacier = 'wolverine' if site == 'EC' else 'kahiltna' if site == 'KPS' else 'gulkana'

    for run_type in runs_dict:
        list_runs = runs_dict[run_type]
        for change in list_runs:
            fn_data = fp_forcings
            fn_out = fp_out
            if run_type == 'temp':
                temp_change_str = '+'+str(change) if change > 0 else str(change)
                fn_data += f'{args.glacier}/{args.glacier}{args.site}_{temp_change_str}C_{args.dt}_forcings.csv'
                fn_out += args.glacier + args.site + '_' + temp_change_str + '_0/'
            elif run_type == 'temp_sameacc':
                temp_change_str = '+'+str(change) if change > 0 else str(change)
                fn_data += f'{args.glacier}/{args.glacier}{args.site}_{temp_change_str}C_sameacc_{args.dt}_forcings.csv'
                fn_out += args.glacier + args.site + '_' + temp_change_str + '_sameacc_0/'
            elif run_type == 'precip':
                fn_data += f'../Data/{args.glacier}{args.site}_{change}_{args.dt}_forcings.csv'
                fn_out += args.glacier + args.site + '_' + str(change) + '_0/'

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
    failed = []
    
    for inputs in list_inputs:
        # Unpack inputs
        fn_out, args, fn_data = inputs

        # Run the model
        print('Beginning',fn_out,'with',fn_data)
        try:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                sim.run_cfm(fn_out, args, fn_data)
        except Exception as e:
            failed.append(fn_out)
            print('FAILED IN', fn_out)
            traceback.print_exc()

    n_failed = len(failed)
    print(f'Finished process with {n_failed} failed')
    print(failed)
    print()

# Run model in parallel
with Pool(n_processes) as processes_pool:
    processes_pool.map(run_cfm_parallel,packed_vars)