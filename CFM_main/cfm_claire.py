"""
This script is the main executor of CFM
developed by @clairevwilson for the use
in firn work in 2025. This script is 
somewhat specialized for this project
but mostly just specifies some arguments
provided by the model developer (Max Stevens)
and executes the model.
"""

import numpy as np 
import pandas as pd
import xarray as xr
import os
import time
import json
import shutil
import argparse
import firn_density_nospin as fdns
import RCMpkl_to_spin as RCM
import socket
machine = socket.gethostname()

if __name__ == '__main__':
    # If running from the command line, parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-new','--new_spin',action='store_false')
    parser.add_argument('-site',action='store',type=str,default='Z')
    parser.add_argument('-dt',action='store',type=str,default='1d')
    parser.add_argument('-tag',action='store',type=str,default=None)
    parser.add_argument('-physrho',nargs='+',default=['GSFC2020'],
                        help='Provide one or more densification schemes')
    parser.add_argument('-input_srho',default=0,
                        help='0 for default, 1 for variable, any number for other constant (e.g. 250)')
    parser.add_argument('-spin_name',default='CFMspin.hdf5', type=str, help='name of spinup file')
    parser.add_argument('-result_name',default='CFMresults.hdf5', type=str, help='name of results file')
    parser.add_argument('-config_name',default='CFMconfig.json',type=str,help='name of config file')
    args = parser.parse_args()

    # Check if inputting a time-varying surface density ("wsrho")
    args.glacier = 'wolverine' if args.site == 'EC' else 'kahiltna' if 'K' in args.site else 'gulkana'
    if type(args.tag) == str and 'pygem' in args.tag:
        # PYGEM FORCINGS
        forcing_fn = f'../../Firn/Forcings/{args.glacier}/{args.glacier}{args.site}_pygem_forcings.csv'
        args.dt = '1MS'
    elif int(args.input_srho) == 1:
        # TIME VARYING SURFACE DENSITY
        forcing_fn = f'../../Firn/Forcings/{args.glacier}/{args.glacier}{args.site}_{args.dt}_forcings_wsrho.csv'
    else:
        # CONSTANT SURFACE DENSITY
        forcing_fn = f'../../Firn/Forcings/{args.glacier}/{args.glacier}{args.site}_{args.dt}_forcings.csv'

def run_cfm(out_fp, args, forcing_fn, physRho='GSFC2020'):
    """
    This function gathers all the inputs and output
    filepaths for CFM and executes the model.
    
    Parameters
    ----------
    out_fp : str
        Output FOLDER to store CFMresults.hdf5
        and other output files
    args : namespace
        Command-line arguments
    forcing_fn : str
        Input FILENAME of forcing data in
        .csv with datetime index column
    """
    # DEFAULT CONFIGURATION FILE (used in all runs in my firn project)
    json_fn = 'my_configs.json'
    with open(json_fn) as file:
        c = json.load(file)

    # READ FORCINGS
    if os.path.exists(forcing_fn):
        print('Forcing with',forcing_fn)
        df = pd.read_csv(forcing_fn,parse_dates=True,index_col=0)

        # Clip to the start of a year (1981 is first full year in the data)
        df = df['1981':].copy()

    else:
        assert os.path.exists(forcing_fn), f'! Forcing file not found: generate and save to {forcing_fn}'

    # START TIMER
    tnow = time.time()

    # DEFINE SPIN DATES
    sds = 1981.0    # spin date start
    sde = 1995.0    # spin date end

    # ADD SOME ARGUMENTS TO CONFIG
    c['physRho'] = physRho
    c['DFresample'] = args.dt
    c['SEB'] = False # surface energy balance module OFF
    c['MELT'] = True # melt module ON
    c['rain'] = True # rainfall ON

    '''
    From Max (I did not touch these args):
    CFM regrids (merges) deeper nodes to save computation. There are 2 mergings
    nodestocombine and multnodestocombine should be adjusted based on the time resolution of the run
    e.g. if DFresample is '1d', nodestocombine = 30 will combine 30 layers at an intermediate depth, 
    and multnodestocombine = 12 will combine 12 of those layers at a greater depth (which in this case 
    will give 3 sections of firn - near the surface very thin layers, representing a day's accumulation,
    middle, which is a month's accumulation, and deep, that should be a year's accumulation. 
    e.g. if I am doing DFresample = '5d', I would set nodestocombine to 6 to still get layers that are a
    month's worth of accumulation. (there is no 'best' way to do this - it is a bit of an art)
    '''
    c['doublegrid'] = True
    c['nodestocombine'] = 30 
    c['multnodestocombine'] = 12

    # SURFACE DENSITY (FIXED OR VARIABLE)
    variable_srho = args.input_srho
    if variable_srho == True:
        # Take srho from inputs file
        c['variable_srho'] = True
        c['srho_type'] = 'userinput'
    else:
        # Use a constant srho
        if int(variable_srho) not in [0,1]:
            # Input a number for the surface density: use this constant
            option_rho = float(variable_srho)
        else:
            # Did not input a number for surface density: specify constant
            option_rho_dict = {'Z':366, 'T':347, 'EC':427, 'KPS':417, 'KQU':400} # KQU: 341
            if args.site in option_rho_dict:
                # Take from site dictinoary
                option_rho = option_rho_dict[args.site]
            else:
                # Take as baseline constant
                option_rho = 400
        # Add surface ensity to config
        c['rhos0'] = option_rho 

    # Set absolute filepath to results folder
    c['resultsFolder'] = out_fp 

    # FORMAT THE CFM FORCING DATA (CREATES SPINUP)
    # Returns climateTS : dictionary with the various climate fields needed in the correct units
    climateTS, StpsPerYr, depth_S1, depth_S2, grid_bottom, SEBfluxes = (
            RCM.makeSpinFiles(df,timeres=c['DFresample'],Tinterp='mean',spin_date_st = sds, 
            spin_date_end = sde, melt=c['MELT'], desired_depth = None, SEB=c['SEB'], rho_bottom=850))
            # May need to adjust rho_bottom if get an error out of this line with really high
            # accumulation rates (means densities at the bottom of the domain are less than 850)

    # Set start date to clip forcing data
    climateTS['forcing_data_start'] = sds

    # MORE ARGS (straight from Max)
    c['stpsPerYear'] = float('%.2f' % (StpsPerYr))
    c['stpsPerYearSpin'] = float('%.2f' % (StpsPerYr))
    c['grid1bottom'] = float('%.1f' %(depth_S1))
    c['grid2bottom'] = float('%.1f' %(depth_S2))
    c['HbaseSpin'] = float('%.1f' %(3000 - grid_bottom))
    c['DIPhorizon'] = np.floor(0.8*grid_bottom) # firn air content, depth integrated porosity 
    c['keep_firnthickness'] = True
    c['grid_outputs'] = True
    c['grid_output_res'] = 0.05

    # Rerun the spin up?
    NewSpin = args.new_spin 
    c['spinFileName'] = args.spin_name
    c['resultsFileName'] = args.result_name

    # Name configuration file
    CFMconfig = out_fp + args.config_name
    # (Creates temporary file and moves it at the end:
    #       I commented this part out)
    # if os.path.exists(os.path.join(c['resultsFolder'],configName)):
    #     CFMconfig = os.path.join(c['resultsFolder'],configName)
    #     shutil.move(CFMconfig, os.getcwd())
    # else:
    #     CFMconfig = configName

    # Dump configuration file for reproducibility
    if not os.path.exists(out_fp):
        os.mkdir(out_fp)
    with open(CFMconfig,'w') as fp:
        fp.write(json.dumps(c,sort_keys=True, indent=4, separators=(',', ': ')))

    # Create CFM instance by passing config file and forcing data
    firn = fdns.FirnDensityNoSpin(CFMconfig, climateTS = climateTS, NewSpin = NewSpin, SEBfluxes = SEBfluxes)

    # RUN THE MODEL
    firn.time_evolve()

    # Print elapsed time
    telap = (time.time()-tnow)/60
    print('main done, {} minutes'.format(telap))

    # (Moves filepaths: I commented this part out)
    # shutil.move(configName,os.path.join(c['resultsFolder'],configName))

if __name__=='__main__':
    # Define density options that didn't throw an error to test out
    # all_density = ['HLdynamic','Arthern2010S','Arthern2010T', 'Barnola1991',
    #        'Ligtenberg2011','Crocus','KuipersMunneke2015','GSFC2020']
    # Loop through densification options (given in command line)
    for o,option in enumerate(args.physrho):
        # add command line filetag
        if args.tag:
            out_fp += args.tag + '_'

        if 'trace' not in machine:
            out_prefix = f'../../Firn/Output/{args.glacier}{args.site}/'
        else:
            out_prefix = ''
        fp = '/trace/group/rounce/cvwilson/Firn/'
        out_fp = fp + f'Output/{args.glacier}{args.site}/'
        fn_data = fp + f'Forcings/{args.glacier}{args.site}/{args.glacier}{args.site}_1d_forcings.csv'

        print('Beginning',out_fp)
        run_cfm(out_fp, args,fn_data, physRho=option)