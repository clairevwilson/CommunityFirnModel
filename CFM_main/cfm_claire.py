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

# parse command-line args
parser = argparse.ArgumentParser()
parser.add_argument('-n','--new_spin',action='store',type=bool,default=True)
parser.add_argument('-site',action='store',type=str,default='Z')
parser.add_argument('-dt',action='store',type=str,default='1d')
parser.add_argument('-tag',action='store',type=str,default=None)
parser.add_argument('-physrho',nargs='+',default=['GSFC2020'],
                    help='Provide one or more densification schemes')
parser.add_argument('-input_srho',default=0,
                    help='0 for default, 1 for variable, any number for other constant (e.g. 250)')
args = parser.parse_args()

# check if inputting surface density
args.glacier = 'wolverine' if args.site == 'EC' else 'kahiltna' if args.site == 'KPS' else 'gulkana'
if type(args.tag) == str and 'pygem' in args.tag:
    forcing_fn = f'../../Firn/Forcings/{args.glacier}/{args.glacier}{args.site}_pygem_forcings.csv'
    args.dt = '1MS'
elif int(args.input_srho) == 1:
    forcing_fn = f'../../Firn/Forcings/{args.glacier}/{args.glacier}{args.site}_{args.dt}_forcings_wsrho.csv'
else:
    forcing_fn = f'../../Firn/Forcings/{args.glacier}/{args.glacier}{args.site}_{args.dt}_forcings.csv'

def run_cfm(out_fp, args, forcing_fn, physRho='GSFC2020'):
    # filenames of configs and data
    glacier = 'wolverine' if args.site == 'EC' else 'kahiltna' if args.site == 'KPS' else 'gulkana'
    json_fn = 'my_configs.json'

    # get configs file
    with open(json_fn) as file:
        c = json.load(file)

    # read forcings
    if os.path.exists(forcing_fn):
        # read already generated forcings
        print(forcing_fn)
        df = pd.read_csv(forcing_fn,parse_dates=True,index_col=0)

        # clip to start of year
        df = df['1981':].copy()

        # correct the accumulation according to the ratio:
        # kp calculated from regression / kp used in simulation
        # if args.site == 'T':
        #     df['BDOT'] *= 3.665 / 3.5
        # elif args.site == 'Z':
        #     df['BDOT'] *= 3.774 / 3.5
        # elif args.site == 'EC':
        #     df['BDOT'] *= 1.650 / 1.75
        # elif args.site == 'KPS':
        #     df['BDOT'] *= 2.470 / 2

    else:
        print('forcing file not found: generate and save to', forcing_fn)
        assert 1==0

    # start timer
    tnow = time.time()

    # define spin dates
    sds = 1981.0    # spin date start
    sde = 1995.0    # spin date end

    c['physRho'] = physRho
    c['DFresample'] = args.dt
    c['SEB'] = False # surface energy balance module OFF
    c['MELT'] = True # melt module ON
    c['rain'] = True # rainfall ON

    '''
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

    ### surface density (fixed or variable)
    variable_srho = args.input_srho
    if variable_srho == True:
        c['variable_srho'] = True
        c['srho_type'] = 'userinput'
    else:
        if int(variable_srho) not in [0,1]:
            option_rho = float(variable_srho)
        else:
            option_rho_dict = {'Z':366, 'T':347, 'EC':427, 'KPS':417} #
            option_rho = option_rho_dict[args.site]
        c['rhos0'] = option_rho 

    # path (within CFM_main that the results will be stored in)
    c['resultsFolder'] = out_fp 

    ### format the CFM forcing data (including creating the spin up)
    ### climateTS is a dictionary with the various climate fields needed, in the correct units.
    climateTS, StpsPerYr, depth_S1, depth_S2, grid_bottom, SEBfluxes = (
        RCM.makeSpinFiles(df,timeres=c['DFresample'],Tinterp='mean',spin_date_st = sds, 
        spin_date_end = sde, melt=c['MELT'], desired_depth = None, SEB=c['SEB'], rho_bottom=850))

    # clip forcing data
    climateTS['forcing_data_start'] = sds

    # bunch of arguments straight from the run_CFM_Claire_notebook
    c['stpsPerYear'] = float('%.2f' % (StpsPerYr))
    c['stpsPerYearSpin'] = float('%.2f' % (StpsPerYr))
    c['grid1bottom'] = float('%.1f' %(depth_S1))
    c['grid2bottom'] = float('%.1f' %(depth_S2))
    c['HbaseSpin'] = float('%.1f' %(3000 - grid_bottom))
    c['DIPhorizon'] = np.floor(0.8*grid_bottom) # firn air content, depth integrated porosity 
    c['keep_firnthickness'] = True
    c['grid_outputs'] = True
    c['grid_output_res'] = 0.05

    # name configuration file
    CFMconfig = out_fp + 'CFMconfig.json'
    # if os.path.exists(os.path.join(c['resultsFolder'],configName)):
    #     CFMconfig = os.path.join(c['resultsFolder'],configName)
    #     shutil.move(CFMconfig, os.getcwd())
    # else:
    #     CFMconfig = configName

    # dump configuration file for reproducibility
    if not os.path.exists(out_fp):
        os.mkdir(out_fp)
    with open(CFMconfig,'w') as fp:
        fp.write(json.dumps(c,sort_keys=True, indent=4, separators=(',', ': ')))

    # rerun the spin up each time if True
    NewSpin = args.new_spin 

    # create CFM instance by passing config file and forcing data
    firn = fdns.FirnDensityNoSpin(CFMconfig, climateTS = climateTS, NewSpin = NewSpin, SEBfluxes = SEBfluxes)

    # run the model
    firn.time_evolve()

    # print time elapsed    
    telap = (time.time()-tnow)/60
    print('main done, {} minutes'.format(telap))

    # store output
    # shutil.move(configName,os.path.join(c['resultsFolder'],configName))

if __name__=='__main__':
    # all_density = ['HLdynamic','Arthern2010S','Arthern2010T', 'Barnola1991',
    #        'Ligtenberg2011','Crocus','KuipersMunneke2015','GSFC2020']
    for o,option in enumerate(args.physrho):
        # add command line filetag
        if args.tag:
            out_fp += args.tag + '_'

        if 'trace' not in machine:
            out_prefix = f'../../Firn/Output/{args.glacier}{args.site}/'
        else:
            out_prefix = ''
        fp = '/trace/group/rounce/cvwilson/Firn/'
        out_fp = fp + f'Output/{args.glacier}{args.site}/{args.glacier}{args.site}_{option}_0/'
        fn_data = fp + f'Forcings/{args.glacier}{args.site}/{args.glacier}{args.site}_1d_forcings.csv'

        print('Beginning',out_fp)
        # try:
        run_cfm(out_fp, args,fn_data, physRho=option)
        # except:
        #     print(out_fp, 'failed')