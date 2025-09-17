import cfm_claire as sim
import argparse

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

failed =[]
fp_forcings = '../../Firn/Forcings/'
base_out_fp = '../../Firn/Output/'
for site in ['T','Z','EC','KPS']:
    args.site = site
    args.glacier = 'wolverine' if site == 'EC' else 'kahiltna' if site == 'KPS' else 'gulkana'

    for temp_change in [-5,-2,-1,0,1,2,5]:
        temp_change_str = '+'+str(temp_change) if temp_change > 0 else str(temp_change)
        fn_data = fp_forcings
        if temp_change != 0:
            fn_data += f'{args.glacier}/{args.glacier}{args.site}_{temp_change_str}C_{args.dt}_forcings.csv'
        else:
            fn_data += f'{args.glacier}/{args.glacier}{args.site}_{args.dt}_forcings.csv'
        # base filepath
        out_fp = base_out_fp
        out_fp += args.glacier + args.site + '_' + temp_change_str + '_0/'

        print('Beginning',out_fp,'with',fn_data)
        try:
            sim.run_cfm(out_fp, args, forcing_fn = fn_data)
        except:
            failed.append(out_fp)
            print(out_fp, 'failed')

    for temp_change in [-5,-2,-1,0,1,2,5]:
        temp_change_str = '+'+str(temp_change) if temp_change > 0 else str(temp_change)
        fn_data = fp_forcings
        if temp_change != 0:
            fn_data += f'{args.glacier}/{args.glacier}{args.site}_{temp_change_str}C_sameacc_{args.dt}_forcings.csv'
        else:
            fn_data += f'{args.glacier}/{args.glacier}{args.site}_{args.dt}_forcings.csv'
        # base filepath
        out_fp = base_out_fp
        out_fp += args.glacier + args.site + '_' + temp_change_str + '_sameacc_0/'

        print('Beginning',out_fp,'with',fn_data)
        try:
            sim.run_cfm(out_fp, args, forcing_fn = fn_data)
        except:
            failed.append(out_fp)
            print(out_fp, 'failed')

    for precip_change in [0.5, 0.667, 0.9, 1, 1.1, 1.5, 2]:
        if precip_change != 1:
            fn_data = f'../Data/{args.glacier}{args.site}_{precip_change}_{args.dt}_forcings.csv'
        else:
            fn_data = f'../Data/{args.glacier}{args.site}_{args.dt}_forcings.csv'
        out_fp = base_out_fp
        out_fp += args.glacier + args.site + '_' + str(precip_change) + '_0/'

        print('Beginning',out_fp,'with',fn_data)
        try:
            sim.run_cfm(out_fp, args, forcing_fn = fn_data)
        except:
            failed.append(out_fp)
            print(out_fp, 'failed')

print(failed)