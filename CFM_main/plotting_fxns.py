import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import os
import h5py
import cmasher as cmr
from scipy.spatial.distance import euclidean

colors = ['#63c4c7','#fcc02e','#4D559C','#BF1F6A','#60C252',
              '#F77808','#298282','#999999','#FF89B0','#427801']
cmap = plt.get_cmap('Blues')
norm = mpl.colors.Normalize(vmin=0, vmax=20)
dates_wolverine = ['2016_05_13', '2016_06_02', '2016_07_12', '2016_08_16',
                   '2016_09_10', '2017_04_26', '2017_06_16', '2017_07_22', 
                   '2017_09_09', '2018_05_02', '2018_09_08', '2019_05_22', 
                   '2019_09_05', '2020_05_12', '2020_09_09', '2021_04_19', 
                   '2021_08_29', '2022_04_28', '2022_09_18', '2023_05_06', 
                   '2023_09_06', '2024_04_26', '2025_05_09']
dates_wolverine_spring = ['2016_05_13', '2017_04_26', '2018_05_02',
                            '2019_05_22', '2020_05_12', '2021_04_19', 
                            '2022_04_28', '2023_05_06','2024_04_26', 
                            '2025_05_09']
dates_kahiltna = ['2024_05_26', '2024_09_30', '2025_05_23']
dates_kahiltna_spring = ['2024_05_26','2025_05_23']

def to_decimal_year(dt):
    if isinstance(dt, pd.DatetimeIndex):
        dt = pd.Series(dt)
    elif not isinstance(dt, pd.Series):
        dt = pd.Series([dt]) 

    year = dt.dt.year
    start_of_year = pd.to_datetime(year.astype(str) + '-01-01')
    end_of_year = pd.to_datetime((year + 1).astype(str) + '-01-01')
    year_elapsed = (dt - start_of_year).dt.total_seconds()
    year_duration = (end_of_year - start_of_year).dt.total_seconds()
    return year + year_elapsed / year_duration

def from_decimal_year(decimal_years):
    decimal_years = pd.Series(decimal_years)
    years = decimal_years.astype(int)
    start = pd.to_datetime(years.astype(str) + '-01-01')
    end = pd.to_datetime((years + 1).astype(str) + '-01-01')
    fraction = decimal_years - years
    duration = (end - start).dt.total_seconds()
    return start + pd.to_timedelta(fraction * duration, unit='s')

def get_density_measured(site, date):
    # open core file
    glacier = 'wolverine' if site == 'EC' else 'kahiltna' if site == 'KPS' else 'gulkana'
    df = pd.read_csv(f'../Data/cores/{glacier}/{glacier}{site}_{date}.csv')

    # parse layer tops and layer bottoms
    layer_tops = df['SBD'].values - df['length'].values
    layer_bottoms = df['SBD'].values
    density = df['density'].values
    return density, layer_tops, layer_bottoms

def plot_density_measured(density, layer_tops, layer_bottoms, site):
    # get glacier name from site
    glacier = 'wolverine' if site == 'EC' else 'kahiltna' if site == 'KPS' else 'gulkana'

    # make plot
    fig, ax = plt.subplots(figsize=(5,3))

    # plot density as lines between layer bottom and top
    for density, top, bottom in zip(density, layer_tops, layer_bottoms):
        ax.plot([density, density], [top, bottom], color='gray')

    # beautify
    ax.set_ylabel('Depth below surface (m)')
    ax.set_xlabel('Density (kg m$^{-3}$)')
    ax.invert_yaxis()
    ax.set_ylim(max(layer_bottoms), 0)
    ax.tick_params(length=5)
    ax.set_title(f'Firn core data {glacier.capitalize()} {site}')
    plt.show()

def get_var_modeled(cfm_fn, date, var='density'):
    # load output
    output = h5py.File(cfm_fn,'r')

    # plot depth vs density
    all_decimal_time = output[var][:, 0]
    date = '2025-04-20' if date == '2025' else date.replace('_','/')
    target_time = to_decimal_year(pd.to_datetime(date))[0]
    index = np.where(np.abs(all_decimal_time - target_time) < 0.0005)[0]
    if len(index) == 0:
        index = len(output[var][:, 0]) - 1

    # get depth and data arrays
    depth_mod = output['depth'][1:]
    density_mod = output[var][index, 1:]
    if len(density_mod) == 1:
        density_mod = density_mod[0]
    return density_mod, depth_mod

def plot_var_modeled(var_mod, depth):
    # make plot
    fig, ax = plt.subplots(figsize=(5,3))

    # plot
    ax.plot(var_mod, depth, color=colors[0])

    # beautify
    ax.invert_yaxis()
    ax.set_ylabel('Depth below surface (m)')
    ax.set_xlabel('Density (kg m$^{-3}$)')
    ax.tick_params(length=5)
    return fig, ax

def simple_plot(site, measured, modeled, print_error=True, savefig=True, t=None, plot_ax=False):
    density_meas, layer_bottoms, layer_tops = measured
    density_mod, depth_mod = modeled

    # make figure
    if not plot_ax:
        fig, (ax, lax) = plt.subplots(1,2, width_ratios=[2,1], figsize=(5,3)) #,gridspec_kw={'hspace':0.4})
        # dummy legend items
        lax.plot(np.nan, np.nan, color='k',label='Measured', linewidth=3)
        lax.plot(np.nan, np.nan, color=colors[1], label='Modeled',linestyle='--', linewidth=3)
        # turn off label ax
        lax.axis('off')
        # add legend
        lax.legend(fontsize=10, loc='center')
    else:
        ax = plot_ax

    density_mod_interp = []
    for density_meas_layer, top, bottom in zip(density_meas, layer_tops, layer_bottoms):
        layer_idx = np.where((depth_mod >= top) & (depth_mod <= bottom))[0]
        if len(layer_idx) > 0:
            density_mod_layer = np.mean(density_mod[layer_idx])
        else:
            density_mod_layer = np.nan
        density_mod_interp.append(density_mod_layer)
        ax.plot([density_meas_layer, density_meas_layer], [top, bottom], color='k', linewidth=3)

    # plot modeled density
    ax.plot(density_mod, depth_mod, color=colors[1], linestyle='--', alpha=1,linewidth=3)

    # Beautify
    ax.invert_yaxis()
    ax.set_ylim(max(layer_bottoms), 0)
    ax.set_xlim(150, 950)
    ax.tick_params(length=5)

    # Calculate error metrics  
      
    density_mod_interp = np.array(density_mod_interp)
    MAE = np.nanmean(np.abs(density_mod_interp - density_meas))
    ME = np.nanmean(density_mod_interp - density_meas)
    if print_error:   
        print('Mean Absolute Error:',MAE,'kg m-3')
        print('Mean Error (Bias):',ME, 'kg m-3')

    glacier = 'wolverine' if site == 'EC' else 'kahiltna' if site == 'KPS' else 'gulkana'
    if plot_ax:
        f = 1
    elif not t:
        fig.suptitle(f'{glacier.capitalize()} {site}',y=0.95)
    else:
        fig.suptitle(t, y=0.95)
    if savefig:
        plt.savefig(f'{glacier}{site}/{glacier}{site}_firn_core.png',dpi=300,bbox_inches='tight')
    if not plot_ax:
        ax.set_ylabel('Depth below surface (m)')
        ax.set_xlabel('Density (kg m$^{-3}$)')
        plt.show()
    else:
        return ax, MAE, ME

def simple_comparison(site, measured_list, modeled_list, label_list, 
                      print_error=True, savefig=True, t=None,
                      plot_ax=False, color_scheme='qualitative',):
    # make figure
    if not plot_ax:
        longest = max([len(j) for j in label_list])
        ratio = longest / 30
        fig, (ax, lax) = plt.subplots(1,2, width_ratios=[1, ratio], figsize=(5,3)) #,gridspec_kw={'hspace':0.4})
        lax.plot(np.nan, np.nan, color='lightgray',label='Measured', linewidth=2)
    else:
        ax = plot_ax
    idx = np.arange(len(measured_list))

    density_meas, layer_bottoms, layer_tops = measured_list[0]
    layer_middles = layer_tops + (layer_bottoms - layer_tops) / 2
    ax.plot(density_meas, layer_middles, color='lightgray', linewidth=3)
    # for density_meas_layer, top, bottom in zip(density_meas, layer_tops, layer_bottoms):
    #     ax.plot([density_meas_layer, density_meas_layer], [top, bottom], color='lightgray', linewidth=3)

    for i, measured, modeled, label in zip(idx, measured_list, modeled_list, label_list):
        if color_scheme == 'qualitative':
            color = colors[i]
        elif color_scheme == 'continuous':
            cmap = cmr.wildfire
            norm = mpl.colors.Normalize(vmin=0, vmax=len(idx)-1)
            if i == 2:
                color = cmap(norm(1.4))
            if i == 4:
                color = cmap(norm(4.6))
            if i == 1:
                color = cmap(norm(0.7))
            if i == 5:
                color = cmap(norm(5.3))
            else:
                color = cmap(norm(i))
        
        density_mod, depth_mod = modeled

        if not plot_ax:
             # dummy legend items
            lax.plot(np.nan, np.nan, color=color, label=label,linestyle='--', linewidth=2)

        # plot modeled density
        ax.plot(density_mod, depth_mod, color=color, linestyle='--', alpha=1,linewidth=2)

         # Calculate error metrics  
        if print_error:     
            print('======',label,'======')
            density_mod_interp = np.array(density_mod_interp)
            MAE = np.nanmean(np.abs(density_mod_interp - density_meas))
            print(f'Mean Absolute Error: {MAE:.1f} kg m-3')

            ME = np.nanmean(density_mod_interp - density_meas)
            print(f'Mean Error (Bias): {ME:.1f} kg m-3')

    # Beautify
    ax.invert_yaxis()
    ax.set_ylim(max(layer_bottoms), 0)
    ax.set_xlim(150, 950)
    ax.tick_params(length=5)

    if not plot_ax:
        # Turn off label ax
        lax.axis('off')

        # Add legend
        lax.legend(fontsize=10, loc='center')

    # Beautify
    ax.set_ylabel('Depth below surface (m)')
    ax.set_xlabel('Density (kg m$^{-3}$)')
    glacier = 'wolverine' if site == 'EC' else 'kahiltna' if site == 'KPS' else 'gulkana'
    if plot_ax:
        return ax
    elif not t:
        fig.suptitle(f'{glacier.capitalize()} {site} firn core comparison',y=0.95)
    else:
        fig.suptitle(t, y=0.95)
    if savefig:
        plt.savefig(savefig,dpi=300,bbox_inches='tight')
    plt.show()

def compare_sites(measured_list, modeled_list, sites, print_error=True, savefig=True, t=None):
    # make figure
    fig, (ax, lax) = plt.subplots(1,2, width_ratios=[2, 1], figsize=(6,4)) #,gridspec_kw={'hspace':0.4})
    lax.plot(np.nan, np.nan, color='k',label='Measured', linewidth=2)
    idx = np.arange(len(measured_list))

    for i, measured, modeled, site in zip(idx, measured_list, modeled_list, sites):
        density_meas, layer_bottoms, layer_tops = measured
        density_mod, depth_mod = modeled

        density_mod_interp = []
        for density_meas_layer, top, bottom in zip(density_meas, layer_tops, layer_bottoms):
            layer_idx = np.where((depth_mod >= top) & (depth_mod <= bottom))[0]
            if len(layer_idx) > 0:
                density_mod_layer = np.mean(density_mod[layer_idx])
            else:
                density_mod_layer = np.nan
            density_mod_interp.append(density_mod_layer)
            ax.plot([density_meas_layer, density_meas_layer], [top, bottom], color=colors[i], linewidth=3)

        # plot modeled density
        ax.plot(density_mod, depth_mod, color=colors[i], linestyle=':', linewidth=2)
        lax.plot(np.nan, np.nan, color=colors[i], linestyle=':', linewidth=2, label=site)

    # Beautify
    ax.invert_yaxis()
    ax.set_ylim(max(layer_bottoms), 0)
    ax.set_xlim(150, 950)
    ax.tick_params(length=5)

    # Turn off label ax
    lax.axis('off')

    # Add legend
    lax.legend(fontsize=10, loc='center')

    # Beautify
    ax.set_ylabel('Depth below surface (m)')
    ax.set_xlabel('Density (kg m$^{-3}$)')
    if t:
        fig.suptitle(t, y=0.95)
    if savefig:
        plt.savefig(f'compare_sites.png',dpi=300,bbox_inches='tight')
    plt.show()

def compare_site_data(sites,dates='default',t=False,savefig=False):
    # make figure
    fig, (ax, lax) = plt.subplots(1,2, width_ratios=[2, 1], figsize=(6,4)) #,gridspec_kw={'hspace':0.4})
    lax.plot(np.nan, np.nan, color='k',label='Measured', linewidth=2)
    idx = np.arange(len(sites))

    if dates == 'default':
        dates_by_site = []
        for site in sites:
            date = dates_wolverine[-1] if site == 'EC' else dates_kahiltna[-1] if site == 'KPS' else '2025_04_20'
            dates_by_site.append([date])
    else:
        dates_by_site = dates

    all_bottoms = []
    for i, site, dates in zip(idx,sites,dates_by_site):
        for date in dates:
            density_meas, layer_tops, layer_bottoms = get_density_measured(site, date)
            for density_meas_layer, top, bottom in zip(density_meas, layer_tops, layer_bottoms):
                ax.plot([density_meas_layer, density_meas_layer], [top, bottom], color=colors[i], linewidth=3)

            # get max depth
            all_bottoms.append(max(layer_bottoms))
        # plot label
        lax.plot(np.nan, np.nan, color=colors[i], linewidth=3, label=site)

    # Beautify
    ax.invert_yaxis()
    ax.set_ylim(max(all_bottoms), 0)
    ax.set_xlim(150, 950)
    ax.tick_params(length=5)

    # Turn off label ax
    lax.axis('off')

    # Add legend
    lax.legend(fontsize=10, loc='center')

    # Beautify
    ax.set_ylabel('Depth below surface (m)')
    ax.set_xlabel('Density (kg m$^{-3}$)')
    if t:
        fig.suptitle(t, y=0.95)
    if savefig:
        plt.savefig(f'compare_site_data.png',dpi=300,bbox_inches='tight')
    plt.show()

def plot_wolverine_years(output, print_error=True):
    # get dates where there is a core
    fp = '../Data/cores/wolverine/'
    all_wolverine_dates = []
    for f in os.listdir(fp):
        if 'wolverineEC' in f:
            date = f.split('EC')[-1][1:-4]
            all_wolverine_dates.append(date)

    # make figure
    fig, axes = plt.subplots(5,4, figsize=(6,6),sharex=True, sharey=True,gridspec_kw={'hspace':0, 'wspace':0})
    axes = axes.flatten()
    lax = axes[-1]

    # dummy legend items
    lax.plot(np.nan, np.nan, color='k',label='Measured', linewidth=3)
    lax.plot(np.nan, np.nan, color=colors[1], label='Modeled',linestyle='--', linewidth=3)

    # loop through dates and plot each date
    max_bottom = []
    for d,date in enumerate(all_wolverine_dates):
        ax = axes[d]

        # load data for this date
        df = pd.read_csv(f'../Data/cores/wolverine/wolverineEC_{date}.csv')
        layer_tops = df['SBD'].values - df['length'].values
        layer_bottoms = df['SBD'].values

        # get measured density array
        density_meas = df['density'].values

        # plot density vs depth
        var = 'density'

        # find index of a given time step
        all_decimal_time = output[var][:, 0]
        target_time = to_decimal_year(pd.to_datetime(date.replace('_','/')))[0]
        index = np.argmin(np.abs(all_decimal_time - target_time))

        # get depth and data arrays
        depth = output['depth'][1:]
        density_mod = output[var][index, 1:]
        if len(density_mod) == 1:
            density_mod = density_mod[0]

        # average the modeled density between the depths of the pit
        density_mod_interp = []
        for density_meas_layer, top, bottom in zip(density_meas, layer_tops, layer_bottoms):
            layer_idx = np.where((depth >= top) & (depth <= bottom))[0]
            if len(layer_idx) > 0:
                density_mod_layer = np.mean(density_mod[layer_idx])
            else:
                density_mod_layer = np.nan
            density_mod_interp.append(density_mod_layer)
            ax.plot([density_meas_layer, density_meas_layer], [top, bottom], color='k', linewidth=3)
            # ax.plot([density_mod_layer, density_mod_layer], [top, bottom], color=colors[r], linewidth=3)
        # lax.plot(np.nan, np.nan, color=cmap(norm(d)), linewidth=3, label=date.replace('_','/'))
        ax.plot(density_mod, depth, color=colors[1], linestyle='--', alpha=1,linewidth=3)
        ax.text(170, 26, date[5:7]+'/'+date[:4])

        # Beautify
        max_bottom.append(np.max(layer_bottoms))
        ax.invert_yaxis()
        ax.set_ylim(np.max(max_bottom), 0)
        ax.set_xlim(150, 950)
        ax.tick_params(length=5)

        # Calculate error metrics  
        if print_error:     
            density_mod_interp = np.array(density_mod_interp)
            MAE = np.nanmean(np.abs(density_mod_interp - density_meas))
            print(date)
            print('         Mean Absolute Error:',MAE,'kg m-3')

            ME = np.nanmean(density_mod_interp - density_meas)
            print('         Mean Error (Bias):',ME, 'kg m-3')

    # Turn off label ax
    lax.axis('off')

    # Add legend
    lax.legend(fontsize=10,loc='center')

    # Beautify
    fig.supylabel('Depth below surface (m)')
    fig.supxlabel('Density (kg m$^{-3}$)')
    fig.suptitle(f'Wolverine EC firn core comparison',y=0.95)
    plt.savefig(f'wolverineEC/wolverineEC_firn_core_all.png',dpi=300,bbox_inches='tight')
    plt.show()

def plot_years_together(output, site, print_error=True, every=1):
    var = 'density'
    # get dates where there is a core
    glacier = 'wolverine' if site == 'EC' else 'kahiltna' if site == 'KPS' else 'gulkana'
    fp = f'../Data/cores/{glacier}/'
    all_dates = []
    avg_depths = np.arange(0, 25.5, 0.5)
    all_density = []
    for f in os.listdir(fp):
        if glacier+site in f:
            date = f.split(site)[-1][1:-4]
            all_dates.append(date)
            df = pd.read_csv(fp + f)
            layer_middle = df['SBD'].values - df['length'].values / 2
            dens_middle = df['density'].values
            dens_interp = np.interp(avg_depths, layer_middle, dens_middle)
            all_density.append(dens_interp)
    avg_density = np.mean(all_density, axis=0)
    all_dates = all_dates[::every]

    # make figure
    fig, (ax1, ax2, ax3, lax) = plt.subplots(1, 4, figsize=(8, 4),
                                        width_ratios=[1,1,1,1],
                                        sharey=True,
                                        gridspec_kw={'hspace':0, 'wspace':0})
    ax1.set_title('Mean')
    ax2.set_title('Measured\nAnomoly')
    ax3.set_title('Modeled\nAnomoly')

    # colormap
    cmap = plt.get_cmap('plasma')
    norm = mpl.colors.Normalize(vmin=0, vmax=len(all_dates)-1)

    # dummy legend items
    # lax.plot(np.nan, np.nan, color='gray',label='Measured', linewidth=2)
    # lax.plot(np.nan, np.nan, color='gray', label='Modeled',linestyle='--', linewidth=2)

    # loop through dates and plot each date
    for d,date in enumerate(all_dates):
        # color
        c = cmap(norm(d))

        # load data for this date
        df = pd.read_csv(f'../Data/cores/{glacier}/{glacier}{site}_{date}.csv')
        layer_middle = df['SBD'].values - df['length'].values / 2
        density_meas = df['density'].values

        # find index of a given time step
        all_decimal_time = output[var][:, 0]
        target_time = to_decimal_year(pd.to_datetime(date.replace('_','/')))[0]
        index = np.argmin(np.abs(all_decimal_time - target_time))

        # get depth and data arrays
        depth_mod = output['depth'][1:]
        density_mod = output[var][index, 1:]
        if len(density_mod) == 1:
            density_mod = density_mod[0]

        # average the modeled density between the depths of the pit
        density_meas_interp = np.interp(avg_depths, layer_middle, density_meas)
        density_mod_interp = np.interp(avg_depths, depth_mod, density_mod)
        ax2.plot(density_meas_interp - avg_density, avg_depths, color=c)
        ax3.plot(density_mod_interp - avg_density, avg_depths, color=c)
        lax.plot(np.nan, np.nan, color=c, linewidth=2, label=date.replace('_','/'))
        ax1.plot(density_meas, layer_middle, color=c, linewidth=1)
        # ax.text(170, 26, date[5:7]+'/'+date[:4])

        # Beautify
        for ax in [ax1, ax2, ax3]:
            ax.invert_yaxis()
            max_depth = 25 if site == 'EC' else 15
            ax.set_ylim(max_depth, 0)
            ax.tick_params(length=5)
        ax1.set_xlim(150, 950)
        for ax in [ax2, ax3]:
            ax.set_xlim(-300, 300)
            ax.axvline(0, linewidth=0.5, color='k')

        # Calculate error metrics  
        if print_error:     
            density_mod_interp = np.array(density_mod_interp)
            MAE = np.nanmean(np.abs(density_mod_interp - density_meas))
            print(date)
            print('         Mean Absolute Error:',MAE,'kg m-3')

            ME = np.nanmean(density_mod_interp - density_meas)
            print('         Mean Error (Bias):',ME, 'kg m-3')

    # Plot mean on top
    ax1.plot(avg_density, avg_depths, color='k')

    # Turn off label ax
    lax.axis('off')

    # Add legend
    lax.legend(ncols=1, fontsize=10,loc='center')

    # Beautify
    fig.supylabel('Depth below surface (m)')
    fig.supxlabel('Density (kg m$^{-3}$)', y=-0.03)
    # fig.suptitle(f'Wolverine EC firn core comparison',y=1)
    plt.savefig(f'{glacier}{site}/{glacier}{site}_firn_core_together.png',dpi=300,bbox_inches='tight')
    plt.show()

def compare_densification(fn, all_rho, date, measured, print_error=True):
    # parse input
    density_meas, layer_bottoms, layer_tops = measured 

    # create figure
    if len(all_rho) > 3:
        fig, (ax, lax) = plt.subplots(1, 2, width_ratios=(2,1), figsize=(5,5))
        # Turn off label ax
        lax.axis('off')
        legend_loc = 'center'
    else:
        fig, ax = plt.subplots(figsize=(3,5))
        lax = ax
        legend_loc = 'best'

    # loop through densification options
    for r,rho in enumerate(all_rho):
        output = h5py.File(fn.replace('RHO', rho),'r')

        # find index of a given time step
        all_decimal_time = output['density'][:, 0]
        target_time = to_decimal_year(pd.to_datetime(date.replace('_','/')))[0]
        index = np.argmin(np.abs(all_decimal_time - target_time))

        # get depth and data arrays
        depth = output['depth'][1:]
        density_mod = output['density'][index, 1:]
        if len(density_mod) == 1:
            density_mod = density_mod[0]

        # average the modeled density between the depths of the pit
        density_mod_interp = []
        for density_meas_layer, top, bottom in zip(density_meas, layer_tops, layer_bottoms):
            layer_idx = np.where((depth >= top) & (depth <= bottom))[0]
            if len(layer_idx) > 0:
                density_mod_layer = np.mean(density_mod[layer_idx])
            else:
                density_mod_layer = np.nan
            density_mod_interp.append(density_mod_layer)
            ax.plot([density_meas_layer, density_meas_layer], [top, bottom], color='k', linewidth=3)
            # ax.plot([density_mod_layer, density_mod_layer], [top, bottom], color=colors[r], linewidth=3)
        lax.plot(np.nan, np.nan, color=colors[r], linewidth=3, label=rho)
        ax.plot(density_mod, depth, color=colors[r], linestyle='--', alpha=0.999)

        # Calculate error metrics       
        density_mod_interp = np.array(density_mod_interp)
        MAE = np.nanmean(np.abs(density_mod_interp - density_meas))
        if print_error:
            print(rho)
            print('         Mean Absolute Error:',MAE,'kg m-3')

            ME = np.nanmean(density_mod_interp - density_meas)
            print('         Mean Error (Bias):',ME, 'kg m-3')

    # Add legend
    lax.legend(loc=legend_loc)

    # Beautify
    ax.invert_yaxis()
    ax.set_ylim(max(layer_bottoms), 0)
    ax.set_ylabel('Depth below surface (m)')
    ax.set_xlabel('Density (kg m$^{-3}$)')
    ax.tick_params(length=5)
    return fig, ax

def profile_permutation_test(wolverine_profiles, site_profile, n_permutations=10000):
    # Stack Wolverine profiles into matrix
    W = np.vstack(wolverine_profiles)  # shape: (n_cores, n_depths)
    site = site_profile  # shape: (n_depths,)
    
    # Compute observed distance from site to Wolverine mean profile
    W_mean = np.mean(W, axis=0)
    observed_dist = euclidean(site, W_mean)
    
    # Combine all profiles
    all_profiles = np.vstack([W, site])
    n_total = all_profiles.shape[0]
    
    # Permutation test
    perm_dists = []
    for _ in range(n_permutations):
        permuted = np.random.permutation(n_total)
        group1 = all_profiles[permuted[:len(W)]]
        group2 = all_profiles[permuted[len(W):]]
        dist = euclidean(np.mean(group1, axis=0), np.mean(group2, axis=0))
        perm_dists.append(dist)
    
    p_value = np.mean(np.array(perm_dists) >= observed_dist)
    return p_value, observed_dist, perm_dists

def permutation_test(wolverine_data, site_data, n_permutations=10000):
    p_values = []
    for i in range(len(site_data)):
        w = [core[i] for core in wolverine_data]
        s = [site_data[i]]
        combined = np.array(w + s)
        observed_diff = np.abs(np.mean(s) - np.mean(w))

        diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            w_sample = combined[:len(w)]
            s_sample = combined[len(w):]
            diffs.append(np.abs(np.mean(s_sample) - np.mean(w_sample)))

        p = np.mean(np.array(diffs) >= observed_diff)
        p_values.append(p)
    return np.array(p_values)

def plot_permutation_test(wolverine_cores, site_cores, sites, 
                          depths, n_permutations=10000):
    fig, axes = plt.subplots(3,2, sharex='col')
    for s in range(len(sites)):
        for i in range(len(wolverine_cores)):
            axes[s,0].plot(wolverine_cores[i], depths, color='gray')
    axes[0,0].plot(np.nan, np.nan, color='gray', label='EC')
    for s,site in enumerate(sites):
        glacier = 'Wolverine' if site == 'EC' else 'Kahiltna' if site == 'KPS' else 'Gulkana'
        p_vals = permutation_test(wolverine_cores, site_cores[site], 
                                  n_permutations=n_permutations)
        axes[s, 0].plot(site_cores[site], depths, color=colors[s+1])
        axes[s, 1].plot(p_vals, depths, color=colors[s+1])
        axes[s, 1].axvline(0.05, color='red', linestyle='--', label='p = 0.05')
        axes[s, 1].set_ylabel(glacier + ' '+site)
        axes[s, 1].yaxis.set_label_position('right')
        axes[s, 1].set_xlim(0, 1)
    axes = axes.flatten()
    for ax in axes:
        ax.invert_yaxis()
    fig.supxlabel('p-value')
    fig.supylabel('Depth (m)')
    fig.suptitle(f'Permutation Test: Site {site} vs Wolverine')
    axes[0].legend()
    axes[-1].legend()
    plt.show()