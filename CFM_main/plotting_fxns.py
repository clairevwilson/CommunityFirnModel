import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import numpy as np
import xarray as xr
import os
import h5py
import cmasher as cmr
from scipy.spatial.distance import euclidean
import socket
if 'trace' in socket.gethostname():
    base_fp = '/trace/group/rounce/cvwilson/Firn/'

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
dates_KPS = ['2024_05_26', '2024_09_30', '2025_05_23']
dates_KPS_spring = ['2024_05_26','2025_05_23']
dates_KQU = ['2024_05_25', '2025_05_23']
site_elevation = {'Z':2081, 'T':1877, 'EC':1348, 'KQU':2630, 'KPS':3053, 'KT1':2690,'KT2':2846,'KT3':2900}
markers = {'EC':'*', 'Z':'s', 'T':'o', 'KPS':'^','KQU':'v','KT1':'1','KT2':'3','KT3':'2'}
site_colors = {'EC':colors[0],'T':colors[1],'Z':colors[2],'KPS':colors[3],'KQU':colors[4],
               'KT1':colors[5], 'KT2':colors[6], 'KT3':colors[7]}

def div_colors(i):
    cmap = cmr.iceburn
    norm = mpl.colors.Normalize(vmin=0, vmax=6)
    
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
    return color

def get_glacier(site):
    if site == 'EC':
        return 'wolverine'
    elif site in ['T','Z']:
        return 'gulkana'
    elif 'K' in site:
        return 'kahiltna'
    
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
    glacier = get_glacier(site)
    df = pd.read_csv(f'../Data/cores/{glacier}/{glacier}{site}_{date}.csv')

    # parse layer tops and layer bottoms
    layer_tops = df['SBD'].values - df['length'].values
    layer_bottoms = df['SBD'].values
    density = df['density'].values
    return density, layer_tops, layer_bottoms

def plot_density_measured(density, layer_tops, layer_bottoms, site):
    # get glacier name from site
    glacier = get_glacier(site)

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

def simple_plot(site, measured, modeled, print_error=True, savefig=True, 
                t=None, plot_ax=False, ylim=False):
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
    if not ylim:
        ax.set_ylim(max(layer_bottoms), 0)
    else:
        ax.set_ylim(ylim, 0)
    if np.any(density_mod > 200):
        ax.set_xlim(150, 950)
    ax.tick_params(length=5)

    # Calculate error metrics  
      
    density_mod_interp = np.array(density_mod_interp)
    MAE = np.nanmean(np.abs(density_mod_interp - density_meas))
    ME = np.nanmean(density_mod_interp - density_meas)
    if print_error:   
        print('Mean Absolute Error:',MAE,'kg m-3')
        print('Mean Error (Bias):',ME, 'kg m-3')

    glacier = get_glacier(site)
    if plot_ax:
        f = 1
    elif not t:
        fig.suptitle(f'{glacier.capitalize()} {site}',y=0.95)
    else:
        fig.suptitle(t, y=0.95)
    if savefig:
        plt.savefig(base_fp + f'Figs/{glacier}{site}_firn_core.png',dpi=300,bbox_inches='tight')
    if not plot_ax:
        ax.set_ylabel('Depth below surface (m)')
        ax.set_xlabel('Density (kg m$^{-3}$)')
        plt.show()
    else:
        return ax, MAE, ME

def simple_comparison(site, measured_list, modeled_list, label_list, 
                      print_error=True, savefig=False, t=None,
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
            color = div_colors(i)
        
        density_mod, depth_mod = modeled

        if not plot_ax:
             # dummy legend items
            lax.plot(np.nan, np.nan, color=color, label=label, linewidth=1)

        # plot modeled density
        ax.plot(density_mod, depth_mod, color=color, linewidth=1)

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
    glacier = get_glacier(site)
    if plot_ax:
        return ax
    elif not t:
        fig.suptitle(f'{glacier.capitalize()} {site} firn core comparison',y=0.95)
    else:
        fig.suptitle(t, y=0.95)
    if savefig:
        plt.savefig(base_fp + 'Figs/' + savefig,dpi=300,bbox_inches='tight')
    plt.show()

def compare_sites(measured_list, modeled_list, sites, print_error=True, savefig=True, t=None):
    # make figure
    fig, (ax, lax) = plt.subplots(1,2, width_ratios=[2, 1], figsize=(6,4)) #,gridspec_kw={'hspace':0.4})
    lax.plot(np.nan, np.nan, color='k',label='Measured', linewidth=2)
    idx = np.arange(len(measured_list))

    bottoms = []
    for i, measured, modeled, site in zip(idx, measured_list, modeled_list, sites):
        density_meas, layer_bottoms, layer_tops = measured
        density_mod, depth_mod = modeled

        density_mod_interp = []
        depth_plot = []
        density_plot = []
        for density_meas_layer, top, bottom in zip(density_meas, layer_tops, layer_bottoms):
            layer_idx = np.where((depth_mod >= top) & (depth_mod <= bottom))[0]
            if len(layer_idx) > 0:
                density_mod_layer = np.mean(density_mod[layer_idx])
            else:
                density_mod_layer = np.nan
            density_mod_interp.append(density_mod_layer)
            depth_plot.append([top, bottom])
            density_plot.append([density_meas_layer, density_meas_layer])
        ax.plot(np.array(density_plot).flatten(), np.array(depth_plot).flatten(), color=colors[i], linewidth=2)

        # plot modeled density
        ax.plot(density_mod, depth_mod, color=colors[i], linestyle=':', linewidth=2)
        lax.plot(np.nan, np.nan, color=colors[i], linestyle=':', linewidth=2, label=site)
        bottoms.append(max(layer_bottoms))

    # Beautify
    ax.invert_yaxis()
    ax.set_ylim(max(bottoms), 0)
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
        plt.savefig(base_fp + f'Figs/compare_sites.png',dpi=300,bbox_inches='tight')
    plt.show()

def compare_site_data(sites,dates='default',t=False,savefig=False):
    # make figure
    fig, (ax, lax) = plt.subplots(1,2, width_ratios=[2, 1], figsize=(6,4)) #,gridspec_kw={'hspace':0.4})
    lax.plot(np.nan, np.nan, color='k',label='Measured (2025)', linewidth=2)
    lax.plot(np.nan, np.nan, color='k', alpha=0.3, linewidth=2, label='Measured (past years)')
    idx = np.arange(len(sites))

    if dates == 'default':
        dates_by_site = []
        for site in sites:
            if site == 'EC':
                date = dates_wolverine[-1]
            elif site == 'KPS':
                date = dates_KPS[-1]
            elif site == 'KQU':
                date = dates_KQU[-1] 
            else:
                date = '2025_04_20'
            dates_by_site.append([date])
    else:
        dates_by_site = dates

    all_bottoms = []
    for i, site, dates in zip(idx,sites,dates_by_site):
        alphas = [0.3] * (len(dates) - 1) + [1]
        for d, date in enumerate(dates):
            density_meas, layer_tops, layer_bottoms = get_density_measured(site, date)
            depth_plot = []
            density_plot = []
            for density_meas_layer, top, bottom in zip(density_meas, layer_tops, layer_bottoms):
                # ax.plot([density_meas_layer, density_meas_layer], [top, bottom], color=colors[i], linewidth=3)
                depth_plot.append([top, bottom])
                density_plot.append([density_meas_layer, density_meas_layer])
            ax.plot(np.array(density_plot).flatten(), np.array(depth_plot).flatten(), color=colors[i], linewidth=2, alpha=alphas[d])

            # get max depth
            all_bottoms.append(max(layer_bottoms))
        # plot label
        lax.plot(np.nan, np.nan, color=colors[i], linewidth=2, label=site)

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
        plt.savefig(base_fp + 'Figs/compare_site_data.png',dpi=300,bbox_inches='tight')
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
    fig.suptitle('Wolverine EC firn core comparison',y=0.95)
    plt.savefig(base_fp + 'Figs/wolverineEC_firn_core_all.png',dpi=300,bbox_inches='tight')
    plt.show()

def plot_years_together(all_output, site, print_error=True, every=1, 
                        dates='all',savefig=False, labels=[]):
    var = 'density'
    # get dates where there is a core
    glacier = get_glacier(site)
    fp = f'../Data/cores/{glacier}/'
    if glacier == 'wolverine':
        all_dates = dates_wolverine if dates == 'all' else dates_wolverine_spring
    elif glacier == 'kahiltna':
        all_dates = dates_KPS if dates == 'all' else dates_KPS_spring
    snow_df = pd.read_csv(f'../Data/cores/{glacier}/{glacier}{site}_snowdepth.csv')
    avg_depths = np.arange(0, 25.5, 0.5)
    all_density = []
    for date in all_dates:
        # get depth of seasonal snow
        df = pd.read_csv(fp + f'{glacier}{site}_{date}.csv')
        layer_middle = df['SBD'].values - df['length'].values / 2
        dens_middle = df['density'].values
        dens_interp = np.interp(avg_depths, layer_middle, dens_middle)
        all_density.append(dens_interp)
    avg_density = np.mean(all_density, axis=0)
    all_dates = all_dates[::every]

    # make figure
    # fig, axes = plt.subplots(len(all_dates), 2, figsize=(4, 8),
    #                                     sharey=True,
    #                                     gridspec_kw={'hspace':0, 'wspace':0})
    fig = plt.figure(figsize=(4, len(all_dates)))
    gs = mpl.gridspec.GridSpec(len(all_dates), 2, figure=fig)
    gs.update(hspace=0, wspace=0) 
    # lax = fig.add_subplot(gs[0, :])  # This spans both columns
    # Create the rest of the subplots
    axes = []
    for i in range(0, len(all_dates)):
        axes.append([])
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            axes[-1].append(ax)
    axes = np.array(axes)
    axes[0, 0].set_title('Data')
    axes[0, 1].set_title('Anomoly')
    # axes[0, 2].set_title('Modeled\nAnomoly')

    # colormap
    # cmap = plt.get_cmap('plasma')
    # cmap = cmr.jungle
    # norm = mpl.colors.Normalize(vmin=-5, vmax=len(all_dates)+2)

    # dummy legend items
    # lax.plot(np.nan, np.nan, color='gray',label='Measured', linewidth=2)
    # lax.plot(np.nan, np.nan, color='gray', label='Modeled',linestyle='--', linewidth=2)

    for i, output in enumerate(all_output):
        # loop through dates and plot each date
        for d,date in enumerate(all_dates):
            ax1, ax2 = axes[d]
            # color
            # c = cmap(norm(d))
            cmap = cmr.iceburn
            norm = mpl.colors.Normalize(vmin=0, vmax=len(all_output)-1)
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

            min_depth = snow_df.loc[snow_df['date'] == date, 'snowdepth'].values[0]

            # Plot mean on bottom
            ax1.plot(avg_density, avg_depths, color='lightgray',label='Mean')

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
            avg_density_plot = avg_density[avg_depths >= min_depth]
            avg_depths_plot = avg_depths[avg_depths >= min_depth]
            density_meas_interp = np.interp(avg_depths_plot, layer_middle, density_meas)
            density_mod_interp = np.interp(avg_depths_plot, depth_mod, density_mod)
            # ax2.plot(density_meas_interp - avg_density_plot, avg_depths_plot, color='k', linestyle=':')
            ax2.plot(density_mod_interp - density_meas_interp, avg_depths_plot, color=color)
            # lax.plot(np.nan, np.nan, color=c, linewidth=2, label=date.replace('_','/'))
            ax1.plot(density_meas, layer_middle, color='k', linestyle='--', label='Measured')
            # ax1.plot(density_mod_interp, avg_depths_plot, color=color, label='Modeled')
            # ax.text(170, 26, date[5:7]+'/'+date[:4])

            # Beautify
            for ax in [ax1, ax2]:
                ax.invert_yaxis()
                max_depth = 26 if site == 'EC' else 15
                tick2 = 20 if site == 'EC' else 10
                ax.set_ylim(max_depth, 0)
                ax.tick_params(length=5)
                ax.set_yticks([0, tick2])
            ax1.set_xlim(150, 950)
            for ax in [ax2]:
                ax.set_xlim(-300, 300)
                ax.axvline(0, linewidth=0.5, color='k')
                ax.set_yticks([])
                ax.set_yticklabels([])

            # Label row with the date
            ax2.set_ylabel(date[:4]) # .replace('_','/'))
            ax2.yaxis.set_label_position('right')

        # Calculate error metrics  
        if print_error:     
            density_mod_interp = np.array(density_mod_interp)
            MAE = np.nanmean(np.abs(density_mod_interp - density_meas))
            print(date)
            print('         Mean Absolute Error:',MAE,'kg m-3')

            ME = np.nanmean(density_mod_interp - density_meas)
            print('         Mean Error (Bias):',ME, 'kg m-3')

    # axes[0,0].legend()
    lax = fig.add_axes((1.1, 0.4, 0.2, 0.3))
    # lax.plot(np.nan, np.nan, color='gray', label='Modeled')
    lax.plot(np.nan, np.nan, color='k', linestyle='--',label='Measured')
    lax.plot(np.nan, np.nan, color='lightgray', label='Mean measured')
    if len(labels) > 0:
        for i in range(len(all_output)):
            cmap = cmr.iceburn
            norm = mpl.colors.Normalize(vmin=0, vmax=len(all_output)-1)
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
            lax.plot(np.nan, np.nan, color=color, label=labels[i])

    # Turn off label ax
    lax.axis('off')

    # Add legend
    lax.legend(ncols=1, fontsize=10,loc='upper center')

    # Beautify
    fig.supylabel('Depth below surface (m)',x=-0.01)
    fig.supxlabel('Density (kg m$^{-3}$)')
    # fig.suptitle(f'Wolverine EC firn core comparison',y=1)
    if savefig:
        plt.savefig(base_fp+'Figs/'+savefig,dpi=300,bbox_inches='tight')
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

def get_dict(sites, output_fn, forcing_fn):
    output_dict = {}
    output_vars = ['refreeze','refreeze_ratio','runoff','density_gradient','DIP','temperature','firn_depth']
    all_vars = output_vars + ['SMELT','BDOT','MELT_ACC_RATIO','RAIN','WATER','ACC_MELT_RATIO','STEMP']
    for site in sites:
        print('starting',site)
        glacier = get_glacier(site)

        # LOAD OUTPUT VARS
        if 'K' in site:
            output_fn = base_fp + 'Output/SITE/SITE_kahiltnatest/CFMresults.hdf5'
            forcing_fn = base_fp + 'Forcings/SITE/SITE_1d_kahiltnatest.csv'
        list_all = ['']
        if site in ['T','Z','KPS','EC']:
            output_fn = base_fp + 'Output/SITE/SITE_CHANGE/CFMresults.hdf5'
            forcing_fn = base_fp + 'Forcings/SITE/SITE_1d_CHANGE_forcings.csv'
            list_all = ['temp'+str(v) if v < 0 else 'temp+'+str(v) for v in [-5,-2,-1,0,1,2,5]]
            # list_all += ['tpx'+str(v) for v in [0.5,0.667,0.9,1,1.1,1.5,2]]

        for string in list_all:
            fn_out = output_fn.replace('SITE', glacier+site).replace('CHANGE',string)
            output = h5py.File(fn_out,'r')
            if site == 'EC' and string == 'temp+5':
                continue

            if len(string) > 1:
                dict_label = site+'_'+string
            else:
                dict_label = site
            output_dict[dict_label] = {}

            # select annual spring dates to get a density gradient at
            spring_dates = pd.date_range('1980-05-01','2024-05-01',freq='YS-MAY')
            all_decimal_time = output['density'][:, 0]
            target_time = to_decimal_year(spring_dates)
            annual_gradients = []
            annual_firndepth = []
            for t in target_time:
                index = np.argmin(np.abs(all_decimal_time - t))
                # get depth and data arrays
                depth = output['depth'][1:]
                density = output['density'][index, 1:]
                # exclude seasonal snow
                if os.path.exists(f'../Data/cores/{glacier}/{glacier}{site}_snowdepth.csv'):
                    snow_df = pd.read_csv(f'../Data/cores/{glacier}/{glacier}{site}_snowdepth.csv')
                    min_depth = np.mean(snow_df['snowdepth'].values)
                else:
                    min_depth = 2
                condition = (depth >= min_depth) & (density <= 830)
                density = density[condition]
                depth = depth[condition]
                if len(depth) > 1:
                    # gradient, b = np.polyfit(depth, density, deg=1)
                    gradient = np.median(np.gradient(density, depth))
                    annual_firndepth.append(depth[-1])
                    if gradient >= 0:
                        annual_gradients.append(gradient)
                    else:
                        annual_gradients.append(np.nan)
                        continue
                    # b = 400
                    # ax.scatter(density, depth)
                    # ax.plot(depth*gradient + b, depth, color='gray',linewidth=0.5)
                else:
                    annual_firndepth.append(0)
                    annual_gradients.append(np.nan)

            # create dataframe with output vars
            times = np.array(from_decimal_year(output['density'][1:, 0]))
            start = pd.to_datetime(times[0]).date()
            end = pd.to_datetime(times[-1]).date()
            date_times = pd.date_range(start, end, freq='d')
            if len(date_times) > len(times):
                date_times = date_times[:-1]
            df = pd.DataFrame({'melt': output['meltvol'][1:, 1]}, index=date_times)
            for var in ['refreeze','runoff','DIP']:
                df[var] = output[var][1:, 1]
            df['temperature'] = np.mean(output['temperature'][1:, 1:], axis=1)
            df_out = df[['refreeze','runoff','melt']].resample('YS-APR').sum().iloc[1:-1]
            df_out['DIP'] = df['DIP'].resample('YS-APR').mean().iloc[1:-1]
            df_out['temperature'] = df['temperature'].resample('YS-APR').mean().iloc[1:-1] - 273.15
            df_out['refreeze_ratio'] = df_out['refreeze'] / df_out['melt']
            df_out['density_gradient'] = annual_gradients
            df_out['firn_depth'] = annual_firndepth

            # LOAD FORCING VARS
            fn_force = forcing_fn.replace('SITE', glacier+site).replace('CHANGE', string)
            df = pd.read_csv(fn_force, index_col=0, parse_dates=True)
            df_mb = df[['SMELT','BDOT','RAIN']].resample('YS-APR').sum().iloc[:-1] / 1000
            df_mb['STEMP'] = df['TS'].resample('YS-APR').mean().iloc[:-1]- 273.15
            df_mb['MELT_ACC_RATIO'] = df_mb['SMELT'] / df_mb['BDOT']
            df_mb['ACC_MELT_RATIO'] = df_mb['BDOT'] / df_mb['SMELT']
            df_mb['WATER'] = df_mb['SMELT'] + df_mb['RAIN']
            for var in all_vars:
                # Load data for this var
                if var in output_vars:
                    output_dict[dict_label][var] = df_out[var]
                else:
                    output_dict[dict_label][var] = df_mb[var]
            output_dict[dict_label]['elevation'] = [site_elevation[site]]
    if 'EC' not in output_dict:
        output_dict['EC'] = output_dict['EC_temp+0']
        output_dict['KPS'] = output_dict['KPS_temp+0']
        output_dict['Z'] = output_dict['Z_temp+0']
        output_dict['T'] = output_dict['T_temp+0']
    return output_dict

def compare_site_characteristics(plot_vars, xvar, output_dict):
    var_dict = {'SMELT':'Melt (m w.e. / yr)', 'BDOT':'Accumulation (m w.e. / yr)', 
                'MELT_ACC_RATIO':'Annual Melt / Accumulation Ratio','RAIN':'Rainfall (m w.e. / yr)',
                'WATER':'Melt + Rainfall (m w.e. / yr)','elevation':'Elevation (m a.s.l.)',
                'refreeze':'Refreeze (m w.e. / yr)','DIP':'Firn Air Content (m)','firn_depth':'Firn depth (m)',
                'runoff':'Runoff (m w.e. / yr)','density_gradient':'Density gradient (kg / m$^{-3}$ / m)',
                'refreeze_ratio':'Refreeze / Melt Ratio (-)', 'ACC_MELT_RATIO':'Annual Accumulation / Melt Ratio',
                'STEMP':'Surface temperature ($^{\circ}$C)','temperature':'Mean firn temperature ($^{\circ}$C)'
                }
    
    nrows = 1
    ncols = len(plot_vars)
    if len(plot_vars) > 2:
        nrows = 2
        ncols = len(plot_vars) // 2
    
    if len(plot_vars) > 1:
        fig, axes = plt.subplots(nrows, ncols, gridspec_kw = {'hspace':0.4}, figsize=(ncols*3, nrows*2))
        axes = axes.flatten()
        lax = fig.add_axes([1, 0.4, 0.1, 0.2])
        fig.supxlabel(var_dict[xvar])
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*2))
        axes = [axes]
        lax = fig.add_axes([1.2, 0.15, 0.3, 0.8])
        axes[0].set_xlabel(var_dict[xvar])

    markers = {'EC':'*', 'Z':'s', 'T':'o', 'KPS':'^','KQU':'v','KT1':'1','KT2':'3','KT3':'2'}
    site_colors = {'EC':colors[0],'T':colors[1],'Z':colors[2],'KPS':colors[3],'KQU':colors[4],
               'KT1':colors[5], 'KT2':colors[6], 'KT3':colors[7]}

    # var_dy = {'SMELT':0.2, 'BDOT':0.3, 
    #         'RAIN':0.2,
    #         'MELT_ACC_RATIO':0.05, 'runoff':0.3,
    #         'DIP':0.03,'refreeze_ratio':0.1,
    #         'density_gradient':0.01
    #             }

    for ax, var in zip(axes, plot_vars):
        xs = []
        ys = []
        for _, site in enumerate(output_dict):
            if xvar != 'elevation':
                x = output_dict[site][xvar].mean()
                xerr = output_dict[site][xvar].std()
            else:
                x = output_dict[site][xvar][0]
                xerr = 0

            if var != 'elevation':
                y = output_dict[site][var].median()
                yerr = output_dict[site][var].std()
            else:
                y = output_dict[site][var][0]
                yerr = 0

            if var == 'refreeze_ratio':
                lower = max(y - yerr, 0)
                upper = y + yerr
                yerr = [[y - lower], [upper - y]]
                ax.set_ylim(0, 1.2)
            if var  =='runoff':
                ax.set_ylim(0, 3)
            xs.append(x)
            ys.append(y)
                
            s = site if '_' not in site else site.split('_')[0]
            if xvar == 'MELT_ACC_RATIO' and x > 1:
                continue
            if '_' in site:
                # color = site_colors[s] # 
                idict = {'temp-5':0,'temp-2':1,'temp-1':2,'temp+1':4,'temp+2':5,'temp+5':6,
                         'tpx0.5':0,'tpx0.667':1,'tpx0.9':2,'tpx1.1':4,'tpx1.5':5,'tpx2':6}
                color = div_colors(idict[site.split('_')[-1]]) if site[-2:] not in ['+0','x1'] else 'gray'
                ax.errorbar(x, y, xerr=xerr, yerr=yerr,color=color, marker=markers[s], markersize=8, alpha=0.6)
            else:
                ax.errorbar(x, y, xerr=xerr, yerr=yerr,color=site_colors[s], marker=markers[s], markersize=8)
            
            if ax == axes[0]:
                elev = site_elevation[s]
                label = site.split('_')[0] + f'({elev} m)'
                if 'EC_temp-1' in output_dict:
                    if '_' not in site:
                        lax.errorbar(np.nan, np.nan, np.nan, np.nan, label=label, 
                             color=site_colors[s], marker=markers[s], markersize=8)
                else:
                    lax.errorbar(np.nan, np.nan, np.nan, np.nan, label=label, 
                             color=site_colors[s], marker=markers[s], markersize=8)
                
            title = var_dict[var]
            if var == 'MELT_ACC_RATIO':
                title = 'Melt / Accumulation Ratio'
            if len(axes) > 1:
                ax.set_title(title)
            else:
                ax.set_ylabel(title.replace('(', '\n('))
            if var == 'density_gradient':
                ax.set_ylim(0, 50)
        # def place_labels(x, y, sites=sites, ax=ax, threshold=0.1):
        #     placed = []
        #     for i, (xi, yi, site) in enumerate(zip(x, y, sites)):
        #         dx, dy = (0.07, var_dy[var])
        #         for xj, yj in placed:
        #             dist = np.hypot(xi + dx - xj, yi + dy - yj)
        #             if dist < threshold:
        #                 dx *= -1
        #         # if site in ['EC','KPS']:
        #         #     ha, va = ('left','bottom')
        #         # else:
        #         ha, va = ['center','center']
        #         ax.text(xi, dy,  site, fontsize=12, ha=ha, va=va, color=colors[i])
        #         # ax.scatter(xi + dx, yi + dy)
        #         placed.append((xi + dx, yi + dy))

        # place_labels(np.array(xs), np.array(ys), sites, ax)

    lax.legend()
    lax.axis('off')
    plt.savefig(base_fp+f'Figs/site_comparisons_{xvar}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_melt_acc(sites):
    fig, ax = plt.subplots(figsize=(3, 3))
    xs = []
    ys = []

    for s, site in enumerate(sites):
        glacier = get_glacier(site)
        forcing_fn = base_fp + f'Forcings/{glacier}{site}/{glacier}{site}_1d_sitecal_forcings.csv'
        df = pd.read_csv(forcing_fn, index_col=0, parse_dates=True)
        df_mb = df[['SMELT','BDOT','RAIN']].resample('YS-APR').sum().iloc[:-1] / 1000
        df_mb['MELT_ACC_RATIO'] = df_mb['SMELT'] / df_mb['BDOT']
        df_mb['ACC_MELT_RATIO'] = df_mb['BDOT']/ df_mb['SMELT']
        df_mb['WATER'] = df_mb['SMELT'] + df_mb['RAIN']
        
        annual_means = df_mb.mean()
        y = annual_means['SMELT']
        yerr = df_mb['SMELT'].std()
        
        x = annual_means['BDOT']
        xerr = df_mb['BDOT'].std()
        xs.append(x)
        ys.append(y)

        ax.errorbar(float(x), float(y), xerr=xerr, yerr=yerr, label=site, color=colors[s], marker=markers[site], markersize=8, )
        ax.plot([0, float(x)], [0, float(y)], color=colors[s], linestyle='--', linewidth=0.8)
        ax.set_ylabel('Melt rate (m w.e. / yr)')
        ax.set_xlabel('Accumulation rate (m w.e. / yr)')

    ax.plot([0, 6], [0, 6], color='k', linestyle='--', linewidth=0.8)
    ax.text(2.8, 3.3, 'Ablation area', rotation=45)
    ax.text(2.8, 2.4, 'Accumulation area', rotation=45)

    ax.set_ylim(0, 5)
    ax.set_xlim(0, 5)
    ax.legend(loc='upper left')
    plt.savefig(base_fp+'Figs/melt_acc_byglacier_wslope.png', dpi=300, bbox_inches='tight')
    plt.show()

def animate(modeled, measured, dates, snowdepths=0):
    fig, (ax, lax) = plt.subplots(1, 2, width_ratios=(1, 0.2))
    ax.set_xlim(400, 900)
    ax.set_ylim(0, 30)
    ax.invert_yaxis()
    lax.plot(np.nan, np.nan, c=colors[0], linestyle=':',label='Model',linewidth=2)
    lax.plot(np.nan, np.nan, c='k',label='Measured')
    lax.fill_between([np.nan, np.nan], np.nan, np.nan, color='gray', alpha=0.3, label='Seasonal snow')
    lax.axis('off')
    lax.legend(loc='center')

    # create plot lines
    line_meas, = ax.plot([], [],color='k')
    line_mod, = ax.plot([], [],color=colors[0], linestyle=':',linewidth=2)
    fill = ax.fill_between([300, 900], 0, 0, color='gray', alpha=0.3)
    if snowdepths == 0:
        snowdepths = [0] * len(modeled)

    def init():
        line_mod.set_data([],[])
        line_meas.set_data([],[])
        return line_meas, line_mod, fill
    
    def update(i):
        density_meas, bottoms, tops = measured[i]
        density_mod, depth_mod = modeled[i]
        snow_depth = snowdepths[i]
        date = dates[i]

        all_meas, all_meas_depths = ([], [])
        for dens, bottom, top in zip(density_meas, bottoms, tops):
            all_meas.append([dens, dens])
            all_meas_depths.append([top, bottom])
        all_meas = np.array(all_meas).flatten()
        all_meas_depths = np.array(all_meas_depths).flatten()
        line_meas.set_data(all_meas, all_meas_depths)
        line_mod.set_data(density_mod, depth_mod)
        for coll in ax.collections:
            coll.remove()
        ax.fill_between([300, 900], 0, snow_depth,color='gray', alpha=0.3)
        ax.set_title(date[:4])
        return line_meas, line_mod
    
    ani = FuncAnimation(fig, update, frames=len(dates),
                            init_func=init, blit=False)
    ani.save(base_fp+'Figs/animation.gif', fps=2)
    return

def animate_sites(sites, modeled, measured, dates, snowdepths=0):
    fig, axes = plt.subplots(1, len(sites) + 1, width_ratios=(1, 1, 1, 1, 1), sharey=True)
    site_axes = axes[:-1]
    lax = axes[-1]
    lax.plot(np.nan, np.nan, c=colors[0], linestyle=':',label='Model',linewidth=2)
    lax.plot(np.nan, np.nan, c='k',label='Measured')
    lax.fill_between([np.nan, np.nan], np.nan, np.nan, color='gray', alpha=0.3, label='Seasonal snow')
    lax.axis('off')
    lax.legend(loc='center')

    # create plot lines
    lines_meas = []
    lines_mod = []
    fills = []
    if snowdepths == 0:
        snowdepths = [0] * len(modeled[0])

    for s,ax in enumerate(site_axes):
        ax.set_xlim(400, 900)
        ax.set_ylim(0, 30)
        ax.invert_yaxis()
        line_meas, = ax.plot([], [], color='k')
        line_mod,  = ax.plot([], [], color=colors[s], linestyle=':', linewidth=2)
        fill = ax.fill_between([300, 900], 0, 0, color='gray', alpha=0.3)
        lines_meas.append(line_meas)
        lines_mod.append(line_mod)
        fills.append(fill)

    def init():
        for line_meas, line_mod in zip(lines_meas, lines_mod):
            line_meas.set_data([], [])
            line_mod.set_data([], [])
        return lines_meas + lines_mod + fills

    def update(i):
        artists = []
        for s in range(len(sites)):
            ax = site_axes[s]
            density_meas, bottoms, tops = measured[s][i]
            density_mod, depth_mod = modeled[s][i]
            snow_depth = snowdepths[s][i]
            if len(snow_depth) == 0:
                snow_depth = [0]
            date = dates[i]

            # Measured profile as vertical bars
            all_meas, all_meas_depths = [], []
            for dens, bottom, top in zip(density_meas, bottoms, tops):
                all_meas.append([dens, dens])
                all_meas_depths.append([top, bottom])
            all_meas = np.array(all_meas).flatten()
            all_meas_depths = np.array(all_meas_depths).flatten()

            lines_meas[s].set_data(all_meas, all_meas_depths)
            lines_mod[s].set_data(density_mod, depth_mod)

            # Remove old fill and add new snow shading
            for coll in ax.collections:
                coll.remove()
            ax.fill_between([300, 900], 0, snow_depth, color='gray', alpha=0.3)

            site = sites[s]
            glacier = get_glacier(site)
            ax.set_title(f'{glacier.capitalize()} {site}\n{date[:4]}')
            artists.extend([lines_meas[s], lines_mod[s]])

        return artists
    
    ani = FuncAnimation(fig, update, frames=len(dates),
                            init_func=init, blit=False)
    ani.save(base_fp+'Figs/sites_animation.gif', fps=2)
    return

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
        glacier = get_glacier(site).capitalize()
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