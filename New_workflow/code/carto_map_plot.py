# Script developed by Kristoffer Hallam on 21/06/2020
# The codes written here are an adaptation of the map_plot.py script
# in order to use the Cartopy mapping library
# Special thanks to Allan Ramalho who gave the codes related to
# the Cartopy maps.

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader

# import config_style_plot as style
from scripts import config_style_plot
config_style_plot.plot_params()

class PlotMap(object):
    r'''Class which defines the lower layer of the map using Cartopy.'''
    # style.plot_params()

    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        # point_map(fig_size=(20,24))
        if 'fig_size' in kwargs: # select figure size
            assert type(kwargs['fig_size']) == tuple, \
                'Keyword-argument for the figure size is not a tuple'
            # fig, self.ax = plt.subplots(figsize=kwargs['fig_size'])
            fig = plt.figure(figsize=kwargs['fig_size'])
        else:
            # fig, self.ax = plt.subplots(figsize=(8.,10.))
            fig = plt.figure(figsize=(8.,10.))
        self.ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        if 'region' in kwargs: # select corners of figure in long and lat
            assert type(kwargs['region']) == list, \
                'Keyword-argument for the region of figure is not a list'
            self.ax.set_extent(kwargs['region'], ccrs.PlateCarree())
        else:
            self.ax.set_extent([-78.,-34.2,-33.8,7.], ccrs.PlateCarree())
        states = cfeature.NaturalEarthFeature(category='cultural', \
            name='admin_1_states_provinces_shp',scale='50m',facecolor='none')
        # Adding states
        self.ax.add_feature(states, edgecolor='k',linestyle=':', linewidth=2,alpha=0.7)
        g1 = self.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', \
            linewidth=1,color='gray',alpha=0.8)
        #adicionando a geometria do shape da bacia do parnaiba
        if 'basin' in kwargs:
            assert type(kwargs['basin']) == str, \
                'Keyword-argument for the basin of figure is not a string'
            self.ax.add_geometries(Reader(kwargs['basin']).geometries(),ccrs.PlateCarree(), \
                facecolor='none',edgecolor='black', linewidth=4,alpha=0.5)
        # Switching right and top labels off
        g1.right_labels = False
        g1.top_labels = False
        # Commands used for georeferencing
        g1.yformatter = LATITUDE_FORMATTER
        g1.xformatter = LONGITUDE_FORMATTER

        g1.xlabel_style = {'size': 14}
        g1.ylabel_style = {'size': 14}
        return self.function(self.ax, *args, **kwargs)

class Sign(object):
    r'''Class which defines plots the maximum and minimum values of
    potential field data.'''

    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        # Defining each argument variable
        # m = args[0]
        ax = args[0]
        lon = np.asarray(args[1])
        lat = np.asarray(args[2])
        for i in range(lon.size):
            if lon[i] > 180.:
                lon[i] = lon[i] - 360.
        height = np.asarray(args[3])
        data = np.asarray(args[4])
        if 'sign' in kwargs:
            assert type(kwargs['sign']) == tuple, 'Keyword-argument sign is not a tuple'
            lon_max = kwargs['sign'][0]
            lon_min = kwargs['sign'][1]
            lat_max = kwargs['sign'][2]
            lat_min = kwargs['sign'][3]
            # Identifies the maximum and minimum indexes of data
            ind_max = np.argmax(data)
            ind_min = np.argmin(data)
            # Identifies the maximum and minimum indexes of altitude
            ind_hax = np.argmax(height)
            ind_hin = np.argmin(height)
            # Plots maximum and minimum signs
            if 'cmap' in kwargs:
                assert type(kwargs['cmap']) == str, 'Keyword-argument sign is not a string'
                if kwargs['cmap'] == 'terrain':
                    textstr = '\n'.join((
                        'Max Value \n'
                        r'$\phi = %.4f^o$' % (lat[ind_hax], ),
                        r'$\lambda = %.4f^o$' % (lon[ind_hax], ),
                        r'$h = %.2f$m' % (height[ind_max])) )
                    # x, y = m(lon_max, lat_max)
                    x, y = ax.projection.transform_point(lon_max, lat_max, \
                        src_crs=ccrs.PlateCarree())
                    ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), \
                        textcoords='data', bbox=dict(boxstyle="round", fc='saddlebrown', \
                            ec="none", alpha=0.8))
                    textstr = '\n'.join((
                        'Min Value \n'
                        r'$\phi = %.4f^o$' % (lat[ind_hin], ),
                        r'$\lambda = %.4f^o$' % (lon[ind_hin], ),
                        r'$h = %.2f$m' % (height[ind_hin])) )
                    # x, y = m(lon_min, lat_min)
                    x, y = ax.projection.transform_point(lon_min, lat_min, \
                        src_crs=ccrs.PlateCarree())
                    ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), \
                        textcoords='data', bbox=dict(boxstyle="round", fc='steelblue', \
                            ec="none", alpha=0.8))
            else:
                if 'residual' in kwargs:
                    assert type(kwargs['residual']) == bool, \
                        'Keyword-argument sign is not a boolean'
                    textstr = '\n'.join((
                        'Max Value \n'
                        r'r = %.2fmGal' % (data[ind_max], ),
                        r'$\phi = %.4f^o$' % (lat[ind_max], ),
                        r'$\lambda = %.4f^o$' % (lon[ind_max], ),
                        r'$h = %.2f$m' % (height[ind_max])) )
                    # x, y = m(lon_max, lat_max)
                    x, y = ax.projection.transform_point(lon_max, lat_max, \
                        src_crs=ccrs.PlateCarree())
                    ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), \
                        textcoords='data', bbox=dict(boxstyle="round", fc='indianred', \
                            ec="none", alpha=0.8))
                    textstr = '\n'.join((
                        'Min Value \n'
                        r'r = %.2fmGal' % (data[ind_min], ),
                        r'$\phi = %.4f^o$' % (lat[ind_min], ),
                        r'$\lambda = %.4f^o$' % (lon[ind_min], ),
                        r'$h = %.2f$m' % (height[ind_min])) )
                    # x, y = m(lon_min, lat_min)
                    x, y = ax.projection.transform_point(lon_min, lat_min, \
                        src_crs=ccrs.PlateCarree())
                    ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), \
                        textcoords='data', bbox=dict(boxstyle="round", fc='royalblue', \
                            ec="none", alpha=0.8))
                else:
                    textstr = '\n'.join((
                        'Max Value \n'
                        r'$\delta g = %.2f$mGal' % (data[ind_max], ),
                        r'$\phi = %.4f^o$' % (lat[ind_max], ),
                        r'$\lambda = %.4f^o$' % (lon[ind_max], ),
                        r'$h = %.2f$m' % (height[ind_max])) )
                    # x, y = m(lon_max, lat_max)
                    x, y = ax.projection.transform_point(lon_max, lat_max, \
                        src_crs=ccrs.PlateCarree())
                    ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), \
                        textcoords='data', bbox=dict(boxstyle="round", fc='indianred', \
                            ec="none", alpha=0.8))
                    textstr = '\n'.join((
                        'Min Value \n'
                        r'$\delta g = %.2f$mGal' % (data[ind_min], ),
                        r'$\phi = %.4f^o$' % (lat[ind_min], ),
                        r'$\lambda = %.4f^o$' % (lon[ind_min], ),
                        r'$h = %.2f$m' % (height[ind_min])) )
                    # x, y = m(lon_min, lat_min)
                    x, y = ax.projection.transform_point(lon_min, lat_min, \
                        src_crs=ccrs.PlateCarree())
                    ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), \
                        textcoords='data', bbox=dict(boxstyle="round", fc='royalblue', \
                            ec="none", alpha=0.8))
        else:
            pass
        return self.function(*args, **kwargs)

@PlotMap
@Sign
def point_map(*args, **kwargs):
    r'''Plots the potential field data configured as a scatter map.
    
    input >
    args   -  tuple          - (lon, lat, height, data, uf)
    Each args variable within the parenthesis is related to the observables.
    OBS: All variables inside the tuple args are required! They represent:

        lon, lat, height: 1D arrays -> observable coordinates
        data:             1D arrays -> observable data
        uf:               string    -> title of figure
    
    kwargs -  dictionaries   - (figsize, region, drawlines, cmap, lim_val, sign, residual, save)
    Each kwargs variable within the parenthesis is related either to the format of the map or
    to a object to be plotted inside the map. OBS: Only the *config* dict is necessary for plotting!
    Keep in mind that as the keyword-arguments (kwargs) are dictionaries, knowing their position is
    irrelevant. The keyword-arguments are stated as:

        fig_size:         tuple     -> size of figure (height, width). Default is (8.,10.) which
        means no tuple is passed to *fig_size* keyword-argument.

        region:           list      -> limit coordinates of the map displayed as follows
        [left_long,right_long,lower_lat,upper_lat]. Default is [-80.,-34.8,-35.,7.] (Brazilian
        territory boundaries) which means no tuple is passed to *edges* keyword-argument.

        drawlines:        tuple     -> contains the interval between meridians and parallels
        defined as (interv_meridians, interv_parallels). Default is (5.,4.) which means no tuple
        is passed to drawlines keyword-argument.

        cmap:             string    -> set the colormap which will be used to represent the data
        values. For elevation map, set 'terrain' or another colormap string to cmap keyword-argument.
        For gravity disturbance map (default), just don't set anything.

        lim_val:          tuple     -> decides whether to limit the colorbar to values determined
        by the user. If kwargs['lim_val'][0] is set to True, the colorbar values will be limited to
        (-/+)kwargs['lim_val'][1]. If kwargs['lim_val'][0] is set to False, the values will vary
        from (-/+)np.max(np.abs(data)), thus no value for kwargs['lim_val'][1] is needed. However,
        if *lim_val* is not set, the colorbar will be set to vary from np.min(data) to
        np.max(data) as a normal map would show.

        sign:             tuple     -> coordinates of upper maximum and lower minimum signs
        (left_upper_long,left_lower_long,left_upper_lat,left_lower_lat) of data values. If no sign
        is desired, just do not pass any value to *sign* keyword-argument (default).

        residual:         boolean   -> display variable $r$ or $\delta g$ in max and min value signs.
        If True is passed to residual keyword-argument, $r$ is displayed. If no value is passed, then
        $\delta g$ shall be displayed instead.

        save:             string    -> choose whether to save the figure. If a string is passed to
        *save*, the figure will be saved under the path of kwargs['save']. If saving is not desired,
        no object should be passed to *save* keyword argument (default).
    
    output >
    map    -            with all the chosen figure objects
    '''
    ax = args[0]
    lon = np.asarray(args[1])
    lat = np.asarray(args[2])
    for i in range(lon.size):
        if lon[i] > 180.:
            lon[i] = lon[i] - 360.
    height = np.asarray(args[3])
    data = np.asarray(args[4])
    # point_map((...), cmap='terrain', colortitle-)
    if 'cmap' in kwargs:
        assert type(kwargs['cmap']) == str, 'Keyword-argument cmap is not a string'
        colormap = kwargs['cmap']
        colortitle = 'm'
    else:
        colormap = 'RdBu_r'
        colortitle = 'mGal'
    if 'lim_val' in kwargs:
        assert type(kwargs['lim_val'][0]) == bool, \
            'Keyword-argument lim_val[0] is not a boolean'
        if kwargs['lim_val'][0] == True:
            if kwargs['lim_val'][1] == True:
                assert type(kwargs['lim_val'][2]) == float or type(kwargs['lim_val'][2]) == int, \
                'Keyword-argument lim_val[2] is not a float or a int'
                assert type(kwargs['lim_val'][3]) == float or type(kwargs['lim_val'][3]) == int, \
                'Keyword-argument lim_val[3] is not a float or a int'
                cax=ax.scatter(lon, lat, c=data, s=30, vmin=kwargs['lim_val'][2], \
                vmax=kwargs['lim_val'][3], cmap=colormap, transform=ccrs.PlateCarree())
            else:
                assert type(kwargs['lim_val'][1]) == float or type(kwargs['lim_val'][1]) == int, \
                'Keyword-argument lim_val[1] is not a float or a int'
                cax=ax.scatter(lon, lat, c=data, s=30, vmin=-kwargs['lim_val'][1], \
                vmax=kwargs['lim_val'][1], cmap=colormap, transform=ccrs.PlateCarree())
        else:
            dmax = np.max(np.abs(data))
            cax=ax.scatter(lon, lat, c=data, s=30, vmin=-dmax, vmax=dmax, cmap=colormap, \
                transform=ccrs.PlateCarree())
    else:
        cax=ax.scatter(lon, lat, c=data, s=30, vmin=data.min(), vmax=data.max(), \
            cmap=colormap, transform=ccrs.PlateCarree())
    ax.set_title('{}'.format(args[5]), fontsize=20, y=1.02)
    if 'shrink' in kwargs:
        assert type(kwargs['shrink']) == float, 'Keyword-argument shrink is not a float'
        if 'aspect' in kwargs:
            assert type(kwargs['aspect']) == float, 'Keyword-argument aspect is not a float'
            cbar=plt.colorbar(cax, shrink=kwargs['shrink'],pad=0.02,aspect=kwargs['aspect'])
        else:
            cbar=plt.colorbar(cax, shrink=kwargs['shrink'],pad=0.02,aspect=20)
    else:
        cbar=plt.colorbar(cax, shrink=0.51,pad=0.02,aspect=20)
    cbar.set_label(colortitle,fontsize=22,labelpad=2)
    cbar.ax.tick_params(labelsize=12,color='black',labelcolor='black')
    if 'save' in kwargs:
        plt.savefig(kwargs['save'],format='png', dpi=300, bbox_inches='tight')
    plt.show()

@PlotMap
@Sign
def grid_map(*args, **kwargs):
    r'''Plots the potential field data configured as a grid.
    
    input >
    args   -  tuple          - (lon, lat, height, data, uf)
    Each args variable within the parenthesis is related to the observables.
    OBS: All variables inside the tuple args are required! They represent:

        lon, lat, height: 1D arrays -> observable coordinates
        data:             1D arrays -> observable data
        uf:               string    -> title of figure
    
    kwargs -  dictionaries   - (figsize, region, drawlines, cmap, lim_val, sign, residual, save)
    Each kwargs variable within the parenthesis is related either to the format of the map or
    to a object to be plotted inside the map. OBS: Only the *config* dict is necessary for plotting!
    Keep in mind that as the keyword-arguments (kwargs) are dictionaries, knowing their position is
    irrelevant. The keyword-arguments are stated as:

        fig_size:         tuple     -> size of figure (height, width). Default is (8.,10.) which
        means no tuple is passed to *fig_size* keyword-argument.

        region:           list      -> limit coordinates of the map displayed as follows
        [left_long,right_long,lower_lat,upper_lat]. Default is [-80.,-34.8,-35.,7.] (Brazilian
        territory boundaries) which means no tuple is passed to *edges* keyword-argument.

        drawlines:        tuple     -> contains the interval between meridians and parallels
        defined as (interv_meridians, interv_parallels). Default is (5.,4.) which means no tuple
        is passed to drawlines keyword-argument.

        cmap:             string    -> set the colormap which will be used to represent the data
        values. For elevation map, set 'terrain' or another colormap string to cmap keyword-argument.
        For gravity disturbance map (default), just don't set anything.

        lim_val:          tuple     -> decides whether to limit the colorbar to values determined
        by the user. If kwargs['lim_val'][0] is set to True, the colorbar values will be limited to
        (-/+)kwargs['lim_val'][1]. If kwargs['lim_val'][0] is set to False, the values will vary
        from (-/+)np.max(np.abs(data)), thus no value for kwargs['lim_val'][1] is needed. However,
        if *lim_val* is not set, the colorbar will be set to vary from np.min(data) to
        np.max(data) as a normal map would show.

        sign:             tuple     -> coordinates of upper maximum and lower minimum signs
        (left_upper_long,left_lower_long,left_upper_lat,left_lower_lat) of data values. If no sign
        is desired, just do not pass any value to *sign* keyword-argument (default).

        residual:         boolean   -> display variable $r$ or $\delta g$ in max and min value signs.
        If True is passed to residual keyword-argument, $r$ is displayed. If no value is passed, then
        $\delta g$ shall be displayed instead.

        save:             string    -> choose whether to save the figure. If a string is passed to
        *save*, the figure will be saved under the path of kwargs['save']. If saving is not desired,
        no object should be passed to *save* keyword argument (default).
    
    output >
    map    -            with all the chosen figure objects
    '''
    # Defining each argument variable
    ax = args[0]
    lon = np.asarray(args[1])
    lat = np.asarray(args[2])
    for i in range(lon.size):
        if lon[i] > 180.:
            lon[i] = lon[i] - 360.
    height = np.asarray(args[3])
    data = np.asarray(args[4])
    
    # Gets shape of grid and reshapes each variable
    Ngrid = lon.size
    rep = []
    for i in range(1, Ngrid):
        if lon[i] == lon[0]:
            rep.append(i)
    print(rep[0])
    rep = rep[0] # take this + 1 out
    Lat = np.reshape(lat, (int(Ngrid/rep), rep))
    Lon = np.reshape(lon, (int(Ngrid/rep), rep))
    Data = np.reshape(data, (int(Ngrid/rep), rep))
    # x, y = m(Lon, Lat)

    if 'cmap' in kwargs:
        assert type(kwargs['cmap']) == str, 'Keyword-argument cmap is not a string'
        colormap = kwargs['cmap']
        colortitle = 'm'
    else:
        colormap = 'RdBu_r'
        colortitle = 'mGal'
    if 'lim_val' in kwargs:
        assert type(kwargs['lim_val'][0]) == bool, \
            'Keyword-argument lim_val[0] is not a boolean'
        if kwargs['lim_val'][0] == True:
            if kwargs['lim_val'][1] == True:
                assert type(kwargs['lim_val'][2]) == float or type(kwargs['lim_val'][2]) == int, \
                'Keyword-argument lim_val[1] is not a float or a int'
                assert type(kwargs['lim_val'][3]) == float or type(kwargs['lim_val'][3]) == int, \
                'Keyword-argument lim_val[1] is not a float or a int'
                cax = ax.pcolor(Lon, Lat, Data, vmin=-kwargs['lim_val'][2], \
                    vmax=kwargs['lim_val'][3], cmap=colormap, transform=ccrs.PlateCarree())
            else:
                assert type(kwargs['lim_val'][1]) == float or type(kwargs['lim_val'][1]) == int, \
                'Keyword-argument lim_val[1] is not a float or a int'
                cax = ax.pcolor(Lon, Lat, Data, vmin=-kwargs['lim_val'][1], \
                    vmax=kwargs['lim_val'][1], cmap=colormap, transform=ccrs.PlateCarree())
        else:
            dmax = np.max(np.abs(data))
            cax = ax.pcolor(Lon, Lat, Data, vmin=-dmax, vmax=dmax, cmap=colormap, \
                transform=ccrs.PlateCarree())
    else:
        cax = ax.pcolor(Lon, Lat, Data, vmin=data.min(), vmax=data.max(), cmap=colormap, \
            transform=ccrs.PlateCarree())
    ax.set_title('{}'.format(args[5]), fontsize=20, y=1.02)
    if 'shrink' in kwargs:
        assert type(kwargs['shrink']) == float, 'Keyword-argument shrink is not a float'
        cbar=plt.colorbar(cax, shrink=kwargs['shrink'],pad=0.02,aspect=20)
    else:
        cbar=plt.colorbar(cax, shrink=0.51,pad=0.02,aspect=20)
    cbar.set_label(colortitle,fontsize=22,labelpad=2)
    cbar.ax.tick_params(labelsize=12,color='black',labelcolor='black')
    if 'save' in kwargs:
        plt.savefig(kwargs['save'],format='png', dpi=300, bbox_inches='tight')
    plt.show()