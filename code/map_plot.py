import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import config_style_plot as style

style.plot_params()
shapefile='../data/Brazil_shp/BRA_adm1'

class PlotMap(object):
    r'''Class which defines the lower layer of the map using Basemap.'''
    #style.plot_params()

    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        if 'fig_size' in kwargs: # select figure size
            assert type(kwargs['fig_size']) == tuple, \
                'Keyword-argument for the figure size is not a tuple'
            fig, self.ax = plt.subplots(figsize=kwargs['fig_size'])
        else:
            fig, self.ax = plt.subplots(figsize=(8.,10.))
        if 'center' in kwargs: # select center of figure
            assert type(kwargs['center']) == tuple, \
                'Keyword-argument for the center of figure is not a tuple'
            center = kwargs['center']
        else:
            center = [-18.,-14.,-55.]
        if 'edges' in kwargs: # select corners of figure in long and lat
            assert type(kwargs['edges']) == tuple, \
                'Keyword-argument for the edges of figure is not a tuple'
            area = [kwargs['edges'][0],kwargs['edges'][1],\
                kwargs['edges'][2],kwargs['edges'][3]]
        else:
            # area = [-80.,-34.8,-35.,7.]
            area = [-78.,-35.,-33.8,7.]
        self.m = Basemap(llcrnrlon=area[0],urcrnrlon=area[1],llcrnrlat=area[2], \
            urcrnrlat=area[3], \
                resolution='l',area_thresh=10000000.,projection='lcc',lat_1=center[0],\
                    lat_2=center[1],lon_0=center[2],ax=self.ax)
        self.m.drawcoastlines()
        self.m.drawcountries()
        if 'drawlines' in kwargs: # draw meridians and parallels
            assert type(kwargs['drawlines']) == tuple, \
                'Keyword-argument drawlines is not a tuple'
            self.m.drawmeridians(np.arange(-100,0,kwargs['drawlines'][0]),labels=[0,0,0,1])
            self.m.drawparallels(np.arange(-50,20,kwargs['drawlines'][1]),labels=[1,0,0,0])
        else:
            self.m.drawmeridians(np.arange(-100,0,5.),labels=[0,0,0,1])
            self.m.drawparallels(np.arange(-50,20,4.),labels=[1,0,0,0])

        # Plots Brazilian states
        shp = self.m.readshapefile(shapefile, 'states', drawbounds=True)
        for nshape, seg in enumerate(self.m.states):
            poly = Polygon(seg, alpha = 0.2, facecolor='1.', edgecolor='k')
            self.ax.add_patch(poly)
        return self.function(self.m, self.ax, *args, **kwargs)

class Sign(object):
    r'''Class which defines plots the maximum and minimum values of
    potential field data.'''

    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        # Defining each argument variable
        m = args[0]
        ax = args[1]
        lon = np.asarray(args[2])
        lat = np.asarray(args[3])
        for i in range(lon.size):
            if lon[i] > 180.:
                lon[i] = lon[i] - 360.
        height = np.asarray(args[4])
        data = np.asarray(args[5])
        if 'sign' in kwargs:
            assert type(kwargs['sign']) == tuple, 'Keyword-argument sign is not a tuple'
            lon_max = kwargs['sign'][0]
            lon_min = kwargs['sign'][1]
            lat_max = kwargs['sign'][2]
            lat_min = kwargs['sign'][3]
            # Identifies the maximum and minimum indexes of data
            ind_max = np.where(data==np.max(data))[0]
            ind_min = np.where(data==np.min(data))[0]
            # Identifies the maximum and minimum indexes of altitude
            ind_hax = np.where(data==np.max(height))[0]
            ind_hin = np.where(data==np.min(height))[0]
            # Plots maximum and minimum signs
            if 'cmap' in kwargs:
                textstr = '\n'.join((
                    'Max Value \n'
                    r'$\phi = %.4f^o$' % (lat[ind_hax[0]], ),
                    r'$\lambda = %.4f^o$' % (lon[ind_hax[0]], ),
                    r'$h = %.2f$m' % (height[ind_max[0]])) )
                x, y = m(lon_max, lat_max)
                ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), textcoords='data', \
                    bbox=dict(boxstyle="round", fc='saddlebrown', ec="none", alpha=0.8))
                textstr = '\n'.join((
                    'Min Value \n'
                    r'$\phi = %.4f^o$' % (lat[ind_hin[0]], ),
                    r'$\lambda = %.4f^o$' % (lon[ind_hin[0]], ),
                    r'$h = %.2f$m' % (height[ind_hin][0])) )
                x, y = m(lon_min, lat_min)
                ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), textcoords='data', \
                    bbox=dict(boxstyle="round", fc='steelblue', ec="none", alpha=0.8))
            else:
                if 'residual' in kwargs:
                    assert type(kwargs['residual']) == bool, 'Keyword-argument sign is not a boolean'
                    textstr = '\n'.join((
                        'Max Value \n'
                        r'r = %.2fmGal' % (data[ind_max[0]], ),
                        r'$\phi = %.4f^o$' % (lat[ind_max[0]], ),
                        r'$\lambda = %.4f^o$' % (lon[ind_max[0]], ),
                        r'$h = %.2f$m' % (height[ind_max[0]])) )
                    x, y = m(lon_max, lat_max)
                    ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), textcoords='data', \
                        bbox=dict(boxstyle="round", fc='indianred', ec="none", alpha=0.8))
                    textstr = '\n'.join((
                        'Min Value \n'
                        r'r = %.2fmGal' % (data[ind_min[0]], ),
                        r'$\phi = %.4f^o$' % (lat[ind_min[0]], ),
                        r'$\lambda = %.4f^o$' % (lon[ind_min[0]], ),
                        r'$h = %.2f$m' % (height[ind_min[0]])) )
                    x, y = m(lon_min, lat_min)
                    ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), textcoords='data', \
                        bbox=dict(boxstyle="round", fc='royalblue', ec="none", alpha=0.8))
                else:
                    textstr = '\n'.join((
                        'Max Value \n'
                        r'$\delta g = %.2f$mGal' % (data[ind_max[0]], ),
                        r'$\phi = %.4f^o$' % (lat[ind_max[0]], ),
                        r'$\lambda = %.4f^o$' % (lon[ind_max[0]], ),
                        r'$h = %.2f$m' % (height[ind_max[0]])) )
                    x, y = m(lon_max, lat_max)
                    ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), textcoords='data', \
                        bbox=dict(boxstyle="round", fc='indianred', ec="none", alpha=0.8))
                    textstr = '\n'.join((
                        'Min Value \n'
                        r'$\delta g = %.2f$mGal' % (data[ind_min[0]], ),
                        r'$\phi = %.4f^o$' % (lat[ind_min[0]], ),
                        r'$\lambda = %.4f^o$' % (lon[ind_min[0]], ),
                        r'$h = %.2f$m' % (height[ind_min[0]])) )
                    x, y = m(lon_min, lat_min)
                    ax.annotate(textstr, xy=(x,y), xycoords='data', xytext=(x,y), textcoords='data', \
                        bbox=dict(boxstyle="round", fc='royalblue', ec="none", alpha=0.8))
        else:
            pass
        return self.function(*args, **kwargs)

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
        uf:               string    -> name of BR state
    
    kwargs -  dictionaries   - (figsize, edges, center, drawlines, cmap, lim_val, sign, residual, save)
    Each kwargs variable within the parenthesis is related either to the format of the map or
    to a object to be plotted inside the map. OBS: Only the *config* dict is necessary for plotting!
    Keep in mind that as the keyword-arguments (kwargs) are dictionaries, knowing their position is
    irrelevant. The keyword-arguments are stated as:

        fig_size:         tuple     -> size of figure (height, width). Default is (8.,10.) which
        means no tuple is passed to *fig_size* keyword-argument.

        edges:            tuple     -> limit coordinates of the map displayed as follows
        (left_long,right_long,lower_lat,upper_lat). Default is [-80.,-34.8,-35.,7.] (Brazilian
        territory boundaries) which means no tuple is passed to *edges* keyword-argument.

        center:           tuple     -> center of figure displayed as follows (lat_1, lat_2, lon_0).
        Check Basemap tutorial for more info on these arguments. Default is [-18.,-14.,-55.] which
        means no tuple is passed to *center* keyword-argument.

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
    m = args[0]
    ax = args[1]
    lon = np.asarray(args[2])
    lat = np.asarray(args[3])
    for i in range(lon.size):
        if lon[i] > 180.:
            lon[i] = lon[i] - 360.
    height = np.asarray(args[4])
    data = np.asarray(args[5])
    
    # Gets shape of grid and reshapes each variable
    Ngrid = lon.size
    rep = []
    for i in range(1, Ngrid):
        if lon[i] == lon[0]:
            rep.append(i)
    rep = rep[0]
    Lat = np.reshape(lat, (Ngrid/rep, rep))
    Lon = np.reshape(lon, (Ngrid/rep, rep))
    Data = np.reshape(data, (Ngrid/rep, rep))
    x, y = m(Lon, Lat)

    # Plotting grid on map
    if 'cmap' in kwargs:
        assert type(kwargs['cmap']) == str, 'Keyword-argument cmap is not a string'
        colormap = kwargs['cmap']
    else:
        colormap = 'RdBu_r'
    if 'lim_val' in kwargs:
        assert type(kwargs['lim_val'][0]) == bool, 'Keyword-argument lim_val[0] is not a boolean'
        if kwargs['lim_val'][0] == True:
            assert type(kwargs['lim_val'][1]) == float or type(kwargs['lim_val'][1]) == int, \
                'Keyword-argument lim_val[1] is not a float or a int'
            cax = m.pcolor(x, y, Data, vmin=-kwargs['lim_val'][1], vmax=kwargs['lim_val'][1], cmap=colormap)
        else:
            dmax = np.max(np.abs(data))
            cax = m.pcolor(x, y, Data, vmin=-dmax, vmax=dmax, cmap=colormap)
    else:
        cax = m.pcolor(x, y, Data, vmin=np.min(data), vmax=np.max(data), cmap=colormap)
    cbar = style.colorbar(cax, "5%", 0.15)
    cbar.set_label('mGal', fontsize=16)
    ax.set_title(args[6])
    # max_min_squares(*args,**kwargs)

@PlotMap
@Sign
def point_map(*args, **kwargs):
    r'''Plots the potential field data configured as a grid.
    
    input >
    args   -  tuple          - (lon, lat, height, data, uf)
    Each args variable within the parenthesis is related to the observables.
    OBS: All variables inside the tuple args are required! They represent:

        lon, lat, height: 1D arrays -> observable coordinates
        data:             1D arrays -> observable data
        uf:               string    -> name of BR state
    
    kwargs -  dictionaries   - (figsize, edges, center, drawlines, cmap, lim_val, sign, residual, save)
    Each kwargs variable within the parenthesis is related either to the format of the map or
    to a object to be plotted inside the map. OBS: Only the *config* dict is necessary for plotting!
    Keep in mind that as the keyword-arguments (kwargs) are dictionaries, knowing their position is
    irrelevant. The keyword-arguments are stated as:

        fig_size:         tuple     -> size of figure (height, width). Default is (8.,10.) which
        means no tuple is passed to *fig_size* keyword-argument.

        edges:            tuple     -> limit coordinates of the map displayed as follows
        (left_long,right_long,lower_lat,upper_lat). Default is [-80.,-34.8,-35.,7.] (Brazilian
        territory boundaries) which means no tuple is passed to *edges* keyword-argument.

        center:           tuple     -> center of figure displayed as follows (lat_1, lat_2, lon_0).
        Check Basemap tutorial for more info on these arguments. Default is [-18.,-14.,-55.] which
        means no tuple is passed to *center* keyword-argument.

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
    m = args[0]
    ax = args[1]
    lon = np.asarray(args[2])
    lat = np.asarray(args[3])
    for i in range(lon.size):
        if lon[i] > 180.:
            lon[i] = lon[i] - 360.
    height = np.asarray(args[4])
    data = np.asarray(args[5])
    
    x, y = m(lon, lat)

    if 'cmap' in kwargs:
        assert type(kwargs['cmap']) == str, 'Keyword-argument cmap is not a string'
        colormap = kwargs['cmap']
    else:
        colormap = 'RdBu_r'
    if 'lim_val' in kwargs:
        assert type(kwargs['lim_val'][0]) == bool, 'Keyword-argument lim_val[0] is not a boolean'
        if kwargs['lim_val'][0] == True:
            assert type(kwargs['lim_val'][1]) == float or type(kwargs['lim_val'][1]) == int, \
                'Keyword-argument lim_val[1] is not a float or a int'
            cax = m.scatter(x, y, s=30, c=data, vmin=-kwargs['lim_val'][1], \
                vmax=kwargs['lim_val'][1], cmap=colormap)
        else:
            dmax = np.max(np.abs(data))
            cax = m.scatter(x, y, s=30, c=data, vmin=-dmax, vmax=dmax, cmap=colormap)
    else:
        cax = m.scatter(x, y, s=30, c=data, vmin=np.min(data), vmax=np.max(data), \
            cmap=colormap)
    cbar = style.colorbar(cax, "5%", 0.15)
    if 'cmap' in kwargs:
        cbar.set_label('m', fontsize=16)
    else:
        cbar.set_label('mGal', fontsize=16)
    ax.set_title(args[6])
    # max_min_squares(*args,**kwargs)