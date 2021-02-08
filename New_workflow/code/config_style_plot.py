import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_params():
    mpl.rcParams['text.latex.preamble'] = ['\\usepackage{gensymb}']
    mpl.rcParams['image.origin'] = 'lower'
    mpl.rcParams['image.interpolation'] = 'nearest'
    mpl.rcParams['image.cmap'] = 'RdBu_r'
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['savefig.dpi'] = 300 # to adjust notebook inline plot size
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['axes.labelsize'] = 10 # fontsize for x and y labels (was 10)
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['font.size'] = 14 # was 10
    mpl.rcParams['legend.fontsize'] = 12 # was 10
    mpl.rcParams['legend.loc'] = 'lower right'
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['figure.figsize'] = [8., 8.]
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['font.family'] = 'serif'
    #    mpl.rcParams['colorbar.orientation'] = 'vertical'
    mpl.rcParams['contour.negative_linestyle'] = 'solid'

def colorbar(mappable, size, pad, orient=None):
    '''Plots colorbar from a mappable.
    
    input >
    mappable:   image  - matplotlib image
    size:       string - size in percentage
    pad:        float  - space between figure and colorbar
    orient:     string - orientation of colorbar
    
    output >
    '''
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    if orient != None:
        cax = divider.append_axes("bottom", size=size, pad=pad)
        c = fig.colorbar(mappable, cax=cax, orientation=orient)
    else:
        cax = divider.append_axes("right", size=size, pad=pad)
        c = fig.colorbar(mappable, cax=cax)
    return c