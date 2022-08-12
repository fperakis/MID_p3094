import h5py
import numpy as np

def visit_func(name, node):
    '''Return all groups and datasets name and shapes of h5 file
    '''
    if isinstance(node, h5py.Group):
        print(node.name)
    elif isinstance(node, h5py.Dataset):
        print('\t', node.name, node.shape)
        
def load_data(filename, key):
    """Load dataset and return entry.
    """
    with h5py.File(filename, 'r') as f:
        data = f[key]
        data = np.asarray(data)
    return data

def get_Iq(filename):
    """Load pulse and train resolved azimuthal intensities and averaged 2d image.
    """
    Iq = load_data(filename, '/pulse_resolved/azimuthal_intensity/I')
    q = load_data(filename, '/pulse_resolved/azimuthal_intensity/q')
    img = load_data(filename, '/average/image_2d')
    img_avg = np.average(img[0], axis=0)
    
    return img_avg, Iq, q

def get_correlations(filename, shapes=False):
    """Load TTCs
    """
    ttcs_ = load_data(filename, '/train_resolved/correlation/ttc')
    q = load_data(filename, '/train_resolved/correlation/q')
    time = load_data(filename, '/train_resolved/correlation/t')
    stride = load_data(filename, '/train_resolved/correlation/stride')
    
    if shapes:
        print('ttcs: ', ttcs_.shape)
        print('stride: ', stride.shape)
        print('q: ', q.shape)
        print('time: ', time.shape)
        
    return ttcs_, stride, q, time

def calculate_g2(ttc):
    """Calculate the g2 function from a TTC
    """
    g2 = []
    for i in range(ttc.shape[0]):
        g2.append(np.diag(ttc, k=i).mean())
    return g2