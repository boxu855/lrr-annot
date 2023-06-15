import os
from os import path
from os.path import isfile, join, dirname, isdir, exists

import numpy as np

from Bio.PDB import *
from scipy.ndimage import gaussian_filter1d, gaussian_filter

import matplotlib.pyplot as plt

def make_dir(*argv):
    mydir = path.join(*argv)    
    if not path.exists(mydir):        
        if len(argv) > 1:
            make_dir(*argv[:-1])            
        os.mkdir(mydir)
    return mydir

def make_path(*argv):
    mypath = path.join(*argv)
    if not path.exists(dirname(mypath)):
        make_dir(*argv[:-1])
    return mypath

#retrieve all files in a directory with a given extension
def get_files_with_ext(directory, ext):
    file_list = []
    for file in os.listdir(directory):
        if file.endswith(ext):
            file_list.append(file)
    return file_list

#return alpha carbon space curve from pdb file
def get_backbone_from_pdb(path):    
    parser = PDBParser()
    chain = next(parser.get_structure('', path).get_chains())
    return np.array([np.array(list(residue["CA"].get_vector())) for residue in chain.get_residues()])

#given two vectors a and b find orthonormal vectors closest to a and b with the same span
def compromise(a, b):
    X = np.array([a,b])
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    Y = u @ vh
    return [*Y]

#helper function for loss function
def get_premidpost(winding, params, slope):
    l, r = params
    l = int(l)
    r = int(r)    
    pre = np.array(winding[:l])
    pre -= np.mean(pre)
    mid = winding[l:r] - (slope * (np.arange(l, r)))
    mid -= np.mean(mid)
    post = np.array(winding[r:])
    if len(post):
        post -= np.mean(post)    
    return pre, mid, post

#compute loss for piecewise linear regression
def loss(winding, params, slope, penalties):
    pre, mid, post = get_premidpost(winding, params,slope)
    return penalties[0] * np.sum(pre ** 2) + penalties[1] * np.sum(mid ** 2) + penalties[0] * np.sum(post ** 2)

#given backbone curve return framing of normal bundle
def get_frame(preX):
    X = gaussian_filter(preX, [1, 0]) # smoothed out curve
    dX = gaussian_filter(X, [1, 0], order = 1) # tangent of backbone
    Y = gaussian_filter(X, [20, 0]) # backbone
    dY = gaussian_filter(Y, [1, 0], order = 1) # tangent of backbone
    dZ = dY / np.sqrt(np.sum(dY ** 2, axis = 1))[:, np.newaxis]
    
    V = np.zeros((len(dZ), 2, 3)) # V[i] is an orthonormal basis for the orthogonal complement of dZ[i]
    np.random.seed(100)
    V[0] = np.random.rand(2, 3)
    for i, z in enumerate(dZ):
        # project onto current tangent vector
        if i:
            V[i] = V[i-1]

        V[i] -= np.outer(V[i] @ z, z)
        V[i] = compromise(*V[i]) 
    Q = np.zeros((len(dZ), 4))
    for i in range(len(Q)):
        Q[i] = [(X[i] - Y[i]) @ V[i,0], (X[i] - Y[i]) @ V[i,1], (X[i] - Y[i]) @ dZ[i], dX[i] @ dZ[i]]
    return X, dX, Y, dY, dZ, V, Q

#given backbone curve return cumulative winding number
def get_winding(preX):
    X, dX, Y, dY, dZ, V, Q = get_frame(preX)    
    s, c, q, dx = Q.T
    ds = gaussian_filter(s, 0.5, order = 1)
    dc = gaussian_filter(c, 0.5, order = 1)
    dq = gaussian_filter(q, 0.5, order = 1)
    r2 = s ** 2 + c ** 2
    summand = (c * ds - s * dc) / r2
    winding = np.cumsum(summand) / (2 * np.pi)
    winding *= np.sign(winding[-1] - winding[0])
    return winding, s, c, q, dx

#returns median slope of secant lines  winding number graph. `small` and `big` refer to lower and upper bounds for sequence
def median_slope(data, small, big):
    slopes = []
    weights = []
    for i in range(len(data) - small):
        for j in range(i + small, min(i + big, len(data))):
            s = (data[j]-data[i])/(j-i)
            slopes.append(s)
            reg = data[i:j] - s * np.arange(i,j)
            reg -= np.mean(reg)
            # weights.append(np.sqrt(j - i))
            weights.append((j - i) / (1 + np.sum(reg ** 2)))
    
    n_bins = int(np.sqrt(len(slopes)))
    scores = [0 for i in range(n_bins)]
    a = min(slopes)
    b = max(slopes) + 0.01
    # print(a, b, n_bins)
    
    # bin_vals = np.arange(a, b, (b - a) / n_bins)
    # assert len(bin_vals) == len(scores)
    
    for s, weight in zip(slopes, weights):
        bin_index = int(n_bins * (s - a) / (b - a))
        try:
            scores[bin_index] += weight
        except:
            print(bin_index, n_bins, s, a, b)
        
    return a + (np.argmax(scores) / n_bins) * (b - a), scores

# regression on winding number graph of backbone curve
def get_regression(preX):
    winding, s, c, q, dx = get_winding(preX)

    n = len(winding)
    parameters = np.array([n // 2, (3 * n) // 4])
    penalties = [1, 1.5]
    epsilon = 0.01
    gradient = np.zeros(2)
    delta = [*np.identity(2)]
    gradient_l = []
    prev_grad = np.array(gradient)
    thresh = .3
    m, scores = median_slope(winding, 150, 250)
    for i in range(10000):
        present = loss(winding, parameters, m, penalties)
        if np.linalg.norm(gradient - prev_grad)< thresh and i > 0:
            break
        gradient_l.append(gradient)
        gradient = np.array([loss(winding, parameters + d, m, penalties) - present for d in delta])
        parameters = parameters - epsilon * gradient

    if parameters[1]>.90*len(winding):
        parameters[1] = len(winding)        

    return winding, m, parameters

#plot regression with option to save file. return standard deviation of middle line segment
def plot_regression(winding, params, slope, save = False, filename = ''):
    l, r = params
    l = int(l)
    r = int(r)
    pre, mid, post = get_premidpost(winding, params, slope)
    
    plt.plot(winding)
    plt.plot(np.arange(l), winding[:l] - pre, c= 'red')
    plt.plot(np.arange(l, r), winding[l:r] - mid, c = 'green')
    plt.plot(np.arange(r, len(winding)), winding[r:] - post, c = 'purple')
    plt.title('Piecewise linear regression on winding number graph')
    plt.axvline(l, linestyle = '--', c= 'r')
    plt.axvline(r, linestyle ='--', c = 'purple')
    plt.xlabel('Residue number')
    plt.ylabel('Winding number')
    
    if save:
        plt.savefig(filename + '.png')
        plt.close()
    else:
        plt.show()
    return np.std(mid)

def get_segs(winding, params, slope):
    segs = []
    
    breakpts = [0]+list(params)+[len(winding)]
    for ii, (a, b) in enumerate(zip(breakpts[:-1], breakpts[1:])):
        a = int(a)
        b = int(b)
        seg = np.array(winding[a:b])
        if ii%2:
            try:
                seg -= slope*np.arange(a, b)
            except:
                print(a, b, params, slope)
                print(seg)
                print((slope*np.arange(a, b)))
                raise Exception()
        seg -= np.mean(seg)
        segs.append(seg)

    return segs

#loss function for 4-breakpoint regression
def loss_multi(winding, params, slope, penalties):
    segs = get_segs(winding, params,slope)
    return np.sum([penalties[ii%2]*np.sum(seg**2) for ii,seg in enumerate(segs)])

# regression with 4 breakpoints
def multi_regression(preX, l, r):
    winding, s, c, q, dx = get_winding(preX)

    m, scores = median_slope(winding, 150, 250)
    pre, mid, post = get_premidpost(winding, (l, r), m)

    
    start = np.where(np.diff(np.sign(pre)))[0][-1]
    if preX.shape[0] - r:
        end = r+np.where(np.diff(np.sign(post)))[0][0]
    else:
        end = preX.shape[0]
    m, scores = median_slope(winding[start:end], 20, 30)

    n = len(winding)
    l = n // 2
    r = (3 * n) // 4
    parameters = np.array([l,l+(r-l)/3,l+2*(r-l)/3, r ])

    penalties = [1, 1.5]
    epsilon = 0.01
    gradient = np.zeros(4)
    delta = [*np.identity(4)]
    prev_grad = np.array(gradient)
    thresh = .3

    for i in range(10000):
        present = loss_multi(winding, parameters, m, penalties)
        if np.linalg.norm(gradient - prev_grad)< thresh and i > 0:
            break
        gradient = np.array([loss_multi(winding, parameters + d, m, penalties) - present for d in delta])
        parameters = parameters - epsilon * gradient
    return winding, m, parameters

#plot regression with 4 breakpoints
def plot_regression_multi(winding, params, slope, save = False, filename = ''):
    segs = get_segs(winding, params, slope)

    plt.plot(winding)
    breakpts = [0]+list(params)+[len(winding)]
    for ii, (a, b) in enumerate(zip(breakpts[:-1], breakpts[1:])):    
        a = int(a)
        b = int(b)        
        g = winding[a:b]
        plt.plot(range(a, b), g - segs[ii])
    if save:
        plt.savefig(filename + '.png')
        plt.close()
    else:
        plt.show()

'''
    plot_regression(winding, parameters, m, save = True, filename = 'fig/piecewise_disc/' + protid)

    with open('pickles/std_dev.pickle', 'wb') as handle:
        pickle.dump(std_dev, handle)
    with open('pickles/slope_d.pickle', 'wb') as handle:
        pickle.dump(slope_d, handle)
    with open('pickles/params_d.pickle', 'wb') as handle:
        pickle.dump(params_d, handle)       
'''