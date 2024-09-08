#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:29:55 2021

@author: stoffer
"""

import numpy as np
import scipy.stats as stats
import sys
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import time
import mkl
import numba
import os
import pickle
try:
    import cc3d
    def labelfnc(x,c):
        return cc3d.connected_components(x,connectivity=c)
except:
    from scipy import ndimage
    def labelfnc(x,c):
        if c==6:
            return ndimage.label(x)[0]
        elif c==26:
            return ndimage.label(x,structure=np.ones((3,3,3)))[0]

def tfce_vol(data, dims, mask, direction='pos', E=0.5, H=2.0, connectivity=6):
    indata = np.zeros(dims)
    indata[mask] = data
    out = tfce(indata, dh=None, E=E, H=H, direction = direction, connectivity=connectivity)
    return out

class PermTest():
    
    def __init__(self, M=None, C=None, group=None, doTFCE=False, dims=None, mask=None, block=None, savedir=None, direction='pos', Ttest=False):
        '''
        Constructor
        Parameters:
            M: ndarray
                Model TxK
            C: ndarray
                Contrast KxM
            group: ndarray
                Observation groupings (T)
            doTFCE: boolean
                Perform TFCE (needs mask)
           mask: ndarray
                Mask XxYxZ
           block: ndarray
                Blocking structure for permutations
        '''
        t0 = time.time()
        self.time = 0
        self.preptime = 0
        self.group = group
        self.M = M
        self.C = C
        self.doTFCE = doTFCE
        self.block = block
        self.mask = mask
        self.dims = dims
        
        self.i = 0
        self.ublocks = None
        self.perms = []
        if savedir is None:
            self.savedir = os.getcwd()
        else:
            self.savedir = savedir
        self.direction = direction
        self.Ttest = Ttest
        if not (self.M is None or self.C is None):
            self.prepare()
        self.preptime += time.time()-t0
        self.lock = multiprocessing.Lock()
    
    def getF(self):
        out = np.zeros(self.mask.shape)
        out[self.mask] = self.F
        return out

    def prepare(self):
        t0 = time.time()
        self.X,self.Z,(self.rnkX,self.rnkZ) = partitionDesign(self.M, self.C)
        self.order = np.arange(self.M.shape[0])
        if not self.block is None:
            self.ublocks = np.unique(self.block)
            blen = np.zeros(len(self.ublocks))
            for i,b in enumerate(self.ublocks):
                blen[i] = np.sum(self.block==b)
            assert np.all(blen==np.mean(blen))
        self.preptime += time.time()-t0

    def test(self, Y):
        t0 = time.time()
        self.affine = None
        if Y.shape[-3:] == self.mask.shape:
            Y = Y[:,self.mask]
        self.E = Y-self.Z@(self.Z.T@Y)
        self.Xo = self.X-self.Z@(self.Z.T@self.X)
        U,S,V = np.linalg.svd(self.Xo,full_matrices=False)
        self.Xo = U[:,:self.rnkX]
        self.EE = np.sum(self.E**2, axis=0)
        XoTE = self.Xo.T@self.E
        EE3 = np.sum((self.Xo@(XoTE))**2, axis=0)
        self.F = EE3/(self.rnkX/(self.M.shape[0]-self.rnkX-self.rnkZ))/(self.EE-EE3)
        if self.Ttest:
            self.F = np.sqrt(self.F)*np.sign(XoTE.reshape(self.F.shape))
        self.F_null = []
        if self.doTFCE:
            self.F_tfce = tfce_vol(self.F, self.dims, self.mask, direction=self.direction)
            self.F_tfce_null = []
        self.preptime += time.time()-t0
        sys.stdout.write(f'Preparation time: {self.preptime:>5.1f}s\n')
        sys.stdout.flush()
    
    def save_perm(self, filename):
        keys = ('Xo','E','EE','mask','order'
        'rnkX','rnkZ','block','ublocks',
        'group','order','dims','doTFCE')
        d = {}
        for key in keys:
            d[key]=getattr(self,key)
        np.savez(filename, d)

    def perm(self, seed=None):
        t0=time.time()
        if not seed is None:
            rnd = np.random.RandomState(seed)
        else:
            rnd = np.random.RandomState()
        P = self.order.copy()
        if self.group is None and self.block is None:
            P = rnd.permutation(np.arange(self.M.shape[0]))
        else:
            P = self.order.copy()
            if self.block is None:
                for g in np.unique(self.group):
                    msk = self.group==g
                    P[msk] = rnd.permutation(self.order[msk])
            else:
                i = 0
                for i,b in enumerate(rnd.permutation(self.ublocks)):
                    P[self.block==self.ublocks[i]] = self.order[self.block==b]
        U,S,V = np.linalg.svd(self.Xo[P], full_matrices=False)
        Xop = U[:,:self.rnkX]
        # XopTE = Xop.T@self.E
        # EE3p = np.sum((Xop@(XopTE))**2, axis=0)
        EE3p, XopTE = SSE(Xop,self.E)
        #assert(np.allclose(EE3p,EE3p2))
        Fp = EE3p/(self.rnkX/(self.M.shape[0]-self.rnkX-self.rnkZ))/(self.EE-EE3p)
        if self.Ttest:
            Fp = np.sqrt(Fp)*np.sign(XopTE.reshape(Fp.shape))
        self.time += time.time()-t0
        if self.doTFCE:
            t0=time.time()
            Fp_tfce = np.max(tfce_vol(Fp, self.dims, self.mask, direction=self.direction))
            self.time += time.time()-t0
            return Fp,Fp_tfce
        else:
            return Fp,None
        
    def p_vox(self):
        if self.Ttest:
            if self.direction == 'pos':
                self.p = stats.t.sf(self.F, self.M.shape[0]-self.rnkX-self.rnkZ)
            elif self.direction == 'both':
                self.p = 2*stats.t.sf(np.abs(self.F),self.M.shape[0]-self.rnkX-self.rnkZ)
        else:
            self.p = stats.f.sf(self.F,self.rnkX,self.M.shape[0]-self.rnkX-self.rnkZ)
        Fsort = np.sort(self.F_null, axis=0)
        idx = np.array([np.searchsorted(Fsort[:,i],self.F[i]) for i in range(self.F.shape[0])])
        self.pperm_vox = ((len(self.F_null)-idx)+1)/(len(self.F_null)+1)
        Fsort = np.sort(np.max(self.F_null, axis=1))
        idx = np.searchsorted(Fsort,self.F)
        self.pperm_max = ((len(self.F_null)-idx)+1)/(len(self.F_null)+1)
        if self.doTFCE:
            Fsort = np.sort(self.F_tfce_null)
            idx = np.searchsorted(Fsort,self.F_tfce)
            self.pperm_tfce = ((len(self.F_tfce_null)-idx)+1)/(len(self.F_tfce_null)+1)
        
    def perm_theshold(self, alpha=0.05):
        self.threshold = np.sort(np.max(self.F_null, axis=1))[int(len(self.F_null)*(1.0-alpha)+1)]
        if self.doTFCE:
            self.tfce_threshold = np.sort(self.F_tfce_null)[int(len(self.F_tfce_null)*(1.0-alpha)+1)]
        self.threshold_voxel = np.sort(self.F_null, axis=0)[int(len(self.F_null)*(1.0-alpha)+1)]
    
    def make_nii(self, filename = None, alpha=0.05):
        import nibabel as nib
        if self.affine is None:
            affine = np.identity(4)
        else:
            affine = self.affine
        self.perm_theshold(alpha=alpha)
        if not np.any(self.F_tfce>=self.tfce_threshold):
            print('no voxels above threshold')
        tfce_map=nib.Nifti1Image(self.F_tfce, affine=affine)
        if not filename is None:
            import nilearn.plotting
            import nilearn.datasets
            viz=nilearn.plotting.view_img(tfce_map,threshold=self.tfce_threshold,
                                          bg_img=nilearn.datasets.load_mni152_template(1))
            viz.save_as_html(filename)
        return tfce_map
    
    def make_nii_log10p(self, filename = None, alpha=0.05):
        import nibabel as nib
        self.p_vox()
        if self.affine is None:
            affine = np.identity(4)
        else:
            affine = self.affine
        tfce_map=nib.Nifti1Image(-np.log10(self.pperm_tfce), affine=affine)
        if not filename is None:
            import nilearn.plotting
            import nilearn.datasets
            viz=nilearn.plotting.view_img(tfce_map,threshold=-np.log10(alpha),
                                          bg_img=nilearn.datasets.load_mni152_template(1))
            viz.save_as_html(filename)
        return tfce_map
            
        
    def iterate(self, n=None,seeds=None):
        mkl.set_num_threads(1)
        i = 0
        if n is None:
            if seeds is None:
                n=1
            n=len(seeds)
        # N=self.i+n
        for i in range(n):
            t1=time.time()
            if seeds is None:
                seed=self.i
                out = self.perm(seed=seed)
            else:
                seed = seeds[i]
                out = self.perm(seed=seed)
            self.F_null.append(out[0])
            if self.doTFCE:
                self.F_tfce_null.append(out[1])
            self.perms.append(seed)
            self.i += 1
            i += 1
            self.time += time.time()-t1
            self.plot_status()
        sys.stdout.write('\n')
        sys.stdout.flush()

    def plot_status(self):
        sys.stdout.write(f'\rit#: {self.i:>5}\ttime: {self.time:>5.1f}s\tit/s: {self.i/self.time:>5.1f}')
        sys.stdout.flush()
    
    def addPerm(self, results):
        self.lock.acquire()
        for res in results:
            self.F_null.append(res[0])
            if self.doTFCE:
                self.F_tfce_null.append(res[1])
            self.i += 1
            self.time += time.time()-self.ptime
            self.ptime = time.time()
            self.plot_status()
            self.perms.append(res[-1])
        self.lock.release()
        
    def dispatch_jobs(self, seeds, ncpus=None):
        t0=time.time()
        with SharedMemoryManager() as smm:
            names = ('E','EE','Xo','mask')
            shm = {}
            shm_np = {}
            nbytes = {}
            dtypes = {}
            shm_names = {}
            shapes = {}
            
            for n in names:
                var = getattr(self, n)
                nbytes[n] = var.nbytes
                shapes[n] = var.shape
                dtypes[n] = var.dtype
                shm[n] = smm.SharedMemory(var.nbytes)
                shm_np[n] = np.frombuffer(dtype=var.dtype, buffer=shm[n].buf[:var.nbytes]).reshape(var.shape)
                np.copyto(shm_np[n], var)
                shm_names[n] = shm[n].name
            if ncpus is None:
                ncpus = multiprocessing.cpu_count()
            pool=multiprocessing.Pool(ncpus)
            preptime = time.time()-t0
            sys.stdout.write(f'Multiprocessing preparation time: {preptime:>5.1f}s\n')
            sys.stdout.flush()
            self.ptime=time.time()
            for s in seeds:
                # print(s)
                # runPerm((s,), (shm_names,dtypes,shapes,nbytes),
                #                         self.rnkX, self.rnkZ, self.block, 
                #                         self.ublocks, self.group, self.order, 
                #                         self.dims, self.doTFCE)
                pool.apply_async(runPerm, 
                                  args=((s,), (shm_names,dtypes,shapes,nbytes),
                                        self.rnkX, self.rnkZ, self.block, 
                                        self.ublocks, self.group, self.order, 
                                        self.dims, self.doTFCE, self.Ttest,
                                        self.direction), 
                                  callback=self.addPerm)
            pool.close()
            pool.join()
            del pool
            

            for n in names:
                del shm_np[n]
                shm[n].close()
                shm[n].unlink()
            sys.stdout.write('\n')
            sys.stdout.flush()

    def init_perm(self, filename):
        d = np.load(filename)
        keys = ('Xo','E','EE','mask','order'
        'rnkX','rnkZ','block','ublocks',
        'group','order','dims','doTFCE')
        for key in keys:
            setattr(self,key,d[key])
    
    def save(self, filename, savedata=True):
        with open(filename ,'wb') as fid:
            if savedata:
                pickle.dump(self,fid)
            else:
                import copy
                PT=copy.copy(self)
                del PT.E
                del PT.EE
                del PT.Xo
                pickle.dump(PT,fid)
        
    def remote_perm(self, seeds, njobs, ncpus=1, name=None):
        nj = np.min((np.ceil(len(seeds)/ncpus),njobs))
        if nj>1:
            self.ptime = time.time()
            if name is None:
                name = self.name
                if name is None:
                    import uuid
                    name=uuid.uuid4().hex
            outname=os.path.join(self.savedir,name+'_perm.npz')
            self.save_perm(outname)
            k = 0
            outnames = []
            for j in range(nj):
                n = np.ceil((len(seeds)-k)/nj)
                seedsj = seeds[k:k+n]
                k += n
                outnames.append(os.path.join(self.savedir,name + f'_{j}.npz'))
                script = os.path.abspath('permtests_parallel_numba.py')
                p = ' '.join([str(s) for s in seedsj])
                cmd = sys.executeable() + f' {script} --pfile {outname} --permutations {p}' +\
                    f' --ncpus={ncpus}' + f' --saveperm {outnames[-1]}'
                os.system(cmd)
            t0 = time.time()
            while len(outnames)>0 and (time.time()-t0)<=60*60*4:
                for i,o in enumerate(outnames):
                    if os.path.isfile(o):
                        d = np.load(o)
                        for out in d:
                            self.addperm(out)
                        outnames.pop(i)
                        os.unlink(out)
                time.sleep(10)
            os.unlink(outname)
        else:
            self.dispatch_jobs(seeds, ncpus=ncpus)
    def load(self,filename):
        with open(filename ,'rb') as fid:
            PT = pickle.load(fid)
        return PT


        
def runPerm(seeds, shared, rnkX, rnkZ, block, ublocks, group, order, dims, 
            doTFCE, Ttest, direction):
    mkl.set_num_threads(1)
    res = []
    shm_names, dtypes, shapes, nbytes = shared 
    names = ('E', 'EE', 'Xo', 'mask')
    v = {}
    shm = []
    for i,s in enumerate(names):
        shm.append(SharedMemory(shm_names[s]))
        v[s] = np.frombuffer(dtype=dtypes[s], 
                             buffer=shm[i].buf[:nbytes[s]]).reshape(shapes[s])
    
    for seed in seeds:
        if not seed is None:
            rnd = np.random.RandomState(seed)
        else:
            rnd = np.random.RandomState()
        if group is None and block is None:
            P = rnd.permutation(np.arange(v['Xo'].shape[0]))
        else:
            P = np.arange(v['Xo'].shape[0])
            if block is None:
                for g in np.unique(group):
                    msk = group==g
                    P[msk] = rnd.permutation(order[msk])
            else:
                i = 0
                for i,b in enumerate(rnd.permutation(ublocks)):
                    P[block==ublocks[i]] = order[block==b]
        U,S,V = np.linalg.svd(v['Xo'][P], full_matrices=False)
        Xop = U[:,:rnkX]
        #EE3p = np.sum((Xop@(Xop.T@v['E']))**2, axis=0)
        if Ttest:
            EE3p, XopTE = SSE(Xop,v['E'])
            Fp = EE3p/(rnkX/(v['Xo'].shape[0]-rnkX-rnkZ))/(v['EE']-EE3p)
            Fp = np.sqrt(Fp)*np.sign(XopTE.reshape(Fp.shape))
        else:
            EE3p, _ = SSE(Xop,v['E'])
            Fp = EE3p/(rnkX/(v['Xo'].shape[0]-rnkX-rnkZ))/(v['EE']-EE3p)
        if doTFCE:
            Fp_tfce = np.max(tfce_vol(Fp, dims, v['mask'], direction=direction))
            res.append((Fp,Fp_tfce,seed))
        else:
            res.append((Fp,seed))
    return res

@numba.njit
def SSE(Xop, E):
    #EE3p = np.sum((Xop@(Xop.T@E))**2, axis=0)
    XTE = np.dot(Xop.T,E)
    SSE = np.zeros(E.shape[1])
    for v in range(E.shape[1]):
        for t in range(Xop.shape[0]):
            for k in range(Xop.shape[1]):
                SSE[v] += (Xop[t,k] * XTE[k,v])**2
    return SSE, XTE
	
def partitionDesign(M,C,returnRz=False):
    '''
    Partitioning of input design matrix, M into effects of interest

    Parameters
    ----------
    M : ndarray (T x K)
        Design matrix
    C : ndarray (K x N)
        Contrast of interest
    returnRz: boolean
        If the residual forming matrix Rz should be returned or not
        default False

    Returns
    -------
    X : ndarray (T x rank(C))
        Partition of interest
    Zr : ndarray (T x rank(M)-rank(C))
        Partition of no interest (nuisance), orthogonalized by
    ranks : tuple (2)
        Tuple of rank(X) and rank(Zr)
    Rz : ndarray (TxT)
        residual forming matrix for nuisance (I-Zr@pinv(Zr))


    Partitioning according to:
    #https://www.sciencedirect.com/science/article/pii/S1053811914000913 Supplement &
    #Ridgway G.R. University College London; 2009.
    #Statistical Analysis for Longitudinal MR Imaging of Dementia. (Ph.D. thesis)
    '''

    T,K = M.shape
    # rnkC = np.linalg.matrix_rank(C)
    #contrast subspace
    X = M@np.linalg.pinv(C.T)
    rnkX = np.linalg.matrix_rank(X)
    #subtract contrast subspace
    Z = M-M@(C@np.linalg.pinv(C))
    #decompose to ensure full rank
    #(may not be needed for residual forming matrix, but makes matrix smaller)
    rnkZ = np.linalg.matrix_rank(Z)
    U,S,V = np.linalg.svd(Z,full_matrices=False)
    Zr = U[:,:rnkZ]
    if returnRz:
        Rz = np.identity(T) - U@U.T
        return X,Zr,(rnkX,rnkZ),Rz
    else:
        return X,Zr,(rnkX,rnkZ)

def tfce_fun(indata, outdata, dh=None, E=0.5, H=2.0, connectivity=6, neg = False, voxdims=None):
    if dh is None:
        dh=indata.max()/100.
        #print('dh=%.3f'%dh)
    if voxdims is None:
        voxdims=np.ones(len(indata.shape),dtype='float32')
    # step through data with dh increments
    for h in np.arange(0, indata.max(), dh):
        #only if there is some above threshold
        if np.max(indata)>=h:
            # threshold the data with current height
            mask = indata > h

            # connected components labelling
            l = labelfnc(mask,connectivity)[mask]

            #count size of clusters
            sizes = np.bincount(l).astype('float32')
            #sizes = bc(l)

            # modulate label size by voxel volume
            sizes *= np.prod(voxdims)

            # compute TFCE
            update_vals = h**H * dh * sizes[l]**E

            if neg:
                outdata[mask] -= update_vals
            else:
                outdata[mask] += update_vals

def tfce(indata, dh=None, E=0.5, H=2.0, direction = 'both', connectivity=6, voxdims=None):
    if voxdims is None:
        voxdims=np.ones(len(indata.shape),dtype='float32')
    outdata = np.zeros_like(indata, dtype=np.float32)


    if direction == "both" or direction == "pos":
        tfce_fun(indata, outdata, dh=dh, E=E, H=H, connectivity=connectivity, voxdims=voxdims)
    if direction == "both" or direction == "neg":
        indata *= -1.
        tfce_fun(indata, outdata, dh=dh, E=E, H=H, connectivity=connectivity, neg = True, voxdims=voxdims)
        indata *= -1.
    return outdata

def demo():
    T = 1000
    np.random.seed(42)
    dims = (64,64,42)
    x = np.random.randn(T,1)
    X = np.concatenate((x, np.ones((T,1))), axis=1)
    Y = np.random.randn(T, *dims)
    Y[:,22:33,22:33,20:25] += 0.05*x[:, None, None] * np.sign(Y[10,22:33,22:33,20:25])
    # Y = Y.reshape(T,-1)
    C = np.array((1,0))[None].T
    mask=np.zeros(dims,dtype='bool')
    mask[10:-10,10:-10, 10:-10] = True
    # mask[-10:,-10:, -10:] = True
    # PT1=PermTest(M=X,C=C,
    #          mask=mask,doTFCE=True,
    #          dims=dims,group=np.arange(1000)//2)
    # PT1.Ttest=True
    # PT1.direction='both'
    # PT1.test(Y)
    # PT1.iterate(seeds=np.arange(100))
    import permtests_parallel_numba as P
    PT2=P.PermTest(M=X,C=C,
             mask=mask,doTFCE=True,
             dims=dims,group=np.arange(1000)//10)
    PT2.Ttest=True
    # PT2.direction='pos'
    PT2.test(Y)
    PT2.dispatch_jobs(seeds=np.arange(100))
    PT2.make_nii(filename='test.html',alpha=0.05)
    # print(np.allclose(np.sort(PT1.F_tfce_null),np.sort(PT2.F_tfce_null)))
    # PT2.dispatch_jobs(seeds=np.arange(100,1100))
    #print(np.sort(PT1.F_tfce_null)-np.sort(PT2.F_tfce_null))
    #print(np.sort(PT1.perms)-np.sort(PT2.perms))
    # PT2.save('PT2.pickle',savedata=False)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--load',  '-l', type=str, required=False)
    parser.add_argument('--pfile', '-f', type=str, required=False)
    parser.add_argument('--name', '-n', type=str, required=False)
    parser.add_argument('--permutations', '-p', type=int, required=False, nargs="+")
    parser.add_argument('--parallel', '-pm', action='store_true')
    parser.add_argument('--id', '-i', type=str, required=False,default=None)
    parser.add_argument('--demo', '-d', action='store_true')
    
    args = parser.parse_args()
    args.demo = True
    if args.demo:
        demo()

    if args.pfile:
        PT=PermTest()
        PT.init_perm(args.pfile)
        output = []
        if parser.parallel>1:
            PT.dispatch_jobs(seeds=parser.seeds)
            for i in range(PT.perms):
                output.append((PT.F_null,PT.F_tfce_null,i))
        else:
            for i in parser.permutations:
                output.append(PT.perm(seed=i))
        np.savez(parser.saveperm, output)
