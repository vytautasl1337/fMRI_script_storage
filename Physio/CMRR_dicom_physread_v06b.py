#!/usr/bin/env python
# -*- coding: utf-8 -*-
versiontxt="""
CMRR dicom physiological recordings parser v. 0.5
06-05-2017 Kristoffer H. Madsen (kristofferm@drcmr.dk)

usage: %prog -i input [optional argument pairs]

v0.1  Initial version

v0.2  (14-08-2017):
     Speed up reading of INFO file
     Fix for interrupted sequences
     Fix for preliminary support of multiecho seqeunces

v0.2a (21-08-2017)
     Fallback to linear extrapolation for compatibility with older scipy/python
     Should now work in python 2.7

v0.3 (28-10-2018)
     Export of tabulator separated raw time series
     Use of newer pydicom package by default
     Added peak prominence functionality (requires scipy >=1.1)

v0.4 (10-12-2018)
     Add support for Philips SCANPHYSLOG

v0.5 (20/12-2019)
    Added file selector dialog box fallback
    Clean up plotting and PDF file reporting
    Added detection of too few peaks for respiration - should properly also be included for pulse
    
v0.6 (02/02-2021)
    Adding function to autodetect resp and pulse prominace based on data.
      this is now default (if no prominance is provided)
      it simply calculates the average signal amplitude in a frequency band
      from 0.01Hz to the stopband for the signal filter

v0.6a (02/02-2021)
    Added output of slicetimes to the mat file in seconds when using DCM input
    
v0.6b (23/04-2022)
    Fixed a path inconsistency when saving the pdf file, occuring when cardiac logging files were not available
"""

import numpy as np
try:
    import pydicom as dicom
except:
    import dicom
import os
import scipy.signal
try:
    scipy.signal.find_peaks
    scipyfp=True
except:
    scipyfp=False
import scipy.interpolate
import scipy.io
import matplotlib.pyplot as plt

def readInfo(data,explines=None):
    if explines is None:
        explines=len(data.split(b'\n'))
    header=[]
    info=dict()
    i=0
    arr=np.zeros((explines,4),dtype=np.uint32)
    for line in data.split(b'\n'):
        line=line.strip(b'\x00')
        if len(line)>0:
            if line.find(b'#')>0:
                line=line[:line.find(b'#')] #strip comments
            if line.find(b'=')>0:
                nam,val=line.split(b'=')
                info[nam.strip()]=val.strip()
            else:
                line=line.split()
                if not line[0].isdigit():
                    header=line
                else:
                    arr[i,0]=np.uint32(line[0])
                    arr[i,1]=np.uint32(line[2])
                    arr[i,2]=np.uint32(line[1])
                    if len(line)>3:
                            arr[i,3]=np.uint32(line[3])
                    i+=1
    return(arr[:i],header,info)

def readStuff(data,lablist=None,marklist=None,explines=None):
    if explines is None:
        explines=len(data.split(b'\n'))
    header=[]
    info=dict()
    i=0
    arr=np.zeros((explines,4),dtype=np.uint32)
    if lablist is None:
        lablist=[]
        warn1=False
    else:
        warn1=True
    if marklist is None:
        marklist=[]
        warn2=False
    else:
        warn2=True
    for line in data.split(b'\n'):
        line=line.strip(b'\x00')
        if len(line)>0:
            if line.find(b'#')>0:
                line=line[:line.find(b'#')] #strip comments
            if line.find(b'=')>0:
                nam,val=line.split(b'=')
                info[nam.strip()]=val.strip()
            else:
                line=line.split()
                if not line[0].isdigit():
                    header=line
                else:
                    try:
                        arr[i,0]=np.uint32(line[0])
                        arr[i,1]=np.uint32(line[2])
                        arr[i,2]=lablist.index(line[1])+1
                    except:
                        if warn1:
                            print('unexpected label {} appending as index {}'.format(line[1],len(lablist)+1))
                        lablist.append(line[1])
                        arr[i,2]=len(lablist)
                    if len(line)>3:
                        try:
                            arr[i,3]=marklist.index(line[3])+1
                        except:
                            if warn2:
                                print('unexpected marker {} appending as index {}'.format(line[3],len(marklist)+1))
                            marklist.append(line[3])
                            arr[i,3]=len(marklist)
                    i+=1
    return(arr[:i],lablist,header,info,marklist)

def ZPLPfilter(sig,fs,fpass=2,fstop=10, gpass=5,gstop=60):
    nqf=fs/2. 
    filter_b,filter_a=scipy.signal.iirdesign(float(fpass)/nqf,float(fstop)/nqf,gpass=gpass,gstop=gstop,analog=0, ftype="butter")
    fsig=scipy.signal.filtfilt(filter_b, filter_a, sig)
    return fsig

def getAmp(x, freqband, fs):
    xf = np.fft.fft(x)
    f = np.fft.fftfreq(x.shape[0],1./fs)
    return np.sqrt(np.sum(np.abs(xf[(f>=freqband[0])&(f<=freqband[1])])**2))/xf.shape[0]

def cleanTS(t,sig,ts,clip=(0,2**12-1),cliprem=(0,0),interp=2):
    valid=np.ones_like(t,dtype=np.bool)
    for c in clip:
        cclip=(sig!=c)
        for ci in range(cliprem[0],cliprem[1]+1):
            if ci<0:
                cclip[0:ci] |= cclip[-ci:]
            elif ci>0:
                cclip[ci:] |= cclip[:-ci]
        valid &= cclip
    #sigc=scipy.interpolate.griddata(t[valid],sig[valid],ts,method='linear',fill_value=np.nan)
    #sigc[np.isnan(sigc)]=scipy.interpolate.griddata(ts[~np.isnan(sigc)],sigc[~np.isnan(sigc)],ts[np.isnan(sigc)],method='nearest')
    #sigi=scipy.interpolate.interp1d(t[valid],sig[valid],kind=2,fill_value='extrapolate')
    #sigi=scipy.interpolate.UnivariateSpline(t[valid], sig[valid], w=None, bbox=[None, None], k=1, s=None, ext=0, check_finite=False)
    #sigi=scipy.interpolate.interp1d(t[valid],sig[valid],kind=2,fill_value='extrapolate')
    #sig[valid!=1]=sigi(t[valid!=1])
    try:
        sigi=scipy.interpolate.interp1d(t,sig,kind=interp,fill_value='extrapolate')
    except:
        print('Cubic extrapolation failed attempting with linear interpolation')
        sigi=scipy.interpolate.interp1d(t,sig,kind='linear',fill_value='extrapolate')
    sigc=sigi(ts)
    return sigc

def peaks(x,mindist=10):
    dx=x[1:]-x[:-1]
    idx=np.where((np.hstack((dx,0.))<=0.) & (np.hstack((0.,dx))>0.))[0]
    if idx[0]==0: idx=idx[1:]
    if idx[-1]==len(x): idx=idx[:-1]
    
    sidx=np.argsort(x[idx])[::-1]
    rem=np.zeros_like(idx,dtype=np.bool)
    
    for i in sidx:
        if not rem[i]:
            rem |= (idx >= (idx[i]-mindist)) & (idx <= (idx[i]+mindist))
            rem[i]=False #do not remove this peak    
    return idx[~rem]

def RETROICOR(slicetimes,peaks,n=1):
    if len(peaks)<2:
	     return
    mtx=np.zeros((len(slicetimes),n*2))
    for i,slt in enumerate(slicetimes):
        peakbefore=peaks[peaks<slt]
        peakafter=peaks[peaks>=slt]
        if len(peakbefore)==0:
            ipi=peaks[peaks>=slt][1]-peakafter[0]
            peakbefore=peakafter[0]-ipi*np.ceil((peakafter[0]-slt)/ipi)
        else:
            peakbefore=peakbefore[-1]
            
        if len(peakafter)==0:
            ipi=peakbefore-peaks[peaks<slt][-2]
            peakafter=peakbefore+ipi*np.ceil(((slt-peakbefore)/ipi))
        else:
            peakafter=peakafter[0]
        dt=slt-peakbefore
        ipi=peakafter-peakbefore
        ph=2*np.pi*dt/ipi
        for j in range(n):
            mtx[i,j*2]=np.sin((j+1)*ph)
            mtx[i,j*2+1]=np.cos((j+1)*ph)
    return(mtx)
    
def plot_phys(ts, sigs, peaks=None, names=None, txt=None, 
              secsperplot=50., pdffile=None, interpfs=100.,
              ltypes=('k','r--'),ptype=('b.','bx'), 
              ltxt=('Raw signal','rectified & filtered'),
              ptxt=('detected peaks','original peaks'),
              peaksig=(1, 0)):
    t=ts[0]
    nplots = int(np.ceil((t.max()-t.min())/secsperplot))
    ii = 0
    intp1 = []
    for k,sig in enumerate(sigs):
        intp1.append(scipy.interpolate.interp1d(ts[k], sigs[k], kind='nearest', fill_value=np.nan, bounds_error=False))
        
    for i in range(int(np.ceil(nplots/3.))):
        fig = plt.figure()
        if i*3+3>nplots:
            nsub = nplots - i * 3
        else:
            nsub = 3    
        for j in range(nsub):
            plt.subplot(3,1,j+1)
            tstart = t[ii]
            tend = tstart + secsperplot
            iiend = ii + np.argmax(np.nonzero(t[ii:] <= tend)[0])
            if not interpfs is None:
                tt = np.arange(tstart, tend, 1./interpfs)
                
            for k,sig in enumerate(sigs):
                if not interpfs is None:
                    sig = intp1[k](tt)    
                else:
                    tt = ts[k][ii:iiend]
                    sig = sig[ii:iiend]
                plt.plot(tt, sig, ltypes[k], label=ltxt[k])
            if not peaks is None and len(peaks) > 0:
                for k, p in enumerate(peaks):
                    idx2 = (p >= tstart) & (p<tend)
                    plt.plot(p[idx2], intp1[peaksig[k]](p[idx2]), ptype[k], label=ptxt[k])
            plt.title(txt)
            plt.xlabel('time [s]')

            if j == (nsub-1) and i == int(np.ceil(nplots/3.)-1):
                plt.legend(bbox_to_anchor=(0, -0.2), loc=2, borderaxespad=0.)
            ii=iiend
        if not pdffile is None:
            pdffile.savefig(fig)
            plt.close()
    if not peaks is None and len(peaks) > 0:
        fig=plt.figure()
        for k, p in enumerate(peaks):
            plt.subplot(len(peaks),1,k+1)
            plt.plot(p[:-1]+0.5*np.diff(p),60./np.diff(p), 'ko-')
            plt.title('Detected %s cycles per minute (%s)'%(txt,ptxt[k]))
            plt.xlabel('time [s]')
        if not pdffile is None:
            pdffile.savefig(fig)
            # print(pdffile)
            plt.close()


def parseScanPhysLog(fn,TR,N,resporder=3,pulseorder=2,fs=500.,makeplot=True,outname='out',\
             outdir=None,niiname=None,mindistr=2.0,mindistp=0.5,prefix='f_',\
             pulsepass=2.0,pulsestop=10.0,resppass=1.0,\
             respstop=5.0,writeF=True,filternii=True,rawtsv=False,\
             respprom=None,pulseprom=None):
    data = np.loadtxt(fn, usecols=range(9), dtype=np.float)
    labels = np.loadtxt(fn, usecols=9, dtype='bytes')
    labels = np.array([int(i,16) for i in labels],dtype=np.uint16)
    labels = np.unpackbits(labels.view(np.uint8)).reshape(-1,2,8)[:,:,::-1].reshape(-1,16)
    markers = [np.nonzero(labels[:,i])[0] for i in range(16)]
    pulsepeakso = markers[1]
    resppeakso = markers[2]
    
    pulse = data[:, 4]
    resp = data[:, 5]

    if len(markers[3])>0:
        print('%i measurement markers found'%len(markers[3]))
        dt = np.diff(markers[3])
        sdt = np.std(dt/fs)
        print('spacing between markers %fms std. %fms [%i-%i samples]'%(dt.mean()/fs,sdt,np.min(dt),np.max(dt)))

    
    if len(markers[10])>0:
        print('%i volume markers found'%(len(markers[8])))
        dt=np.diff(markers[8])
        sdt=np.std(dt/fs)
        print('spacing between markers %fms std. %fms [%i-%i samples]'%(dt.mean()/fs,sdt,np.min(dt),np.max(dt)))
    
    if len(markers[9]) > 0:
        print('%i volume trigger markers found'%(len(markers[7])))
        dt = np.diff(markers[7])
        sdt = np.std(dt/fs)
        print('spacing between markers %fms std. %fms [%i-%i samples]'%(dt.mean()/fs,sdt,np.min(dt),np.max(dt)))        
        TRsamp = (dt[-1]-dt[0])/(len(dt)-1)
        fs = TRsamp/TR
        N = len(dt)       
        print('using timing for triggers markers - overriding sampling rate to %.1fHz and nvols to %i'%(fs,N))
        t = np.arange(0, len(pulse)) / fs
        t -= t[markers[9][0]]
    else:
        print('no volume trigger markers found - using scan end marker "0020" and assuming %.1fHz sampling rate'%fs)
        t = np.arange(0, len(pulse)) / fs
        # print(markers[5])
        # print(t[markers[5][-1]])
        # print(N*TR)
        t -= t[markers[5][-1]]
        t += N*TR
        print('Log start time %.1fs (%.1f TRs) - Log end time: %.1fs (%.1f TRs)'%(t[0],t[0]/TR,t[-1],t[-1]/TR))

    nsamp = markers[5][-1]-markers[4][-1]
    print('%i samples - %.3f s between last start and end markers'%(nsamp, nsamp/fs))
    
    try:
        nsamp1 = markers[5][-1]-markers[8][-1]
        print('%i samples - %.3f s between last man. start and end markers'%(nsamp1, nsamp1/fs))
    except:
       pass
        
    voltime=np.arange(0,N*TR,TR) + 0.5*TR
    if rawtsv:
        raw=list()
        raw.append(t-voltime[0])
        rawhdr='t'
        rawfmt = ('%.4f',)
    matlabdata=dict()
    print('Processing pulseoximetry')
    p=cleanTS(t,pulse,t,(-np.inf,np.inf))
    if rawtsv:
        raw.append(p)
        rawhdr += '\tpulse'
        rawfmt += ('%i',)
    pf=ZPLPfilter(p,fs,pulsepass,pulsestop)
    if pulseprom is None:
        pulseprom = 0.5 * getAmp(pf,[0.01, pulsestop],fs)
        print('Setting pulse prominace to %.0f'%pulseprom)
        
    if not pulseprom is None and pulseprom>0:
        if scipyfp:
            pulsepeaks=scipy.signal.find_peaks(pf,distance=int(mindistp*fs),prominence=pulseprom)[0]
        else:
            print('scipy.signal.find_peaks is not availible please upgrade to scipy 1.1.0 or higher to use peak prominence function')
    else:
        pulsepeaks=peaks(pf,mindist=int(mindistp*fs))
    if len(pulsepeaks)>0 and pulseorder>0:
        Xp=RETROICOR(voltime,t[pulsepeaks],pulseorder)
        Xp -= np.mean(Xp,0)[np.newaxis,:]
        Xp /= np.std(Xp,0)[np.newaxis,:]
        matlabdata['R']=Xp
    if makeplot:
        from matplotlib.backends.backend_pdf import PdfPages
        pdffile = PdfPages(os.path.join(outdir,outname+'.pdf'))
        plot_phys((t,t),(pulse,pf),
                   (t[pulsepeaks],t[pulsepeakso]),
                   txt='cardiac',pdffile=pdffile)

    print('Processing respiration')
    r=cleanTS(t,resp,t,(-np.inf,np.inf))
    if rawtsv:
        raw.append(r)
        rawhdr += '\tresp'
        rawfmt += ('%i',)
    rf=ZPLPfilter(r,fs,resppass,respstop)
    if respprom is None:
        respprom = 0.5 * getAmp(rf,[0.01, respstop],fs)
        print('Setting resp prominace to %.0f'%respprom)
    if not respprom is None and respprom>0:
        if scipyfp:
            resppeaks=scipy.signal.find_peaks(rf,distance=int(mindistr*fs),prominence=respprom)[0]
        else:
            print('scipy.signal.find_peaks is not availible please upgrade to scipy 1.1.0 or higher to use peak prominence function')
    else:
        resppeaks=peaks(rf,int(mindistr*fs))
    if len(resppeaks)>0 and resporder>0:
        Xr=RETROICOR(voltime,t[resppeaks],resporder)
        Xr-=np.mean(Xr,0)[np.newaxis,:]
        Xr/=np.std(Xr,0)[np.newaxis,:]
        try:
            matlabdata['R']=np.hstack((matlabdata['R'],Xr))
        except:
            matlabdata['R']=Xr
    if makeplot:
        plot_phys((t,t),(resp,rf),
                   (t[resppeaks],t[resppeakso]),
                   txt='respiration',pdffile=pdffile)
        
        
    try:
        pdffile.close()
    except:
        pass
        
    try:
        matlabdata['R']
        scipy.io.savemat(os.path.join(outdir,outname+'.mat'),matlabdata)
    except:
        pass
    if rawtsv:
            np.savetxt(os.path.join(outdir,outname+'.tsv'),np.array(raw).T,delimiter='\t',header=rawhdr,fmt=rawfmt)
    if not niiname is None:
        import nibabel as nib
        datavols=nib.load(niiname)
        outnii=os.path.join(os.path.dirname(niiname),prefix+os.path.basename(niiname))
        dataout=np.empty(datavols.shape,dtype=np.float32,order=datavols.dataobj.order)
        if writeF:
            pout=np.empty(datavols.shape[:3],dtype=np.float32,order=datavols.dataobj.order)
            rout=np.empty(datavols.shape[:3],dtype=np.float32,order=datavols.dataobj.order)
        #outhd=datavols.get_header().copy()
        S=dataout.shape[-2]
        T=dataout.shape[-1]
        for s in range(S):
            voltimet=np.arange(0, N*TR, TR) + s*(TR/S)
            Xrt=RETROICOR(voltimet,t[resppeaks],resporder)
            Xrt-=np.mean(Xrt,0)[np.newaxis,:]
            Xrt/=np.std(Xrt,0)[np.newaxis,:]
            Xpt=RETROICOR(voltimet,t[pulsepeaks],pulseorder)
            Xpt-=np.mean(Xpt,0)[np.newaxis,:]
            Xpt/=np.std(Xpt,0)[np.newaxis,:]
            Xt=np.hstack((Xrt,Xpt,np.ones((Xrt.shape[0],1))))
            Xrt=np.hstack((Xrt,np.ones((Xrt.shape[0],1))))
            Xpt=np.hstack((Xpt,np.ones((Xpt.shape[0],1))))
            yr = datavols.dataobj[...,s,:].reshape(-1,len(voltimet)).T
            if filternii:
                beta=np.linalg.lstsq(Xt,yr,rcond=None)[0]
                dataout[...,s,:]=(yr-Xt.dot(beta))\
                     .reshape(datavols.shape[0],datavols.shape[1],T)
                dataout[...,s,:]+= beta[-1,:].reshape(datavols.shape[0],datavols.shape[1],1)
            if writeF:
                dofr=yr.shape[0]-np.linalg.matrix_rank(Xt)
                eps=np.spacing(1)
                RSS   = np.sum(((np.identity(T)-Xt.dot(np.linalg.pinv(Xt.T.dot(Xt)).dot(Xt.T))).dot(yr))**2,0)
                RSS_r = np.sum(((np.identity(T)-Xpt.dot(np.linalg.pinv(Xpt.T.dot(Xpt)).dot(Xpt.T))).dot(yr))**2,0)
                RSS_p = np.sum(((np.identity(T)-Xrt.dot(np.linalg.pinv(Xrt.T.dot(Xrt)).dot(Xrt.T))).dot(yr))**2,0)
                dof_p = np.linalg.matrix_rank(Xt)-np.linalg.matrix_rank(Xrt)
                dof_r = np.linalg.matrix_rank(Xt)-np.linalg.matrix_rank(Xpt)
                RSS_p = dofr/dof_p*((RSS_p-RSS+eps)/(RSS+eps))
                RSS_r = dofr/dof_r*((RSS_r-RSS+eps)/(RSS+eps))
                pout[...,s]=RSS_p.reshape(datavols.shape[0],datavols.shape[1])
                rout[...,s]=RSS_r.reshape(datavols.shape[0],datavols.shape[1])
                print('Slice %i of %i'%(s+1,S))
            
            
        if writeF:
            outFr=nib.Nifti1Image(rout,datavols.affine)
            outFr.get_header().set_intent('f test',(dof_r,dofr))
            outFr.to_filename(os.path.join(outdir,'resp_'+prefix+os.path.basename(niiname)))
            outFp=nib.Nifti1Image(pout,datavols.affine)
            outFp.get_header().set_intent('f test',(dof_p,dofr))
            outFp.to_filename(os.path.join(outdir,'pulse_'+prefix+os.path.basename(niiname)))
        if filternii:
            outvols=nib.Nifti1Image(dataout,datavols.affine)
            outvols.to_filename(outnii)
    return    

def parseDCM(fn,resporder=3,pulseorder=2,fs=400.,makeplot=True,outname='out',\
             outdir=None,niiname=None,mindistr=2.0,mindistp=0.5,prefix='f_',\
             pulsepass=2.0,pulsestop=10.0,resppass=1.0,\
             respstop=5.0,writeF=True,filternii=True,rawtsv=False,\
             respprom=None,pulseprom=None):
    d = dicom.read_file(fn)
    try:
        data = d[0x7fe1, 0x1010].value
    except:
       raise Exception('Dicom key [0x7fe1, 0x1010] not found in specified file, are you sure this is a CMRR dicom physiological logging file?') 
    rows = np.int(d.get('AcquisitionNumber'))
    cols = len(data)//rows
    numFiles = cols//(2**10)
    flength = len(data)//numFiles
    foundECG  = 0
    foundRESP = 0
    foundPULSE = 0
    foundEXT  = 0
    for fileno in range(numFiles):
        f=data[fileno*flength:(fileno+1)*flength]
        flen=np.array(f[:4]).view('<i4')
        fnlen=(np.array(f[4:8]).view('<i4'))
        fnname=f[8:8+fnlen]
        #print('Found %s'%fnname)
        logData=f[1025:flen+1025]
        if fnname[-9:]==b'_Info.log':
            fnINFO = logData
        elif fnname[-8:]==b'_ECG.log':
            fnECG  = logData
            foundECG = 1
        elif fnname[-9:]==b'_RESP.log':
            fnRESP = logData
            foundRESP = 1
        elif fnname[-9:]==b'_PULS.log':
            fnPULSE = logData
            foundPULSE = 1
        elif fnname[-8:]==b'_EXT.log':
            fnEXT = logData
            foundEXT = 1
    (sliceinfo,header,info)=readInfo(fnINFO)
    FirstTime=np.uint32(info[b'FirstTime'])
    # print(FirstTime)
    LastTime=np.uint32(info[b'LastTime'])
    # print(LastTime)
    sliceinfo[:,(1,3)]-=FirstTime
    slicetimes=np.mean(sliceinfo[:,(1,3)],1)
    #print(slicetimes.shape)
    #print(slicetimes)
    #print(sliceinfo)
    noSamp=LastTime-FirstTime+1
    # print(noSamp)
    t=np.arange(0,noSamp,1.)/fs
    #expSamp=noSamp+8
    st=t[[int(slicetimes[(sliceinfo[:,0]==0) & (sliceinfo[:,2]==s)][0]) for s in range(np.max(sliceinfo[:,2]))]]
    st-=st.min()
    sliceno=sliceinfo[int(np.ceil(np.mean(sliceinfo[:,2]))),2]
    if not np.any((sliceinfo[:,2]==sliceno) & (sliceinfo[:,0]==np.max(sliceinfo[:,0]))):
        maxvol=np.max(sliceinfo[:,0])-1
        print('Volume %i appears to be incomplete - using %i volumes'%(maxvol+2,maxvol+1))        
    else:
        maxvol=np.max(sliceinfo[:,0])
    nechos=int(((sliceinfo[:,0]==0)&((sliceinfo[:,2]==0))).sum())
    if nechos>1:
        print('Multiecho sequence detected (%i echos) using time for first echo for regressors'%(nechos,))
    voltime=t[np.array([int(slicetimes[(sliceinfo[:,0]==i) & (sliceinfo[:,2]==sliceno)][0]) for i in range(maxvol+1)]).ravel()]
    print('%i Volumes detected.'%voltime.shape[0])
    print('Average TR: %gms '%(np.mean(np.diff(voltime))*1000.,))
    if rawtsv:
        raw=list()
        raw.append(t-voltime[0])
        rawhdr='t'
        rawfmt = ('%.4f',)
    matlabdata=dict()
    matlabdata['slicetimes']=st
    if foundPULSE:
        print('Processing pulseoximetry')
        (pulse,plist,header,pinfo,pmlist)=readStuff(fnPULSE,[b'PULS'],[b'PULS_TRIGGER',b'RESP_TRIGGER',b'ECG_TRIGGER',b'EXT1_TRIGGER'])
        pulse[:,0]-=FirstTime
        # print(pulse)
        pulsepeakso=np.nonzero(pulse[:,3]==1)[0]
        p=cleanTS(pulse[pulse[:,2]==1,0]/fs,pulse[pulse[:,2]==1,1],t)
        if rawtsv:
            raw.append(p)
            rawhdr += '\tpulse'
            rawfmt += ('%i',)
        pf=ZPLPfilter(p,fs,pulsepass,pulsestop)
        if pulseprom is None:
            pulseprom = 0.5 * getAmp(pf,[0.01, pulsestop],fs)
            print('Setting pulse prominace to %.0f'%pulseprom)
        if not pulseprom is None and pulseprom>0:
            if scipyfp:
                pulsepeaks=scipy.signal.find_peaks(pf,distance=int(mindistp*fs),prominence=pulseprom)[0]
            else:
                print('scipy.signal.find_peaks is not availible please upgrade to scipy 1.1.0 or higher to use peak prominence function')
        else:
            pulsepeaks=peaks(pf,mindist=int(mindistp*fs))
        if len(pulsepeaks)>0 and pulseorder>0:
            Xp=RETROICOR(voltime,t[pulsepeaks],pulseorder)
            Xp -= np.mean(Xp,0)[np.newaxis,:]
            Xp /= np.std(Xp,0)[np.newaxis,:]
            matlabdata['R']=Xp
        
        if makeplot:
            from matplotlib.backends.backend_pdf import PdfPages
            pdffile = PdfPages(os.path.join(outdir,outname+'.pdf'))
            plot_phys((pulse[:,0]/fs,t),(pulse[:,1],pf),
                                 (t[pulsepeaks],pulse[pulsepeakso,0]/fs),
                                 txt='cardiac',pdffile=pdffile)

    if foundRESP:
        print('Processing respiration')
        (resp,rlist,header,rinfo,rmlist)=readStuff(fnRESP,[b'RESP'],[b'PULS_TRIGGER',b'RESP_TRIGGER',b'ECG_TRIGGER',b'EXT1_TRIGGER'])
        resp[:,0]-=FirstTime
        # print(resp)
        resppeakso=np.nonzero(resp[:,3]==2)[0]
        #validr=(resp[:,2]==1) & (resp[:,1]!=0) & (resp[:,1]!=2**12-1)
        #respi=scipy.interpolate.interp1d(resp[validr,0]/fs,resp[validr,1],kind='cubic',fill_value='extrapolate')
        #r=respi(t)
        r=cleanTS(resp[resp[:,2]==1,0]/fs,resp[resp[:,2]==1,1],t)
        if rawtsv:
            raw.append(r)
            rawhdr += '\tresp'
            rawfmt += ('%i',)
        rf=ZPLPfilter(r,fs,resppass,respstop)
        if respprom is None:
            respprom = 0.5 * getAmp(rf,[0.01, respstop],fs)
            print('Setting resp prominace to %.0f'%respprom)
        if not respprom is None and respprom>0:
            if scipyfp:
                resppeaks=scipy.signal.find_peaks(rf,distance=int(mindistr*fs),prominence=respprom)[0]
            else:
                print('scipy.signal.find_peaks is not availible please upgrade to scipy 1.1.0 or higher to use peak prominence function')
        else:
            resppeaks=peaks(rf,int(mindistr*fs))
        if len(resppeaks)<2:
            print('Too few respiration peaks try changing respiration peak prominance.')
        else:    
       
            if len(resppeaks)>0 and resporder>0:
                Xr=RETROICOR(voltime,t[resppeaks],resporder)
                Xr-=np.mean(Xr,0)[np.newaxis,:]
                Xr/=np.std(Xr,0)[np.newaxis,:]
                try:
                    matlabdata['R']=np.hstack((matlabdata['R'],Xr))
                except:
                    matlabdata['R']=Xr
        if makeplot:
            try:
                pdffile
            except:
                from matplotlib.backends.backend_pdf import PdfPages
                pdffile = PdfPages(os.path.join(outdir, outname+'.pdf'))
            plot_phys((resp[:,0]/fs,t),(resp[:,1],rf),
                                 (t[resppeaks],resp[resppeakso,0]/fs),
                                 txt='respiration',pdffile=pdffile)
    try:
        pdffile.close()
    except:
        pass
    
    if foundECG:
        print('Processing ECG')
        (ecg,elist,header,einfo,emlist)=readStuff(fnECG,[b'ECG1',b'ECG2',b'ECG3',b'ECG4'],[b'PULS_TRIGGER',b'RESP_TRIGGER',b'ECG_TRIGGER',b'EXT1_TRIGGER'])
        ecg[:,0]-=FirstTime
        #if len(ecgpeaks)>0:
        #    Xe=RETROICOR(slicetimes,ecg[ecgpeaks,0],2)
        e=np.empty((4,len(t)))
        for i in range(4):
            e[i,:]=np.interp(t,ecg[ecg[:,2]==i+1,0]/fs,ecg[ecg[:,2]==i+1,1])
            if rawtsv:
                raw.append(e[i,:])
                rawhdr += '\tECG%i'%i
                rawfmt += ('%i',)
    if foundEXT:
        print('Processing external channels')
        (ext,extlist,header,extinfo,extmlist)=readStuff(fnEXT,[b'EXT'],[b'PULS_TRIGGER',b'RESP_TRIGGER',b'ECG_TRIGGER',b'EXT1_TRIGGER'])
        ext[:,0]-=FirstTime
        extsamp=np.arange(0,LastTime-FirstTime,np.int(extinfo[b'SampleTime']))
        extmiss=extsamp[np.min(np.abs(ext[:,0,np.newaxis]-extsamp),0)>=np.int(extinfo[b'SampleTime'])//2]
        idx=np.argsort(np.hstack((extmiss,ext[:,0])))
        exttime=np.hstack((extmiss,ext[:,0]))[idx]
        extsig=np.hstack((np.zeros_like(extmiss),ext[:,1]))[idx]
    try:
        matlabdata['R']
        scipy.io.savemat(os.path.join(outdir,outname+'.mat'),matlabdata)
    except:
        pass
    if rawtsv:
            np.savetxt(os.path.join(outdir,outname+'.tsv'),np.array(raw).T,delimiter='\t',header=rawhdr,fmt=rawfmt)
    if not niiname is None:
        import nibabel as nib
        datavols=nib.load(niiname)
        outnii=os.path.join(os.path.dirname(niiname),prefix+os.path.basename(niiname))
        dataout=np.empty(datavols.shape,dtype=np.float32,order=datavols.dataobj.order)
        if writeF:
            pout=np.empty(datavols.shape[:3],dtype=np.float32,order=datavols.dataobj.order)
            rout=np.empty(datavols.shape[:3],dtype=np.float32,order=datavols.dataobj.order)
        #outhd=datavols.get_header().copy()
        S=dataout.shape[-2]
        T=dataout.shape[-1]
        for s in range(S):
            voltimet=t[[int(slicetimes[(sliceinfo[:,0]==i) & (sliceinfo[:,2]==s)][0]) for i in range(T)]]
            Xrt=RETROICOR(voltimet,t[resppeaks],resporder)
            Xrt-=np.mean(Xrt,0)[np.newaxis,:]
            Xrt/=np.std(Xrt,0)[np.newaxis,:]
            Xpt=RETROICOR(voltimet,t[pulsepeaks],pulseorder)
            Xpt-=np.mean(Xpt,0)[np.newaxis,:]
            Xpt/=np.std(Xpt,0)[np.newaxis,:]
            Xt=np.hstack((Xrt,Xpt,np.ones((Xrt.shape[0],1))))
            Xrt=np.hstack((Xrt,np.ones((Xrt.shape[0],1))))
            Xpt=np.hstack((Xpt,np.ones((Xpt.shape[0],1))))
            yr = datavols.dataobj[...,s,:].reshape(-1,len(voltimet)).T
            if filternii:
                beta=np.linalg.lstsq(Xt,yr,rcond=None)[0]
                dataout[...,s,:]=(yr-Xt.dot(beta))\
                     .reshape(datavols.shape[0],datavols.shape[1],T)
                dataout[...,s,:]+= beta[-1,:].reshape(datavols.shape[0],datavols.shape[1],1)
            if writeF:
                dofr=yr.shape[0]-np.linalg.matrix_rank(Xt)
                eps=np.spacing(1)
                RSS   = np.sum(((np.identity(T)-Xt.dot(np.linalg.pinv(Xt.T.dot(Xt)).dot(Xt.T))).dot(yr))**2,0)
                RSS_r = np.sum(((np.identity(T)-Xpt.dot(np.linalg.pinv(Xpt.T.dot(Xpt)).dot(Xpt.T))).dot(yr))**2,0)
                RSS_p = np.sum(((np.identity(T)-Xrt.dot(np.linalg.pinv(Xrt.T.dot(Xrt)).dot(Xrt.T))).dot(yr))**2,0)
                dof_p = np.linalg.matrix_rank(Xt)-np.linalg.matrix_rank(Xrt)
                dof_r = np.linalg.matrix_rank(Xt)-np.linalg.matrix_rank(Xpt)
                RSS_p = dofr/dof_p*((RSS_p-RSS+eps)/(RSS+eps))
                RSS_r = dofr/dof_r*((RSS_r-RSS+eps)/(RSS+eps))
                pout[...,s]=RSS_p.reshape(datavols.shape[0],datavols.shape[1])
                rout[...,s]=RSS_r.reshape(datavols.shape[0],datavols.shape[1])
                print('Slice %i of %i'%(s+1,S))
            
            
        if writeF:
            outFr=nib.Nifti1Image(rout,datavols.affine)
            outFr.get_header().set_intent('f test',(dof_r,dofr))
            outFr.to_filename(os.path.join(outdir,'resp_'+prefix+os.path.basename(niiname)))
            outFp=nib.Nifti1Image(pout,datavols.affine)
            outFp.get_header().set_intent('f test',(dof_p,dofr))
            outFp.to_filename(os.path.join(outdir,'pulse_'+prefix+os.path.basename(niiname)))
        if filternii:
            outvols=nib.Nifti1Image(dataout,datavols.affine)
            outvols.to_filename(outnii)
    return

if __name__=="__main__":
    from optparse import OptionParser
    parser = OptionParser(usage=versiontxt)
    parser.add_option("-i", "--input", dest="file",help="dicom input file")
    parser.add_option("-P", "--pulseorder",dest="pulseorder",default=3,help="Pulse oximetry RETROICOR expansion order (default: 3)")
    parser.add_option("-R", "--resporder",dest="resporder",default=2,help="Respiration RETROICOR expansion order (default: 2)")
    parser.add_option("-f", "--sampleRate",dest="fs",default=None,help="TICS frequency in Hz (default: 400 Hz for Siemens, 500Hz for Philips)")
    parser.add_option("-b", "--pulsepass",dest="pulsepass",default=2.0,help="Pulse oximetry filter passband (default: 2 Hz)")
    parser.add_option("-c", "--pulsestop",dest="pulsestop",default=10.0,help="Pulse oximetry filter stopband (default: 10 Hz)")
    parser.add_option("-j", "--resppass",dest="resppass",default=1.0,help="Respiration filter passband (default: 1 Hz)")
    parser.add_option("-k", "--respstop",dest="respstop",default=5.0,help="Respiration filter stopband (default: 5 Hz)")
    parser.add_option("-v", "--pulseprom",dest="pulseprom",default=None,help="Min prominence for pulse peaks (default: Auto detect)")
    parser.add_option("-w", "--respprom",dest="respprom",default=None,help="Min prominence for resp peaks (default: Auto detect)")
    parser.add_option("-t", "--rawtsv",dest="rawtsv",default=False,help="Write tsv files with raw signals time column is relative to first volume (default: False)")
    parser.add_option("-o", "--outdir",dest="outdir",default=None,help="Output directory, default to same as dicom input")
    parser.add_option("-n", "--outname",dest="outname",default=None,help="Name of output files, default to dicom filename with phys_ prefix.")
    parser.add_option("-O", "--mindist_pulse",dest="mindistp",default=0.5,help="Minimum peak distance in seconds for pulse oximetry.")
    parser.add_option("-r", "--mindist_resp",dest="mindistr",default=2.,help="Minimum peak distance in seconds for respiration.")
    parser.add_option("-d", "--data4D",dest="data",default=None,help="4D times series nifti file for filtering or generation of F-tests.")
    parser.add_option("-l", "--filter",dest="filternii",default=True,help="Filter 4D nifti file if defined (1/0), default: 1. Filtered file will be placed in same directory as input nifti.")
    parser.add_option("-p",   "--plot"  ,dest="makeplot",default=True,help="Generate plots, default: 1")
    parser.add_option("-F",   "--Ftest"  ,dest="writeF",default=True,help="Generate F-tests if 4D nifti is defined (1/0), default: 1")
    parser.add_option("-x",   "--prefix"  ,dest="prefix",default="physfilt_",help="prefix for filtered 4D nifti, default physfilt_")
    parser.add_option("-a", "--TR",dest="TR",default=None,help="Volume repetition time (only relevant for Philips SCANPHYSLOG)")
    parser.add_option("-s", "--nvols",dest="nvols",default=None,help="Number of volumes (only relevant for Philips SCANPHYSLOG)")
    parser.add_option("-y", "--SPL",dest="forceScanPhyslog",default=0,help="Force Philips SCANPHYSLOG")
    #parser.add_option("-z", "--fsl",dest="fsl",default=None,help="FSL compatible output filename")
    
    (options, args) = parser.parse_args()
    #(path,ext)=os.path.splitext(options.file)
    pulseorder=int(options.pulseorder)
    resporder=int(options.resporder)
    mindistp=float(options.mindistp)
    mindistr=float(options.mindistr)
    
    forceScanPhyslog=bool(options.forceScanPhyslog)
    
    pulsepass=float(options.pulsepass)
    pulsestop=float(options.pulsestop)
    resppass=float(options.resppass)
    respstop=float(options.respstop)
    if not options.respprom is None:
        options.respprom=float(options.respprom)
    if not options.pulseprom is None:
        options.pulseprom=float(options.pulseprom)
    
    if options.file is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        options.file = filedialog.askopenfilename()
        print('Selected input file: %s'%options.file)
            
    if options.outname is None:
        outname='phys_'+os.path.splitext(os.path.basename(options.file))[0]
    else:
        outname=options.outname
    if options.outdir is None:
        outdir=os.path.dirname(options.file)
    else:
        outdir=options.outdir
    writeF=int(options.writeF)>0
    makeplot=int(options.makeplot)>0
    filternii=int(options.filternii)>0
    
    if forceScanPhyslog or (os.path.split(options.file)[1].find('SCANPHYSLOG') >= 0 and \
       (os.path.splitext(options.file)[1]=='.log' or os.path.splitext(options.file)[1]=='.LOG')):
        if options.fs is None:
            fs=500.
        else:
            fs=float(options.fs)

        if options.nvols is None or options.TR is None:
            print('For Philips SCANPHYSLOG, TR and nvols must be specified.')
        else:
            print('Asumming Philips SCANPHYSLOG')
            info=parseScanPhysLog(options.file,TR=float(options.TR),N=int(options.nvols),resporder=resporder,pulseorder=pulseorder\
                  ,niiname=options.data,fs=fs,mindistp=mindistp,mindistr=mindistr,\
                  outdir=outdir,outname=outname,makeplot=makeplot,writeF=writeF,
                  prefix=options.prefix,filternii=filternii,pulsepass=pulsepass,\
                  pulsestop=pulsestop,resppass=resppass,respstop=respstop,
                  rawtsv=options.rawtsv,respprom=options.respprom,pulseprom=options.pulseprom)
    else:
        print('Assuming CMRR DCM physlog')

        if options.fs is None:
            fs=400.
        else:
            fs=float(options.fs)
        info=parseDCM(options.file,resporder=resporder,pulseorder=pulseorder\
                  ,niiname=options.data,fs=fs,mindistp=mindistp,mindistr=mindistr,\
                  outdir=outdir,outname=outname,makeplot=makeplot,writeF=writeF,
                  prefix=options.prefix,filternii=filternii,pulsepass=pulsepass,\
                  pulsestop=pulsestop,resppass=resppass,respstop=respstop,
                  rawtsv=options.rawtsv,respprom=options.respprom,pulseprom=options.pulseprom)
