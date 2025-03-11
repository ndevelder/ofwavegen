#!/usr/bin/python
#----------------------------------------------------------------------------#
import sys
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def writeSpec(fout, waveEngine, f, amps, phases, waveK):

    nvals=len(f)
    H=amps*2.0
    T=1.0/f
    ## directions are all 0 for now
    DD=[0.0]*nvals

    numfmt='({:15.8e})'
    if (waveEngine == 0):
        outfmt='((' + (numfmt+' ')*(nvals-1) + numfmt + '));\n'
    elif (waveEngine == 1):
        outfmt='(' + (numfmt+' ')*(nvals-1) + numfmt + ');\n'
    else:
        outfmt='(\n' + (numfmt+'\n')*(nvals-1) + numfmt + '\n);\n'
        outfmt2='(\n' + ('(\t'+numfmt+'\t0\t0 )\n')*(nvals-1) + '(\t' + numfmt + '\t0\t0 )' + '\n);\n'

    # Wave Heights
    if (waveEngine != 2):
        lineout='waveHeights\n'
    else:
        lineout='waveAmps nonuniform List<scalar>\n'
    fout.write(lineout)
    if (waveEngine == 0):
        lineout='1\n'
    else:
        lineout=str(nvals)+'\n'

    fout.write(lineout)
    if (waveEngine != 2):
        lineout=outfmt.format(*H)
    else:
        lineout=outfmt.format(*H/2)
    fout.write(lineout)

    # Wave Periods
    if (waveEngine != 2):
        lineout='wavePeriods\n'
    else:
        lineout='wavePeriods nonuniform List<scalar>\n'
    fout.write(lineout)
    if (waveEngine == 0):
        lineout='1\n'
    else:
        lineout=str(nvals)+'\n'
    fout.write(lineout)
    if (waveEngine != 2):
        lineout=outfmt.format(*T)
    else:
        lineout=outfmt.format(*f)
    fout.write(lineout)

    # Wave Phases
    if (waveEngine != 2):
        lineout='wavePhases\n'
    else:
        lineout='wavePhases nonuniform List<scalar>\n'

    fout.write(lineout)
    if (waveEngine == 0):
        lineout='1\n'
    else:
        lineout=str(nvals)+'\n'
    fout.write(lineout)
    lineout=outfmt.format(*phases)
    fout.write(lineout)

    # Wave Angles/Dirs/Numbers
    if (waveEngine != 2):
        lineout='waveAngles\n'
        fout.write(lineout)
        if (waveEngine == 0):
            lineout='1\n'
        else:
            lineout=str(nvals)+'\n'
        fout.write(lineout)
        lineout=outfmt.format(*DD)
        fout.write(lineout)

        lineout='waveDirs\n'
        fout.write(lineout)
        if (waveEngine == 0):
            lineout='1\n'
        else:
            lineout=str(nvals)+'\n'
        fout.write(lineout)
        lineout=outfmt.format(*DD)
        fout.write(lineout)
    else:
        lineout='waveNumbers nonuniform List<vector>\n'
        fout.write(lineout)
        lineout=str(nvals)+'\n'
        fout.write(lineout)
        lineout=outfmt2.format(*waveK)
        fout.write(lineout)

def fftSignal(Fs, signalShort):
    nShort = len(signalShort)
    signalFFT = np.fft.fft(signalShort)/nShort
    signalFFT = 2*signalFFT[range(int(nShort/2))]
    Freqs = np.arange(int(nShort/2))*Fs/nShort
    return ([signalFFT, Freqs])

def waveNumber(g, omega, d):
    k0 = 1;
    err = 1;
    count = 0;
    while (err >= 10e-8 and count <= 100):
        f0 = omega*omega - g*k0*np.tanh(k0*d)
        fp0 = -g*np.tanh(k0*d)-g*k0*d*(1-np.tanh(k0*d)*np.tanh(k0*d))
        k1 = k0 - f0/fp0
        err = abs(k1-k0)
        k0 = k1
        count += 1

    if (count >= 100):
        print('Can\'t find solution for dispersion equation!')
        exit()
    else:
        return(k0)

def eta(amps,wavenum,freqs,phases,xshift,times):
    eleCalc = np.zeros(len(times))
    for A, k, freq, phi in zip(amps,wavenum,freqs,phases):
        eleCalc += A*np.cos(-k*xshift-(2.0*np.pi)*freq*times - phi) 
    return eleCalc

def searchwaves(nt,time,wvht):
    lentime = len(time)
    maxs = lentime-nt
    stimes = np.arange(0,maxs,step=2)
    hsfull = 4.0*np.sqrt(np.var(wvht))
    hsdifftrack = 1000.0
    hspos = 0
    buf = (hsfull/2)*1.1

    for t in stimes:
        searchtimes = time[t:t+nt]
        searchwvht = wvht[t:t+nt]
        hs = 4.0*np.sqrt(np.var(searchwvht))
        hsdiff = np.abs(hs - hsfull)
        htmax = np.max(searchwvht)
        if hsdiff < hsdifftrack:
            if htmax < (hsfull/2 + buf):
                if hs < hsfull:
                    hsdifftrack = hsdiff
                    hspos = t
    
    print('Hs final:',4.0*np.sqrt(np.var(wvht[hspos:hspos+nt])),'hspos',hspos,'hsfull',hsfull)
    print('Hs0',4.0*np.sqrt(np.var(wvht[0:nt])))
    return time[hspos:hspos+nt],wvht[hspos:hspos+nt]

def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate irregular wave input for OpenFOAM v2306 from elevation data")
    parser.add_argument(
        "-i",
        "--infile",
        help="Input wave signal",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-d",
        "--wavedirection",
        help="Wave direction (single value supported currently)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-e",
        "--waveengine",
        help="Wave engine",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        help="Output file name/path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-c",
        "--timecrop",
        help="Crop the time",
        required=False,
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-x",
        "--xshift",
        help="Domain shift in x",
        required=False,
        type=float,
        default=0.0,
    )


    args = parser.parse_args()

    #----------------------------------------------------------------------------#
    # Settings
    #----------------------------------------------------------------------------#
    g = 9.81
    d = 4.0             # Tank half height
    x = args.xshift		# wave probe is shifted X distance from the tank inlet
    fmin = 0.05
    fmax = 4.0
    simlength = args.timecrop

    savedir = os.path.dirname(args.outputfile)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    #----------------------------------------------------------------------------#
    # Import wavemaker time series
    #----------------------------------------------------------------------------#
    ufData = pd.read_csv(args.infile, sep=",")
    
    timesIn = ufData['Time'].to_numpy()
    dt = timesIn[1]-timesIn[0]
    print("dt is:",dt)
    Fs = 1/dt

    HtIn = ufData['Wave Probe C'].to_numpy()/1000.0

    if args.timecrop > 0.0:
        ntimes = int(simlength/dt)
    else:
        ntimes = len(timesIn)

    print('Using n timesteps:', ntimes, "for",ntimes*dt,"seconds of data at dt",dt )

    times,Ht = searchwaves(ntimes,timesIn,HtIn)

    directout = pd.DataFrame({'time':times, 'waveheight':Ht})
    directout.to_csv(args.outputfile.replace('.txt','_directout.csv'), index=False)

    
    #----------------------------------------------------------------------------#
    # Create the OpenFOAM input files
    #----------------------------------------------------------------------------#
    [magFFT, Freqs] = fftSignal(Fs, Ht)
    Amps = np.abs(magFFT)
    Phases = np.angle(magFFT)
    
    # Calculate wave numbers
    waveNum = np.zeros(len(Freqs))
    for i in range(len(Freqs)):
            om = 2*np.pi*Freqs[i]
            waveNum[i] = waveNumber(g, om, d)
    
    # Convert wave signal to time series incorporating the shift from tank inlet
    eleCalc = eta(Amps, waveNum, Freqs, Phases, x, times)

    #eleOutput = pd.DataFrame({'time':times,'elevation':eleCalc})
    #eleOutput.to_csv('elevation_'+args.outputfile)
    
    # FFT the new signal
    [magFFT2, Freqs2] = fftSignal(Fs, eleCalc)
    Amps2 = np.abs(magFFT2)
    Phases2 = np.angle(magFFT2)
    
    # Apply cut off frequencies
    indx_in = next(x[0] for x in enumerate(Freqs2) if x[1] > fmin)
    indx_out = next(x[0] for x in enumerate(Freqs2) if x[1] > fmax)
    Amps2 = Amps2[indx_in:indx_out]
    Freqs2 = Freqs2[indx_in:indx_out]
    Phases2 = Phases2[indx_in:indx_out]
    
    # Re-calculate wave numbers
    waveNum2 = np.zeros(len(Freqs2))
    for i in range(len(Freqs2)):
            om = 2*np.pi*Freqs2[i]
            waveNum2[i] = waveNumber(g, om, d)

    # Convert wave signal to time series incorporating the shift from tank inlet
    eleCalcR = np.zeros(len(times))
    eleCalcR = eta(Amps2, waveNum2, Freqs2, Phases2, 0.0, times)


    #eleOutputR = pd.DataFrame({'time':times,'elevation':eleCalcR})
    #eleOutputR.to_csv('elevationR_'+args.outputfile)

    
    #Write data to file
    with open(args.outputfile,'w') as fout:
            fout.write('nFreqs\t'+str(len(Freqs2)-1)+';\n')
            # Need to send negative phase to work with sign in Openfoam v2306
            writeSpec(fout, args.waveengine, Freqs2[1:], Amps2[1:], -1.0*Phases2[1:], waveNum2[1:])

    #----------------------------------------------------------------------------#
    # Example plots
    #----------------------------------------------------------------------------#
    fig, ax = plt.subplots(2,2,figsize=(12,12))

    ax[0,0].plot(times,Ht, color='black',label="Original")
    ax[0,0].plot(times,eleCalc,':',label="Reconstruction")
    ax[0,0].plot(times,eleCalcR,'--',label="Reconstruction filtered")
    ax[0,0].set_title("Wave heights")
    ax[0,0].set_xlabel("Time (s)")

    ax[0,1].semilogx(Freqs,Amps)
    ax[0,1].semilogx(Freqs2,Amps2)
    ax[0,1].set_title("Amplitude")
    ax[0,1].set_xlabel("Freq (Hz)")

    ax[1,0].semilogx(Freqs,Phases)
    ax[1,0].semilogx(Freqs2,Phases2)
    ax[1,0].set_title("Phase")
    ax[1,0].set_xlabel("Freq (Hz)")
    
    plt.savefig('wavegenoutput.png')

if __name__ == "__main__":
    main()

