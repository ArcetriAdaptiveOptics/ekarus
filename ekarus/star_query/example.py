# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 12:05:35 2025

@author: lomba

example for the use of the StarQuery class 
contains also the function to compute the statistical analysis of target star

the ticho_2 catalogue can be downloaded at 
    https://drive.google.com/drive/folders/1VQTy8Uk5qHhnRifWXwCqgAWWiSyg2WzS?usp=drive_link
    just save the I_256 folder wherever you want on your pc and set the "path_tycho" variable
    
original data from
    https://cdsarc.cds.unistra.fr/viz-bin/Cat?I/259#browse -> FTP
I downloaded and unzipped the full folder. Do not unzip the single file. The program does it itself
For some reason now i can't download the folder anymore   


"""
import pandas as pd
import gzip
from pathlib import Path
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle
from datetime import datetime, timedelta


from star_query import StarQuery



#%%cima ekar observatory

'''Define lat°, lon°, height [m] for the observation post.'''

lat, lon = 45.84877616799979, 11.569718997174123
height = 1366


#%%load the catalogue 

'''
To load a catalogue pass the path to the unzipped "I_259" folder containing the zipped files downloaded from vizier.
Either as string or Path object is fine. 

Leave the "catalog_name" as "tycho_2". This simply tell the system wich submethod to use. The reason for that is that in 
this way other catalogs (Gaia, HR..) can be loaded if necessary (not implemented yet), you just need to make the corresponding
loader.

The loader for the Tycho catalogue takes two more parameters. Simply the list of sections and summplementary files you want 
to load. By default loads the whole catalogue.

'''

path_tycho = Path("C:\\Users\\lomba\\Documents\\work\\EKARUS\\ticho_2")

th = StarQuery(lat=lat, lon=lon, height=height)
th.load(catalog_name="tycho2", path=path_tycho, sections=np.arange(20), suppl=[1,2])


#%%assigning a location

'''
Location can be reassigned as tuple. The system then converts it to a astropy EarthLocation object.
When reassigning location the timezone is updated ("_tzname") and the Sun and Moon attributes are redefined.
The columns ["alt_max", "sky_time", "rise", "set", "obs_time"] are then recomputed.

The Sun in particular is important because it's used to calculate the available observational time for the current night.'
'''

th.location = (lat, lon, height)



#%%save a copy of the full catalogue

'''This will be useful later.'''

df0 = th.df.copy()

#%%set some filtering for the current local time
'''
The "mag_max" attribute tells the system the maximum apparent magnitude we are intrested in. It may be useful to change
it to an interval of magnitudes.

Settin "alt_min" [deg] we are calling a property method that stores the value as an attribute, but also updates the
["sky_time"] columns in the df attribute. That's because changing the minimum altitude of a star
on the local horizon changes the time it spends above it. 

NOTE: 
    -All the columns added to the dataframe of stars represents time intervals as hours (saved as float for speed).
    -In particular the "sky_time" is the time spent by the star above the min_alt, while the "obs_time" is actual observation time
    for the current night
    -The "rise", "set", "t_culm" columns intead are the time to the respective event as difference from the current hour.
    -To simplify (in my mind) things those time intervals are wrapped between the last midday and the next one. This way we are
    always referring to che closest night available and there is no discontinuity before and after midnight.

'''

th.mag_max = 7
th.alt_min = 70


#%%
'''
The "time" attribute is a datetime object. To easiness of use it needs to be assigned as a string in ISO format ( can
be changed). Assigning a new time reconputes the ["rise", "set", "obs_time"] columns for the current "nocturnal" day.

Also updates the timezone. When needed an astropy object is defined from it, i discovered that if a timezone is specified 
astropy automatically considers it for computing UTC time, so just assign the local time. Should also considers solar/legal
hours discontinuities.

Of course the sun and moon parameters are updated. 
Again, the dates for the events are wrapped between the last midday and the next one.

Plotting the sky we will see a ring of star (main grid is equatorial) being all the ones culminating above the minimum
altitude at some time of the day.

'''


th.time = datetime.now().isoformat()
th.plot_sky()


#%%select only the current valid stars and plot them
'''
The "filter_sky" method return a copy of the current df filtered by alm_min and mag_max. If "current=True" only the stars
currently (at this hour) above alt_min are returned.

You will see only the disk of stars above the alt_min at the current set time .
They should be centered with the altazimutal grid
'''

df = th.filter_sky(current=1)
th.plot_sky(df=df)


#%%filter the sky and select only those stars obsesrvable for the current night
'''
This time a trail of stars will be displayed. All the ones that will spend the specified amount of time
above the alt_min between the dusk and dawn for the current "nocturnal" day.
Meaning the 24h from the last midday to the next one.
'''

th.time = th.sun.dusk
df = th.filter_sky(current=0)
mask =  df['obs_time'] > 0
data = df[mask].copy()
th.plot_sky(df=data)

#%%single night analysis
'''
Plot a 2d histogram (image) of the number of stars binned by magnitude and observable time for the current night.
Observable time is defined as the total time spent above the minimum altitude between the dusk and dawn. 
'''

# Number of bins
N_mag = 5   # magnitude bins
M_time = 20 # time bins

# 2D histogram
counts, mag_edges, time_edges = np.histogram2d(
    data["mag"], 
    data["obs_time"], 
    bins=[N_mag, M_time],
    range=[[data["mag"].min(), data["mag"].max()], 
            [data["obs_time"].min(), data["obs_time"].max()]]
)

print("2D array shape:", counts.shape)
# print(counts)

plt.figure(figsize=(10,6))
plt.imshow(counts, origin='lower', aspect='auto',
            extent=[time_edges[0], time_edges[-1], mag_edges[0], mag_edges[-1]],
           cmap='viridis')
plt.colorbar(label='Count')
plt.xlabel("Time (hours)")
plt.ylabel("Magnitude")
plt.title("2D Histogram: Magnitude vs Time")
plt.show()

#%%do it for several nights
'''
Same procedure but for 'n' days around the year. For each one results are stored and at the final mean is computed. 
The system automatically calculates the observable time during each night, so there is no need to specify  anything
other than the date.
'''
n = 100

dates = [ datetime.now() + timedelta(days=i * (365 / n)) for i in range(n)]

#%%
'''
To make sure we are not losing data (maybe because some of the following sections, see later, has been run) let's reassign
the original dataframe. 
NOTE: .df is a property method wich not only stores the input dataframe, but also performs it's update with the current 
parameters. so there isn't a set required order of operations (do whatewher you want it SHOULD not break).

'''

#changing the system paramets before df reassignment is fine
th.alt_min = 70
th.mag_max = 7

th.df = df0

# th.alt_min = 70
# th.mag_max = 10

#%%
'''
To speed up computation, a reduced copy of the catalogue can be reassigned to the th object.
Filter the sky for the current parameters and reassign the result as the new catalogue.
we still have the 'df0' copy anyway.

'''



df = th.filter_sky(current=0)
th.df = df
th.plot_sky()

#%%
'''
Set the number of bins to perform the analysis and runs it.
I'm using a rough binning for the time ("M_time") because low magnitude stars are better grouped.
Higher sampling allows for better visualization of high magnitude stars.
'''

N_mag = th.mag_max*2+1   # magnitude bins
# M_time = 47 # time bins
M_time = 100 # time bins
# M_time = 23 # time bins


obs_stars = []
for i in tqdm(np.arange(n)):

    th.time = dates[i]
    mask =  th.df['obs_time'] > 0
    data = th.df[mask].copy()
    
    counts, mag_edges, time_edges = np.histogram2d(
        data["mag"], 
        data["obs_time"], 
        bins=[N_mag, M_time],
        range=[[0, th.mag_max], 
                [0, 24]]
    )
    obs_stars.append(counts)

obs_stars = np.array(obs_stars)
star_mean = obs_stars.mean(axis=0)



#%%
'''
I tried some different visualizations. First is simply an image where colour is the counts and the axis are
magnitude and observable time. The problem is that low magnitude stars counts are too dark.
In all of them the dashed red line represent the maximum observable time for the stars in the database
and is the upper limit.

'''
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm

fig, ax = plt.subplots()

# Plot the image
img = ax.imshow(
    star_mean,

    origin='lower',
    aspect='auto',
    extent=[time_edges[0], time_edges[-1], mag_edges[0], mag_edges[-1]],
    cmap='viridis',
)

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Magnitude")
ax.set_title("2D Histogram: Magnitude vs Time")
cbar = fig.colorbar(img, ax=ax)
cbar.set_label("Count (original scale)")

# Add vertical line at max(th.df["sky_time"])
ax.axvline(x=max(th.df["sky_time"]), color='red', linestyle='--', linewidth=0.5)

plt.show()


#%%
'''
Option number 2: An histogram whith the results for each magnitude (basically the previous image rows) 
are piled on top of each other and separated by colour. Cool, but it's hard to distinguish magnitudes.
'''


import matplotlib as mpl

img = star_mean
H, W = img.shape

histograms = np.array(img.T)  # shape (W, num_bins)

# Stacked bar chart
x = np.arange(W)  # one bar per column
bottom = np.zeros(W)
cmap = plt.get_cmap('viridis', H)
fig, ax = plt.subplots(figsize=(6, 5))

for i in range(H):
    ax.bar(x, histograms[:, i], bottom=bottom, color=cmap(i), label=f'Bin {i+1}')
    bottom += histograms[:, i]

# ax.set_xticks(np.round(time_edges[:-1],2))
ax.set_xticks(x)

ax.set_xlabel("Image Columns")
ax.set_ylabel("Counts")
ax.set_title("Stacked Histograms per Column")
norm = mpl.colors.Normalize(vmin=min(df.mag), vmax=max(df.mag))

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # required for ScalarMappable
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("apparent magnitude")

plt.show()


#%%
'''
Last but not least (best one in my opinion) simply plot the rows of the image as funtion of the time_edges
for the computed histograms. Only drawback it's confusing having points instead of bins.
NOTE: In this case each point represent the number of star of such magnitude that are visible in the sky for 
AL LEAST the amount of time. The last point before the red line (max observable time) drops because the bin
is smaller (missing data) in respect to the prevoius ones.
'''

# cut = int(max(th.df["sky_time"]))

img = star_mean
fig, ax = plt.subplots()
# plt.figure()
plt.title("observable stars by time and magnitude")
for i, row in enumerate(img):
    # plt.plot(time_edges[:cut+1], row[:cut+1], "+-", label=f"{mag_edges[i+1]:.1f} mag")
    plt.plot(time_edges[:-1], row, "+-", label=f"{mag_edges[i+1]:.1f} mag")

# 
plt.xlabel("observation time")
plt.ylabel("counts")
plt.legend()
ax.axvline(x=max(th.df["sky_time"]), color='red', linestyle='--', linewidth=0.5)














