# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:52:08 2025

@author: lomba


class for preprocessing astronomical star catalogues and program observations
written for performing the sky coverage analysis fot the EKARUS project


requires: 
    astropy
    pandas
    dateutil
    zoneinfo
    astral
    timezonefinder v.6.0.1  <- reinstall numpy then. 2.2.2 works for me
    tqdm
"""


import pandas as pd
import gzip
from pathlib import Path
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time #, TimeDelta
from dateutil import tz
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from astral import LocationInfo
from astral.sun import sun, dawn, dusk
from timezonefinder import TimezoneFinder
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle
from astral.moon import moonrise, moonset
from astropy.coordinates import solar_system_ephemeris, get_body
from typing import Union, Iterable


#%%

class StarQuery():
    '''Querying class for star catalogues.
    NOTE: day goes from midday to midday.'''
    
    def __init__(self, lat:float, lon:float, height:float, 
                 time:str=datetime.now().isoformat()):
        
        #location on earth
        self._location = EarthLocation(
            lat=lat * u.deg,
            lon=lon * u.deg,
            height=height * u.m
        )
        
        #observation
        self.mag_max = np.inf #maximum magnitude to observe
        self._alt_min = self.location.lat.deg #minimum altitude on the horizon
        self._time = datetime.fromisoformat(time) #local time and date
        self._update_timezone() #set the self._dtime for the timezone
        
        #stars dfset
        self._df = None  #original full dfset of stars
        
        #constants
        self._SDAY = 23.934469583333335 #sidereal day [h]
        self._WSKY = 360 / self._SDAY #sky angular speed deg/h
        
        #sun and moon
        self.sun = Sun(self)
        self.moon = Moon(self)
    
    
    def load(self, catalog_name:str, path:str, **kwargs):
        '''Load some star catalog. The implementation is specific for each.
        It opens the path and from the 'catalog_name' chooses the filed to read.'''
        
        def tycho2(tycho_path, sections=np.arange(20), suppl=[1,2]):
            '''Load the ticho_2 catalogue. 
            "sections" is an iterable with the numbers of the sections you want to load.
            "suppl" is another one with the number of supplementary entries.
            NTE: some important targets (vega) are in the supplementary files.'''
            
            #read main sections
            columns = ['TYC123','pflag','RAmdeg','DEmdeg','pmRA','pmDE','e_RAmdeg','e_DEmdeg','e_pmRA',
                        'e_pmDE','pRAm','EpDEm','Num','q_RAmdeg','q_DEmdeg','q_pmRA','q_pmDE','BTmag','e_BTmag','VTmag',
                        'e_VTmag','prox','TYC','HIP_CCDM','RAdeg','DEdeg','EpRA-1990','EpDE-1990','e_RAdeg','e_DEdeg',
                         'posflg','corr']
    
            df_list = []
            for i in tqdm(sections, "loading"):
                # path = Path("C:\\Users\\lomba\\Documents\\work\\star_catalogs\\ticho_2\\I_259\\tyc2.dat.00.gz")
                path = Path(tycho_path / Path("I_259\\tyc2.dat."+f"{i:02}"+".gz"))
    
                # print(path)
                with gzip.open(path, 'rt') as f:
                    df = pd.read_csv(f, sep='|', names=columns, dtype=str)
                
                # print(df.loc[:,'TYC123'])
                df['RAdeg'] = pd.to_numeric(df['RAdeg'], errors='coerce')
                df['DEdeg'] = pd.to_numeric(df['DEdeg'], errors='coerce')
                df['VTmag'] = pd.to_numeric(df['VTmag'], errors='coerce')
                # df['BTmag'] = pd.to_numeric(df['BTmag'], errors='coerce')
    
                # filter the catalog for colums and apparent magnitude
                df = df.loc[:,['TYC123', 'HIP_CCDM', 'RAdeg', 'DEdeg', 'VTmag']]
                df = df.dropna(subset=['TYC123', 'RAdeg', 'DEdeg', 'VTmag'])

                df = df.rename(columns={'TYC123': 'id',
                                        'HIP_CCDM': 'HIP',
                                        'RAdeg': 'RA',
                                        'DEdeg': 'DE',
                                        'VTmag': 'mag'})
                df['HIP'] = df['HIP'].str.replace(r'\s+', '', regex=True)
                df_list.append(df)
            
            #read supplementary files
            tycho2_supp_columns = ['TYC123','Tflag','RAdeg','DEdeg','pmRA','pmDE','e_RAmdeg','e_DEmdeg','e_pmRA',
                              'e_pmDE','mflag','BTmag','e_BTmag','VTmag', 'e_VTmag','prox','TYC','HIP_CCDM']

            for i in tqdm(suppl):
                # path = Path("C:\\Users\\lomba\\Documents\\work\\star_catalogs\\ticho_2\\I_259\\tyc2.dat.00.gz")
                path = Path(tycho_path / Path(f"I_259\\suppl_{i}.dat.gz"))
                
                with gzip.open(path, 'rt') as f:
                    df = pd.read_csv(f, sep='|', names=tycho2_supp_columns, dtype=str)
                
                df['RAdeg'] = pd.to_numeric(df['RAdeg'], errors='coerce')
                df['DEdeg'] = pd.to_numeric(df['DEdeg'], errors='coerce')
                df['VTmag'] = pd.to_numeric(df['VTmag'], errors='coerce')
                # df['BTmag'] = pd.to_numeric(df['BTmag'], errors='coerce')
    
                # filter the catalog for colums and apparent magnitude
                df = df.loc[:,['TYC123', 'HIP_CCDM', 'RAdeg', 'DEdeg', 'VTmag']]
                df = df.dropna(subset=['TYC123', 'RAdeg', 'DEdeg', 'VTmag'])

                df = df.rename(columns={'TYC123': 'id',
                                        'HIP_CCDM': 'HIP',
                                        'RAdeg': 'RA',
                                        'DEdeg': 'DE',
                                        'VTmag': 'mag'})
                df['HIP'] = df['HIP'].str.replace(r'\s+', '', regex=True)
                df_list.append(df)
                
                
            df_combined = pd.concat(df_list, ignore_index=True)
            return df_combined.set_index('id')
            
            
        path = Path(path) 
        if catalog_name=="tycho2": df = tycho2(tycho_path=path, sections=kwargs["sections"], suppl=kwargs["suppl"])
        
        #assign the catalogue. select only positive declinations
        self._df = df.loc[df['DE'] > 0].copy()
        # self._df = df
        #call lat assignation to update additional
        self._update_timezone()
        self._update_max_altitude()
        self._update_sky_time()

    @property
    def location(self):
        return self._location
    @location.setter
    def location(self, val:tuple):
        '''Set the location on eart as (lat, lon, height).
        Automatically updates:
            timezone
            star max altitude on the horizon
            star max time on sky above set alt_min
            sun and moon positions/rise/set
            star time to culmination/rise/set in hours'''
        lat, lon, height = val
        self._location = EarthLocation(
            lat=lat,
            lon=lon,
            height=height
        )
        # self._location = val
        self._update_timezone()
        self._update_max_altitude()
        self._update_sky_time()
        self.sun = Sun(self)
        self.moon = Moon(self)
        self._time_to_culmination()

    @property
    def alt_min(self):
        '''Stars min altitude on the horizon.'''
        return self._alt_min
    @alt_min.setter    
    def alt_min(self, val):
        '''Set min altitude on the horizon.
        Updates stars remaining sky time'''
        if val <= self.location.lat.deg: 
            raise ValueError("Minimum altitude on the local horizon must be greater than latitude.")
        self._alt_min = val
        self._update_sky_time()
        
    @property
    def time(self):
        '''Return the local time.'''
        return self._time 
    @time.setter
    def time(self, val:str):
        '''Set the local time. Updates stars time to culmination'''
        self._time = datetime.fromisoformat(str(val)).replace(microsecond=0)
        self._time = self.time.astimezone(ZoneInfo(self._tzname))
        self.sun = Sun(self)
        self.moon = Moon(self)
        self._time_to_culmination()
        
    @property
    def df(self):
        return self._df
    @df.setter
    def df(self, df):
        self._df = df
        self.alt_min = self._alt_min
        self.time = self._time
        
    def _update_max_altitude(self):
        '''Calculates the max altitude for all stars in the catalogue.
        Considers culmination to nord and sud both.'''
        #update maximum altitude on the horizon
        lat = self._location.lat.deg  # Convert to float degrees
        dec = self.df['DE'].values  # Declination in degrees
        
        max_alt = 90 - np.abs(lat - dec)
        self.df['alt_max'] = max_alt
        
    
    def _update_timezone(self):
        '''Updates the time offset in hours from the timezone shift.
        I think (not 100% sure) that legal/solar time are managed.'''
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=self.location.lon.deg, lat=self.location.lat.deg)
        self._tzname = timezone_str
        self._time = self.time.astimezone(ZoneInfo(timezone_str))
    
    
    def _wrap_time_array(self, dt):
        '''Wrap the input "dt" array [hours] between the last midday and the next one.
        '''
        
        #wrapping of negative values (move forward 1day)
        if self.time.time() > time(12,0):
            t_off = (self.time - self.time.replace(hour=12)).total_seconds() / 3600
        else: 
            t_off = (self.time - (self.time-timedelta(hours=self._SDAY)).replace(hour=12))
            t_off = t_off.total_seconds() / 3600
        dt = np.where(dt<-t_off, dt+self._SDAY, dt)
        
        #wrapping positive values (move back 1 day)
        if self.time.time() > time(12,0):
            t_off = ( (self.time+timedelta(hours=self._SDAY)).replace(hour=12) - self.time )
            t_off = t_off.total_seconds() / 3600
        else: 
            t_off = (self.time.replace(hour=12) - self.time ).total_seconds() / 3600
        dt = np.where(dt>t_off, dt-self._SDAY, dt)
        return dt
        
    
    def _time_to_culmination(self):
        '''Calculates the differential time to culmination from the currently set time.
        Performs the wrapping of time from midday to midday.'''
        
        #calc the culmination coordinates in equatorial ref system
        alt = self.df["alt_max"].values
        az = np.where(self.df["DE"].values >=self.location.lat.deg, 0, 180)
        now = Time(self.time)
        
        # Convert to equivalent culmination position in ICRS (RA/Dec)
        altaz_frame = AltAz(obstime=now, location=self._location)
        altaz_coords = SkyCoord(alt=alt*u.deg, az=az*u.deg, frame=altaz_frame)
        eq_culm = altaz_coords.transform_to("icrs")
        
        # #calc the datetime of culmination
        #difference between the equatorial culmination RA and current one divided by sky angular speed
        d_ra = -(eq_culm.ra.value - self.df["RA"].values)
        dt = d_ra / self._WSKY  #time to culmination (positive=future, negative=past) [h]
        
        #culmination, rising and setting above min altitude
        self.df["t_culm"] = self._wrap_time_array(dt=dt)
        self.df["rise"] = self._wrap_time_array(dt = dt - self.df.sky_time / 2 )
        self.df["set"]  = self._wrap_time_array(dt = dt + self.df.sky_time / 2 )
        
        #available time of observation
        obs_time = pd.Series(0.0, index=self.df.index)
        rise, set_ = self.df["rise"], self.df["set"]
        dusk = (self.sun.dusk - self.time).total_seconds()/3600
        dawn = (self.sun.dawn - self.time).total_seconds()/3600
        
        # Case 1: the star rises and sets normally
        mask1 = rise <= set_
        obs_time[mask1] = (np.minimum(set_[mask1], dawn) - np.maximum(rise[mask1], dusk)).clip(lower=0)
        
        #not properly tested. only relevant for small alt_min
        # Case 2: the star sets then rises before dusk or after dawn
        mask2 = (set_ < rise) & ((rise <= dusk) | (dawn <= set_))
        obs_time[mask2] = dawn - dusk
        
        # Case 3: all other cases
        mask3 = ~(mask1 | mask2)
        obs_time[mask3] = (dawn - rise[mask3]).clip(lower=0) + (set_[mask3] - dusk).clip(lower=0)
        # obs_time[mask3] = t.clip(lower=0)
        
        self.df["obs_time"] = obs_time

        
    
    def _update_sky_time(self):
        '''Calculates the time [h] spent by each star above the minimum altitude.
        NOTE: demonstration is only valid if DE>=0 and for Alt>=lat, 
        to generalize it's probably enough to add some abs ..
        Stars below the minimum altitude have sky_time=np.nan because of the arccos.
        NOTE: since this is a time interval the time wrapping must not be performed.''' 

        if min(self.df["DE"])<0: raise ValueError("Declination negative")
        
        de = self.df["DE"].values
        l = np.sin((90-de)*np.pi/180)
        OI = np.cos((90-de)*np.pi/180)
        # a = OI * np.tan((Alt_min-lat)*np.pi/180)
        MQ = np.sin(self.alt_min*np.pi/180)
        OM = MQ / np.sin(self.location.lat.deg*np.pi/180)
        IM = OM - OI
        g = IM * np.tan(self.location.lat.deg*np.pi/180)

        if min(g)<0: raise ValueError("Smt went wrong. it's not possible.")
        if min(l)<=0: raise ValueError("l zero or negative")
        
        #compute sky time for stars above min altitude. everything else set to np.nan
        ratio = g / l
        theta = np.full_like(ratio, np.nan, dtype=float)
        mask = ratio <= 1
        theta[mask] = 2 * np.arccos(ratio[mask]) * 180 / np.pi
        t = theta / self._WSKY 
        
        self.df["sky_time"] = t
        
        
    def filter_sky(self, current=False):
        """Return a copy of the current stars dataframe (self.df) filtered with the current settings.
        The 'current' flag return only the stars visible right now above the alt_min."""
        mask = (self.df['alt_max'] >= self.alt_min) & (self.df['mag'] <= self.mag_max)
        
        if current: 
            ra = self.df['RA'].values
            dec = self.df["DE"].values
            equatorial_coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            altaz_frame = AltAz(obstime=Time(self.time), location=self.location)
            altaz_coords = equatorial_coords.transform_to(altaz_frame)
            mask =  mask & (altaz_coords.alt >= self.alt_min*u.deg)
            
        return self.df.loc[mask].copy()
        
    def plot_sky(self, df=None, star_id=False):
    # def plot_sky(self, df=None):
        '''Plot the input stars dataframe. If None (default) is provided, the current one filtered
        with the system settings is displayed. 
        If 'star_id'==True, the stars ids are printed next to them.
        The main grid is a equatorial one of the northern sky. In red is the altazimutal grid at the 
        current time and location.'''

        df = self.filter_sky() if df is None else df
        
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.set_ylim(90, 0)   # reversed order!
        
        # Set angular ticks (theta axis)
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['0h', '3h  RA', '6h', '9h', '12h', '15h', '18h', '21h'])

        # Set radial ticks
        ax.set_yticks([80, 60, 40, 20])
        ax.set_yticklabels(['80°', '60°', '40°', '20° DE'])
        
        sc = ax.scatter(np.deg2rad(df.RA.values), df.DE.values, c=df.mag.values, cmap='viridis_r', s=10)
        if star_id:
            for idx, row in df.iterrows():
                ra = np.deg2rad(row.RA)  # RA in radians
                de = row.DE  # declination (already in deg)
                plt.text(ra, de, str(idx), fontsize=10, ha='left', va='top')
        cbar = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.1)
        cbar.set_label("apparent magnitude")
        
        ax.set_theta_zero_location("N")   # 0° at the top (north)
        ax.set_theta_direction(-1)        # azimuth increases clockwise
        ax.set_rlabel_position(180)       # radius labels at left
        ax.set_title(f" Stars culminating ≥ {self.alt_min}°", va='bottom', pad=20)
        
        #plot some relevant objects
        p_ra = Angle("2h31m49s").degree
        p_deg = Angle("89°15′50.8").degree
        ax.plot(np.deg2rad(p_ra), p_deg, 'r+', markersize=10)
        ax.text(np.deg2rad(p_ra), p_deg, 'Polaris', fontsize=10, ha='left', va='top')
        
        p_ra = Angle("18h36m55.41s").degree
        p_deg = Angle("38°47′35.8").degree
        # ax.plot(np.deg2rad(p_ra), p_deg, 'r+', markersize=10)
        ax.text(np.deg2rad(p_ra), p_deg, 'Vega', fontsize=10, ha='left', va='bottom')
        ax.set_rlabel_position(45)
        
        #plot the AltAz grid at the current time
        altaz_frame = AltAz(obstime=Time(self.time), location=self.location)
        
        for i in [0,45,90,135,180,225,270,315]:
            alt_obs = np.linspace(90,0,100)
            az_obs = np.full(100, i)
            zenith = SkyCoord(alt=alt_obs*u.deg, az=az_obs*u.deg, frame=altaz_frame)
            icrs = zenith.transform_to('icrs')
            theta0 = icrs.ra.radian  # RA in radians
            r0 =  icrs.dec.value  # radius: 90° - Dec
            ax.plot(theta0, r0, linewidth=0.5, color="r")
        
        
        for i in [0, 20,40,60,80]:
            alt_obs = np.full(100, i)
            az_obs = np.linspace(0,360,100)
            zenith = SkyCoord(alt=alt_obs*u.deg, az=az_obs*u.deg, frame=altaz_frame)
            icrs = zenith.transform_to('icrs')
            theta0 = icrs.ra.radian  # RA in radians
            r0 =  icrs.dec.value  # radius: 90° - Dec
            ax.plot(theta0, r0, linewidth=0.5, color="r")
            
        #plot sun and moon
        sun = SkyCoord(alt=self.sun.alt*u.deg, az=self.sun.az*u.deg, frame=altaz_frame)
        icrs = sun.transform_to('icrs')
        theta0 = icrs.ra.radian  # RA in radians
        r0 =  icrs.dec.value  # radius: 90° - Dec
        ax.plot(theta0, r0, 'o', markersize=10, color='red')
        ax.text(theta0, r0, 'Sun', fontsize=10, ha='left', va='bottom')
        
        moon = SkyCoord(alt=self.moon.alt*u.deg, az=self.moon.az*u.deg, frame=altaz_frame)
        icrs = moon.transform_to('icrs')
        theta0 = icrs.ra.radian  # RA in radians
        r0 =  icrs.dec.value  # radius: 90° - Dec
        ax.plot(theta0, r0, 'o', markersize=10, color='red')
        ax.text(theta0, r0, 'Moon', fontsize=10, ha='left', va='bottom')
        
        plt.show()
        
    
    def _shift_midday_day(self, new_time: Iterable[datetime]) -> datetime:
        """
        Shift datetime so that days run from midday to midday.
        If dt is before midday and new_time is after midday -> shift back one day.
        If dt is after midday and new_time is before midday -> shift forward one day.
        Same as _wrap_time_array but for datetime objs instead of floats.
        """
        midday = time(12, 0)
        dt = self.time
        
        def adjust(single_new_time: datetime) -> datetime:
            if dt.time() < midday and single_new_time.time() >= midday:
                return single_new_time - timedelta(hours=self._SDAY)
            elif dt.time() >= midday and single_new_time.time() < midday:
                return single_new_time + timedelta(hours=self._SDAY)
            return single_new_time
        
        return [adjust(nt).replace(microsecond=0) for nt in new_time]
        

    
class Sun:
    '''Contains the sun rising and setting time for the current "nocturnal day".'''
    def __init__(self, main):
        # Create location
        location = LocationInfo(
            name="", 
            region="", 
            timezone=main._tzname, 
            latitude=main.location.lat.deg, 
            longitude=main.location.lon.deg
        )

        # Current datetime for comparison
        now = main.time  # should be timezone-aware datetime

        # Compute sun events
        s = sun(location.observer, date=now.date(), tzinfo=location.timezone)
        dawn_time = dawn(location.observer, date=now.date(), tzinfo=location.timezone)
        dusk_time = dusk(location.observer, date=now.date(), tzinfo=location.timezone)

        self.rise = main._shift_midday_day([s['sunrise']])[0].replace(microsecond=0)
        self.set = main._shift_midday_day([s['sunset']])[0].replace(microsecond=0)
        self.dawn = main._shift_midday_day([dawn_time])[0].replace(microsecond=0)
        self.dusk = main._shift_midday_day([dusk_time])[0].replace(microsecond=0)
        
        
        # Sun coordinates
        altaz_frame = AltAz(obstime=Time(now), location=main.location)
        sun_coord = get_sun(Time(main.time))
        sun_altaz = sun_coord.transform_to(altaz_frame)
        
        self.alt = sun_altaz.alt.deg
        self.az = sun_altaz.az.deg
    
    
        # String representation
    # String representation using default datetime format
    def __str__(self):
        return (
            f"Sun Events:\n"
            f"  Set  : {self.set}\n"
            f"  Dusk : {self.dusk}\n"
            f"  Dawn : {self.dawn}\n"
            f"  Rise : {self.rise}\n"
            f"Sun Position:\n"
            f"  Altitude : {self.alt:.2f}\n"
            f"  Azimuth  : {self.az:.2f}"
        )
    
    
    
class Moon:
    '''Same as Sun, only for Moon.'''
    def __init__(self, main):
        # Observer location
        self.location_info = LocationInfo(
            name="",
            region="",
            timezone=main._tzname,
            latitude=main.location.lat.deg,
            longitude=main.location.lon.deg
        )

        now = main.time  # must be timezone-aware datetime

        # Compute Moon rise and set
        
        

        try: 
            mr = moonrise(self.location_info.observer, date=now.date(), tzinfo=self.location_info.timezone)
            self.rise = main._shift_midday_day([mr])[0].replace(microsecond=0)
        except (ValueError, AttributeError): self.rise = None
        try: 
            ms = moonset(self.location_info.observer, date=now.date(), tzinfo=self.location_info.timezone)
            self.set = main._shift_midday_day([ms])[0].replace(microsecond=0)
        except (ValueError, AttributeError): self.set = None
        

        # Current Moon coordinates
        altaz_frame = AltAz(obstime=Time(now), location=main.location)
        solar_system_ephemeris.set('builtin')
        moon_coord = get_body('moon', Time(now), location=main.location).transform_to(altaz_frame)
        self.alt = moon_coord.alt.deg
        self.az = moon_coord.az.deg

        # Calculate Moon phase
        # self.phase = self.calculate_moon_phase(now)


    def __str__(self):
        return (
            f"Moon Events:\n"
            f"  Set  : {self.set}\n"
            f"  Rise : {self.rise}\n"
            f"Moon Position:\n"
            f"  Altitude : {self.alt:.2f}\n"
            f"  Azimuth  : {self.az:.2f}\n"
            # f"  Phase    : {self.phase:.1f}"
        )
    