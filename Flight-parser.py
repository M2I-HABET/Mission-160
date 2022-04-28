#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:07:20 2022

@author: Matthew E. Nelson

"""

# IMPORTS
# ===============================

# Import Mapping software
import tilemapbase

# Pandas for data analysis
import pandas as pd

# Setup Matplotlib to work in Jupyter notebooks
#%matplotlib inline

#Import needed libraries, mainly numpy, matplotlib and datetime
from math import radians, sin, cos, atan2,sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
import simplekml

# Import the Image function from the IPython.display module.
#from IPython.display import Image

# DEFINES
#===============================

# Adjustment for magnetic change
B = 53380.4/1000 # uT
I = 68 * 0.0174533

# approximate radius of earth in km
R = 6373.0

#G_EARTH = 9806.65 # -9.80665 m/s2
G_EARTH = 1 # -9.80665 m/s2

# Launch Coordinates in Decimal Degress
launch_loc = (41.59189627804254, -93.55588561331234)

#Flight ID - Example LX-158-C
flight_id = 'L-160-B'

#time1  = datetime.strptime('8-01-2008 00:00:00', date_format)
#set the date and time format
date_format = "%m-%d-%Y %H:%M:%S"
launch_time = datetime.strptime('4-26-2022 15:35:00',date_format)

har_df = pd.read_csv("Data/HAR_FDR.csv")
bert_df = pd.read_csv("Data/BERT_FDR.csv")
har_df.columns =['Ident', 'Lat', 'Lon', 'Altitude','Temp','Pressure','Humidity','Close']
bert_df.columns = ['Temp', 'Pressure', 'Humidity', 'Lat','Lon','Altitude','GPS','AccelX','AccelY','AccelZ','GyroX','GyroY','GyroZ','MagX','MagY','MagZ']

har_temp = har_df.Temp
bert_temp = bert_df.Temp
har_humidity = har_df.Humidity
har_pressure = har_df.Pressure
bert_humidity = bert_df.Humidity
bert_pressure = bert_df.Pressure
har_lat = har_df.Lat / 10000000
bert_lat = bert_df.Lat / 10000000
har_lon = har_df.Lon / 10000000
bert_lon = bert_df.Lon / 10000000
har_alt = har_df.Altitude / 1000
bert_alt = bert_df.Altitude / 1000


bert_df["Lon"] /= 10000000
bert_df["Lat"] /= 10000000
bert_df["Altitude"] /= 1000

har_df["Lon"] /= 10000000
har_df["Lat"] /= 10000000
har_df["Altitude"] /= 1000

bert_accelx = bert_df.AccelX
bert_accely = bert_df.AccelY
bert_accelz = bert_df.AccelZ
bert_gyrox = bert_df.GyroX
bert_gyroy = bert_df.GyroY
bert_gyroz = bert_df.GyroZ
bert_magx = bert_df.MagX
bert_magy = bert_df.MagY
bert_magz = bert_df.MagZ

# Functions
# =================================
def deg2rad(x):
    return x*0.0174533

def rad2deg(x):
    return x*57.2958

def integrate_gyro(g,ts):
    t0 = ts[0]
    result = np.zeros(g.shape[0])
    for i in range(1,g.shape[0]):
        dt=(ts[i]-t0)/1000.0
        result[i]=result[i-1]+g[i]*dt
        t0 = ts[i]
    return result

def process_and_plot_gyro(df):
    ts = df.values[:,0]
    gx = df.values[:,10] 
    gy = df.values[:,11] 
    gz = df.values[:,12] 
    t = np.arange(gx.shape[0])    
    gxi = integrate_gyro(gx,ts)
    gyi = integrate_gyro(gy,ts)
    gzi = integrate_gyro(gz,ts)
    fig = plt.figure(figsize=(20, 14))
    plt.plot(t,rad2deg(gxi),t,rad2deg(gyi),t,rad2deg(gzi))
    plt.title("Angles from gyroscope")
    plt.grid(which="Both")
    plt.legend(["x","y","z"])
    plt.ylabel("Degrees")
    plt.xlabel("Time")
    return gxi,gyi,gzi

def process_pitch_roll(df):
    ax = df.values[:,7] 
    ay = df.values[:,8] 
    az = df.values[:,9] 
    ax=np.clip(ax,-G_EARTH,G_EARTH)
    ay=np.clip(ay,-G_EARTH,G_EARTH)
    az=np.clip(az,-G_EARTH,G_EARTH)
    pitch = np.arcsin(-ax/-G_EARTH)
    roll = np.arctan2(ay,az)
    return pitch, roll

def plot_gyro_vs_acc(gyro,acc):
    fig = plt.figure(figsize=(20, 14))
    t=np.arange(gyro.shape[0])
    plt.plot(t,gyro,t,acc)
    plt.grid(which="Both")
    plt.title("Pitch and Roll - Gyroscope vs Accelerometer")
    plt.legend(["gyroscope","accelerometer"])
    plt.xlabel("Samples")
    plt.ylabel("Rotation(degrees)")
    
def calculate_mag_correction(df):
    mx = df.values[:,13] 
    my = df.values[:,14]
    mz = df.values[:,15]
    corr_x = (mx.min() + mx.max())/2
    corr_y = (my.min() + my.max())/2
    corr_z = (mz.min() + mz.max())/2
    return corr_x,corr_y,corr_z

def process_yaw(df, pitch, roll):
    corr_x,corr_y,corr_z = calculate_mag_correction(df)
    mx = df.values[:,13] - corr_x
    my = df.values[:,14] - corr_y
    mz = df.values[:,15] - corr_z    

    yaw = np.arctan2(
        np.cos(pitch) * mz*np.sin(roll)-my*np.cos(roll),
        mx + B * np.sin(I)*np.sin(roll)
    )
    return yaw

def process_and_plot_yaw(df):
    pitch,roll=process_pitch_roll(df)
    yaw = process_yaw(df, pitch, roll)
    t = np.arange(df.shape[0])
    fig = plt.figure(figsize=(20, 14))
    plt.plot(t,rad2deg(yaw))
    plt.grid(which='Both')
    plt.title("Yaw rotation from magnetometer and accelerometer")
    plt.xlabel("Samples")
    plt.ylabel("Degrees")
    
# =============================================


# Print out the information
print("Launch date is:",launch_time.date())
print("Launch time is:",launch_time.time())
time_sec = len(bert_df)
flight_time = timedelta(seconds=time_sec)
landing_time = launch_time+timedelta(seconds=time_sec)
print("Flight time is:",flight_time)
print("Landing time is:",landing_time)


lat1 = radians(launch_loc[0])
lon1 = radians(launch_loc[1])
lat2 = radians(bert_lat.iloc[-1])
lon2 = radians(bert_lon.iloc[-1])

dlon = lon2 - lon1
dlat = lat2 - lat1

a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
c = 2 * atan2(sqrt(a), sqrt(1 - a))

distance = R * c

print("Result: %.2f km" % distance)

print("The maximum temperature recorded inside the payload was",har_temp.max(),"C")
print("The minimum temperature recorded inside the payload was",har_temp.min(),"C")
print("The maximum temperature recorded inside BERT was",bert_temp.max(),"C")
print("The minimum temperature recorded inside BERT was",bert_temp.min(),"C")

print("The maximum humidity recorded inside the payload was",har_humidity.max(),"%")
print("The minimum humidity recorded inside the payload was",har_humidity.min(),"%")
print("The maximum humidity recorded inside BERT was",bert_humidity.max(),"%")
print("The minimum humidity recorded inside BERT was",bert_humidity.min(),"%")

print("The maximum dewpoint recorded inside the payload was {:.2f} C".format(har_temp.max()-((100 - har_humidity.max())/5.0)))
print("The minimum dewpoint recorded inside the payload was {:.2f} C".format(har_temp.min()-((100 - har_humidity.min())/5.0)))
print("The maximum dewpoint recorded inside BERT was {:.2f} C".format(bert_temp.max()-((100 - bert_humidity.max())/5.0)))
print("The minimum dewpoint recorded inside BERT was {:.2f} C".format(bert_temp.min()-((100 - bert_humidity.min())/5.0)))

print("The maximum pressure recorded inside the payload was",har_pressure.max(),"hPa")
print("The maximum pressure recorded inside BERT was",bert_pressure.max(),"hPa")
print("The minimum pressure recorded inside the payload was",har_pressure.min(),"hPa")
print("The minimum pressure recorded inside BERT was",bert_pressure.min(),"hPa")

print("The maximum altitude obtained is",har_alt.max(),"m, or",(har_alt.max()*3.2808),"ft")

# Graph the data

har_alt.plot(title="{} Altitude Plot (HAR)".format(flight_id),ylabel="Altitude in meters",xlabel="Time at 5 sec intervals",figsize=(20, 10));
plt.savefig('Plots/har_alt_plot.pdf',bbox_inches = "tight",dpi = 500)

bert_alt.plot(title="{} Altitude Plot (BERT)".format(flight_id),ylabel="Altitude in meters",xlabel="Time in seconds",figsize=(20, 10));
plt.savefig('Plots/bert_alt_plot.pdf',bbox_inches = "tight",dpi = 500)

har_temp.plot(title="{} Temperature Plot (HAR)".format(flight_id),ylabel="Temperature in C",figsize=(20, 10));
plt.savefig('Plots/har_temp_plot.pdf',bbox_inches = "tight",dpi = 500)

bert_df.Temp.plot(title="{} Temperature Plot (BERT)".format(flight_id),ylabel="Temp in C",figsize=(20, 10));
plt.savefig('Plots//bert_temp_plot.pdf',bbox_inches = "tight",dpi = 500)

har_df.Humidity.plot(title="{} Humidity Plot (HAR)".format(flight_id),ylabel="Humidity as %",xlabel="Time at 5 sec intervals",figsize=(20, 10));
plt.savefig('Plots/har_alt_plot.pdf',bbox_inches = "tight",dpi = 500)

bert_df.Humidity.plot(title="{} Humidity Plot (BERT)".format(flight_id),ylabel="Humidity (%)",figsize=(20, 10));
plt.savefig('Plots/bert_humidity_plot.pdf',bbox_inches = "tight",dpi = 500)

har_df.Pressure.plot(title="{} Pressure Plot (HAR)".format(flight_id),ylabel="Pressure in hPa",xlabel="Time at 5 sec intervals",figsize=(20, 10));
plt.savefig('Plots/har_pressure_plot.pdf',bbox_inches = "tight",dpi = 500)

bert_df.Pressure.plot(title="{} Pressure Plot (BERT)".format(flight_id),ylabel="Pressure in hPa",figsize=(20, 10));
plt.savefig('Plots/bert_pressure_plot.pdf',bbox_inches = "tight",dpi = 500)

har_df.plot(title="{} Temp vs Altitude Plot".format(flight_id),ylabel="Altitude in meters",xlabel="Temp in C",x="Temp",y="Altitude",figsize=(20, 10))
plt.savefig('Plots/tempalt_plot.pdf',bbox_inches = "tight",dpi = 500)

har_df.plot(title="{} Pressure vs Altitude Plot".format(flight_id),ylabel="Altitude in meters",xlabel="Pressure in hPa",x="Pressure",y="Altitude",figsize=(20, 10))
plt.savefig('Plots/pressurealt_plot.pdf',bbox_inches = "tight",dpi = 500)

# Setup fixed points for graphing with

# Turn on subplots
fig, ax1 = plt.subplots(figsize=(20,10))
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Humidity (%)', color=color)
ax1.plot(har_df.Humidity,color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Temperature in C', color=color)  # we already handled the x-label with ax1
ax2.plot(har_df.Temp)
ax2.tick_params(axis='y', labelcolor=color)

# Always have a good title
plt.title("{} Temp vs Humidity".format(flight_id),color='c')
# This allows us to save our pretty graph so we can frame it later
plt.savefig('Plots/temp_humidity.pdf',bbox_inches = "tight",dpi = 500)

# Setup fixed points for graphing with

# Turn on subplots
fig, ax1 = plt.subplots(figsize=(20,10))
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Pressure (hPa)', color=color)
ax1.plot(har_df.Pressure,color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Temperature in C', color=color)  # we already handled the x-label with ax1
ax2.plot(har_df.Temp)
ax2.tick_params(axis='y', labelcolor=color)

# Always have a good title
plt.title("{} Temp vs Pressure".format(flight_id),color='c')
# This allows us to save our pretty graph so we can frame it later
plt.savefig('Plots/temp_pressure.pdf',bbox_inches = "tight",dpi = 500)

# Setup fixed points for graphing with

# Turn on subplots
fig, ax1 = plt.subplots(figsize=(20,10))
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Pressure (hPa)', color=color)
ax1.plot(har_df.Pressure,color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Humidity (%)', color=color)  # we already handled the x-label with ax1
ax2.plot(har_df.Humidity)
ax2.tick_params(axis='y', labelcolor=color)

# Always have a good title
plt.title("{} Humidity vs Pressure".format(flight_id),color='c')
# This allows us to save our pretty graph so we can frame it later
plt.savefig('Plots/humidity_pressure.pdf',bbox_inches = "tight",dpi = 500)


tilemapbase.init(create=True)

kml = simplekml.Kml()
linestring = kml.newlinestring(name=flight_id)
har_df.apply(lambda X: linestring.coords.addcoordinates([( X["Lon"],X["Lat"],X["Altitude"])]) ,axis=1)

linestring.altitudemode = simplekml.AltitudeMode.relativetoground
linestring.extrude = 1
linestring.linestyle.color = simplekml.Color.green
linestring.linestyle.width = 5
linestring.polystyle.color = simplekml.Color.orange
#pol = kml.newpolygon(name= 'ACTONO', description= 'Acton County', 
#outerboundaryis=tuples, extrude=extrude, altitudemode=altitudemode)

#Styling colors
#pol.style.linestyle.color = simplekml.Color.green
#pol.style.linestyle.width = 5
#pol.style.polystyle.color = simplekml.Color.changealphaint(100, 
#simplekml.Color.green)

#Saving
kml.save("KML/flight.kml")

fig = plt.figure(figsize=(20, 20))
# Always have a good title and labels
plt.ylabel('Latitude (DD.MM)', color=color)
plt.xlabel('Longitude (DD.MM)', color=color)

plt.title("{} HAR GPS Plot".format(flight_id),color='r')
plt.plot(bert_lon,bert_lat)
plt.savefig('Plots/gps_plot_nomap.pdf',bbox_inches = "tight",dpi = 500)


ur = (42.133700, -93.494635)
ll = (42.042358, -93.692460)
fig = plt.figure(figsize=(15, 15))
ax = Axes3D(fig)
ax.plot3D(har_lon,har_lat,har_alt)

ax.set_title(u'{} 3D plot of flight Path'.format(flight_id))
ax.set_xlabel(u'Longitude (°E)', labelpad=10)
ax.set_ylabel(u'Latitude (°N)', labelpad=10)
ax.set_zlabel(u'Altitude (meters)', labelpad=20)
ax.plot3D(har_lon, har_lat, har_alt, color = 'green', lw = 1.5)
plt.savefig('Plots/3D_Map_View.pdf',bbox_inches = "tight",dpi = 300)

# Define the `extent`
color='blue'

# Zoom, the higher the number the zoomed out it will be. This is centered on
# the launch location
degree_range = 1.9

extent = tilemapbase.Extent.from_lonlat(launch_loc[1] - degree_range, launch_loc[1] + degree_range,
                  launch_loc[0] - degree_range, launch_loc[0] + degree_range)
extent = extent.to_aspect(1.0)

# Convert to web mercator
path = [tilemapbase.project(x,y) for x,y in zip(har_lon, har_lat)]
x, y = zip(*path)

fig, ax = plt.subplots(figsize=(20,20))

plotter = tilemapbase.Plotter(extent, tilemapbase.tiles.build_OSM(), width=800)
plotter.plot(ax)
plt.ylabel('Latitude (Mercator)', color=color)
plt.xlabel('Longitude (Mercator)', color=color)

plt.title("{} GPS Plot on Street map".format(flight_id),color='r')

ax.plot(x, y,"b-")
plt.savefig('Plots/gps_plot_map.pdf',bbox_inches = "tight",dpi = 300)

# Define the `extent`
color='blue'

# Zoom, the higher the number the zoomed out it will be. This is centered on
# the landing location
degree_range = 0.05

extent = tilemapbase.Extent.from_lonlat(har_lon.iloc[-1] - degree_range, har_lon.iloc[-1] + degree_range,
                  har_lat.iloc[-1] - degree_range, har_lat.iloc[-1] + degree_range)
extent = extent.to_aspect(1.0)

# Convert to web mercator
path = [tilemapbase.project(x,y) for x,y in zip(har_lon, har_lat)]
x, y = zip(*path)

fig, ax = plt.subplots(figsize=(20,20))
t = tilemapbase.tiles.Carto_Light
plotter = tilemapbase.Plotter(extent, t, width=800)
plotter.plot(ax)
plt.ylabel('Latitude (Mercator)', color=color)
plt.xlabel('Longitude (Mercator)', color=color)

plt.title("{} GPS Plot Landing".format(flight_id),color='r')

ax.plot(x, y,"b-")
plt.savefig('Plots/gps_plot_map_landing.pdf',bbox_inches = "tight",dpi = 300)


# Convert the Pandas dataframes to NumPy arrays
accel = np.array([bert_df.AccelX.to_numpy,bert_df.AccelY.to_numpy,bert_df.AccelZ.to_numpy])
gyro = np.array([bert_df.GyroX.to_numpy,bert_df.GyroY.to_numpy,bert_df.GyroZ.to_numpy])
mag = np.array([bert_df.MagX.to_numpy,bert_df.MagY.to_numpy,bert_df.MagZ.to_numpy])

    
gxi,gyi,gzi = process_and_plot_gyro(bert_df)
    
pitch,roll=process_pitch_roll(bert_df)
plot_gyro_vs_acc( rad2deg(gxi),rad2deg(pitch))


corr_x,corr_y,corr_z = calculate_mag_correction(bert_df)
mx = bert_df.values[:,13] - corr_x
my = bert_df.values[:,14] - corr_y
mz = bert_df.values[:,15] - corr_z  

t = np.arange(mx.shape[0])
fig = plt.figure(figsize=(20, 14))
plt.plot(t,mx,t,my,t,mz)
plt.grid(which='Both')
    
process_and_plot_yaw(bert_df)

