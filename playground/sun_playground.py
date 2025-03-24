import math
import ephem
from datetime import datetime
from vpython import *

from airstart3d.sun import Sun



# Set up the VPython scene
scene = canvas(title="Sun Simulation - 30x Speed")
scene.lights = []  # Remove default lighting

# initialze sun:
sun = Sun(datetime(2025, 3, 8, 6, 0), '46.5', '7.9', True)


# Set elements into scene
box(pos=vector(0, 0, 0), size=vector(10, 0.1, 5))
sphere(pos=vector(0.5, 0.5, 0.5), radius=1.0)

L = 50
R = L/100
d = L-2
xaxis = cylinder(pos=vector(0,0,0), axis=vector(d,0,0), radius=R, color=color.yellow)
yaxis = cylinder(pos=vector(0,0,0), axis=vector(0,d,0), radius=R, color=color.yellow)
zaxis = cylinder(pos=vector(0,0,0), axis=vector(0,0,-d), radius=R, color=color.yellow)        
k = 1.02
h = 0.05*L
text(pos=xaxis.pos+k*xaxis.axis, text='east', height=h, align='center', billboard=True, emissive=True)
text(pos=yaxis.pos+k*yaxis.axis, text='alt', height=h, align='center', billboard=True, emissive=True)
text(pos=zaxis.pos+k*zaxis.axis, text='north', height=h, align='center', billboard=True, emissive=True)

# Create Observer at kleine Scheidegg


# Time simulation settings
time_speed = 30  # 30x speed
time_step = 30  # 30 real seconds per iteration
total_simulation_seconds = 24 * 3600  # Full day in seconds
num_steps = total_simulation_seconds // time_step  # Total iterations

# Simulation loop
delta_seconds = 0
for step in range(int(num_steps)):
    rate(60)  # Controls the simulation speed (adjust if needed)
    
    # Update observer time
    #observer.date = ephem.Date(observer.date + ephem.second * time_step)

    # Compute new sun position
    #sun_direction = get_sun_vector(observer)

    # Update light direction and Sun sphere position
    #sun_light.direction = sun_direction
    #sun_sphere.pos = sun_direction * 10

    sun.update(delta_seconds)

    delta_seconds += time_step
    print('Delta Seconds', delta_seconds)
    #print(f"Time: {observer.date} | Sun Dir: {sun_direction}")

print("Simulation Complete!")




