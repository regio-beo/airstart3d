
import math
import ephem
from datetime import datetime

from vpython import distant_light, sphere, vector, color

class Sun:

    def __init__(self, date, lat, lon, render_sun=False):
        self.date = date 
        self.render_sun = render_sun

        # create observer
        self.observer = ephem.Observer()
        self.observer.lat, self.observer.lon = lat, lon  # kleine Scheidegg
        self.observer.elevation = 700
        self.start_date = ephem.Date(date)  # Start time (UTC)
        self.observer.date = self.start_date

        # init scene:
        if self.render_sun:
            self.sun_light = distant_light(direction=vector(0, 1, 0), color=color.white*0.9)        
            self.sun_sphere = sphere(pos=vector(0, 1, 0) * 100, radius=25, color=color.yellow, emissive=True)
        self.update(0)

        
    def update(self, delta_seconds):
        #update observer
        self.observer.date = ephem.Date(self.start_date + ephem.second * delta_seconds)        
        direction = self.get_sun_direction()
        if self.render_sun:
            self.sun_light.direction = direction        
            self.sun_sphere.pos = direction * 1000


    def get_sun_direction(self):
        sun = ephem.Sun(self.observer)
        azimuth = float(sun.az)  # Convert from radians
        altitude = float(sun.alt)

        # Convert azimuth/altitude to a unit vector (flipped z-axis for right-hand system)
        x = math.cos(altitude) * math.sin(azimuth)
        y = math.sin(altitude)
        z = -math.cos(altitude) * math.cos(azimuth)  # Flip z for right-hand system

        return vector(x, y, z).norm()  # Normalize to unit vector