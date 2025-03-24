
from datetime import datetime

from vpython import scene, color, rate, quad, textures, sleep, vector, vertex

from airstart3d.plot_3d import plot3D, create_axes
from airstart3d.texture import *
from airstart3d.sun import Sun
from airstart3d.elevation import read_elevation_data_32632

# Playground for the elevation stuff

if __name__ == '__main__':


    # Create a quad with an initial texture
    v0 = vertex(pos=vector(-1, -1, 0))
    v1 = vertex(pos=vector(1, -1, 0))
    v2 = vertex(pos=vector(1, 1, 0))
    v3 = vertex(pos=vector(-1, 1, 0))
    quad1 = quad(vs=[v0, v1, v2, v3], texture=textures.wood)

    # Wait and then change the texture
    #sleep(2)
    #quad1.texture = textures.metal  # Change the texture at runtime
    #sleep(5)

    # use Sun only
    scene.lights = []
    scene.ambient = color.white * 0.05
    sun = Sun(datetime(2025, 1, 3, 7, 0), '46.5', '7.9', True)

    # add axes
    create_axes()

    # now create a elevation tile:
    # x, y at top_left?
    #x=420011.130248355
    #y=5170090.784371755  
    # Airstart Region  
    x = 420011-2000
    y = 5170090
    width = 5000
    elevation_data = read_elevation_data_32632(x, y, width)


    # Render Texture:
    # Render Contour Tile:
    #texture = ContourTexture(x, y, elevation_data)
    #url = texture.render()

    # Slope Texture
    #texture = SlopeTexture(x, y, elevation_data)
    #texture.render()

    # Curvature Texture
    #texture = CurvatureTexture(x, y, elevation_data)
    #texture.render()

    # Thermal Texture
    #sun = Sun(datetime(2025, 3, 8, 11, 00), '46.6', '7.5', False)
    texture = ThermalTexture(x, y, elevation_data, skip_d8=True)
    url = texture.render(sun.get_sun_direction())

    t = 0
    L = elevation_data.shape[0]
    w2 = width//2
    #plot_triggers(x, y, elevation_data, vector(-w2, -2000, -w2), width) # not provided at the moment
    p = plot3D(lambda x, y: elevation_data[x, y]-2500, L, -w2, w2, -w2, w2, 0, 1000, texture=url)

    
    total_time = 12*3600
    speedup = 30*60 # seconds per real-second
    fps = 1
    dt = 1/fps * speedup

    while t < total_time:
    #while t == 0:
        rate(fps) # 30 fps:

        # update sun
        sun.update(t)

        # update texture
        url = texture.render(sun.get_sun_direction())
        print(f'updated: {url}')
        p.update_texture(url)

        t += dt






    
