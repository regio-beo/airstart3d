import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

from datetime import datetime
from airstart3d.elevation import read_elevation_data_32632
from airstart3d.sun import Sun

# Vertex Shader
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 position;
out vec3 fragColor;

uniform vec3 sunDirection;

void main()
{
    // compute sun direction for frag color
    gl_Position = vec4(position, 1.0);
    fragColor = sunDirection;
}
"""

# Fragment Shader
FRAGMENT_SHADER = """
#version 330 core
in vec3 fragColor;
out vec4 color;
void main()
{
    color = vec4(fragColor, 1.0);        

}
"""

def create_shader_program():
    return compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

def main():

    # Read elevation data and sun direction
    x = 420011
    y = 5170090
    width = 1000
    elevation_data = read_elevation_data_32632(x, y, width)

    sun = Sun(datetime(2025, 3, 22, 11, 0), '46.5', '7.9', False)    


    # Run shader

    pygame.init()
    pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    shader = create_shader_program()

    # uniforms
    unf_sun_direction = glGetUniformLocation(shader, "sunDirection")

    glUseProgram(shader)

    # Debug
    #glDisable(GL_CULL_FACE)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    vertices = []
    rows = elevation_data.shape[0]
    cols = elevation_data.shape[1]
    max_elevation = np.max(elevation_data)
    for row in range(rows-1):
        for col in range(cols-1):            
            v1 = [col/cols - 0.5, -row/rows + 0.5, elevation_data[row, col]/max_elevation]
            v2 = [col/cols - 0.5, -((row+1)/rows) + 0.5, elevation_data[row+1, col]/max_elevation]
            v3 = [(col+1)/cols - 0.5, -((row+1)/rows) + 0.5, elevation_data[row+1, col+1]/max_elevation]
            v4 = [(col+1)/cols - 0.5, -row/rows + 0.5, elevation_data[row, col+1]/max_elevation]            

            for v in [v1, v2, v4, v2, v3, v4]:
                vertices += v

            # compute normals
            
            

    #print(vertices)
    #print(f'use {len(vertices)} vertices -> {int(len(vertices)/9)} triangles')

    #vertices = [
    #    -0.5, -0.5, 0.0,
    #     0.5, -0.5, 0.0,
    #     0.0,  0.5, 0.0
    #]

    num_vertices = int(len(vertices)/3)
    vertices = (GLfloat * len(vertices))(*vertices)
    
    
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    
    running = True
    time = 0
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
        
        # set sun 
        sun_direction = sun.update(time*1000) # 10 milliseconds
        #= sun.get_sun_direction()
        print(sun_direction)
        glUniform3f(unf_sun_direction, sun_direction.x, sun_direction.y, sun_direction.z)

        glClear(GL_COLOR_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, num_vertices)
        
        pygame.display.flip()
        pygame.time.wait(10)
        time += 10/1000
    
    pygame.quit()

if __name__ == "__main__":
    main()
