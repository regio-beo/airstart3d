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
layout (location = 1) in vec3 normal;

out vec3 fragColor;
out vec3 fragNormal;

void main()
{
    // compute sun direction for frag color
    gl_Position = vec4(position, 1.0);
    fragColor = position+0.5;
    fragNormal = normal;

}
"""

# Fragment Shader
FRAGMENT_SHADER = """
#version 330 core
in vec3 fragColor;
in vec3 fragNormal;

uniform vec3 sunDirection;

out vec4 color;
void main()
{
    vec3 redish = vec3(0.8, 0.3, 0.3);

    float diff = max(dot(sunDirection, fragNormal), 0.0);
    //color = vec4(fragColor*diff, 1.0);        
    color = vec4(diff*redish, 1.0);

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

    
    rows = 3+2 #elevation_data.shape[0]
    cols = 3+2 #elevation_data.shape[1]
    max_elevation = np.max(elevation_data)
    vertices = np.zeros((rows, cols, 3))
    for row in range(rows):
        for col in range(cols):      
            
            vertex = [col/cols -0.5, -row/rows + 0.5, elevation_data[row, col]/max_elevation]
            vertices[row, col] = vertex

            #v1 = [col/cols - 0.5, -row/rows + 0.5, elevation_data[row, col]/max_elevation]
            #v2 = [col/cols - 0.5, -((row+1)/rows) + 0.5, elevation_data[row+1, col]/max_elevation]
            #v3 = [(col+1)/cols - 0.5, -((row+1)/rows) + 0.5, elevation_data[row+1, col+1]/max_elevation]
            #v4 = [(col+1)/cols - 0.5, -row/rows + 0.5, elevation_data[row, col+1]/max_elevation]            

            #for v in [v1, v2, v4, v2, v3, v4]:
            #    vertices += v

    # compute normals
    normals = np.zeros((rows-1, cols-1, 3))    
    for row in range(rows-1):
        for col in range(cols-1):
            v = vertices[row, col]
            a = vertices[row+1, col]
            b = vertices[row, col+1]
            n = np.cross(a, b)
            n = n/np.linalg.norm(n)
            normals[row, col] = n
    
    # Transform into GL Structures:
    gl_vertices = []    
    for row in range(rows-2):
        for col in range(cols-2):

            # get vertices:
            v1 = vertices[row, col]
            v2 = vertices[row+1, col]
            v3 = vertices[row+1, col+1]
            v4 = vertices[row, col+1]

            # get normals:
            n1 = normals[row, col]
            n2 = normals[row+1, col]
            n3 = normals[row+1, col+1]
            n4 = normals[row, col+1]

            print('n1', n1)

            # interleave normals
            for v in [v1, n1, v2, n2, v4, n4, v2, n2, v3, n3, v4, n4]:
                gl_vertices += list(v)            
    
            

    #print(vertices)
    #print(f'use {len(vertices)} vertices -> {int(len(vertices)/9)} triangles')

    #vertices = [
    #    -0.5, -0.5, 0.0,
    #     0.5, -0.5, 0.0,
    #     0.0,  0.5, 0.0
    #]

    num_vertices = int(len(gl_vertices)/6)
    gl_vertices = (GLfloat * len(gl_vertices))(*gl_vertices)
    

    
    
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(gl_vertices), gl_vertices, GL_STATIC_DRAW)
    
    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
    glEnableVertexAttribArray(0)

    # Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
    glEnableVertexAttribArray(1)
    
    running = True
    time = 0
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
        
        # set sun 
        sun_direction = sun.update(time*100) # 10 milliseconds        
        glUniform3f(unf_sun_direction, sun_direction.x, sun_direction.y, sun_direction.z)

        glClear(GL_COLOR_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, num_vertices)
        
        pygame.display.flip()
        pygame.time.wait(10)
        time += 10/1000
    
    pygame.quit()

if __name__ == "__main__":
    main()
