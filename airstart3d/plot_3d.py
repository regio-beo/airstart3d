from vpython import *

def create_axes():
    # Add axes:
    L = 500
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

''' i will borrow that: '''
class plot3D:
    def __init__(self, f, L, xmin, xmax, ymin, ymax, zmin, zmax, texture=None):
        # The x axis is labeled y, the z axis is labeled x, and the y axis is labeled z.
        # This is done to mimic fairly standard practive for plotting
        #     the z value of a function of x and y.
        self.f = f
        self.texture = texture
        if not xmin: self.xmin = 0
        else: self.xmin = xmin
        if not xmax: self.xmax = 1
        else: self.xmax = xmax
        if not ymin: self.ymin = 0
        else: self.ymin = ymin
        if not ymax: self.ymax = 1
        else: self.ymax = ymax
        if not zmin: self.zmin = 0
        else: self.zmin = zmin
        if not zmax: self.zmax = 1
        else: self.zmax = zmax
        
        self.L = L
        #R = L/100
        #d = L-2
        #xaxis = cylinder(pos=vec(0,0,0), axis=vec(0,0,d), radius=R, color=color.yellow)
        #yaxis = cylinder(pos=vec(0,0,0), axis=vec(d,0,0), radius=R, color=color.yellow)
        #zaxis = cylinder(pos=vec(0,0,0), axis=vec(0,d,0), radius=R, color=color.yellow)
        #k = 1.02
        #h = 0.05*L
        #text(pos=xaxis.pos+k*xaxis.axis, text='x', height=h, align='center', billboard=True, emissive=True)
        #text(pos=yaxis.pos+k*yaxis.axis, text='y', height=h, align='center', billboard=True, emissive=True)
        #text(pos=zaxis.pos+k*zaxis.axis, text='z', height=h, align='center', billboard=True, emissive=True)
    
        self.vertices = []
        for x in range(L):
            for y in range(L):
                val = self.evaluate(x,y)
                x_rel = (x/L)*(self.xmax-self.xmin)+self.xmin
                y_rel = (y/L)*(self.ymax-self.ymin)+self.ymin
                self.vertices.append(self.make_vertex(x_rel, y_rel, x, y, val ))
        
        self.quads = []
        self.make_quads()
        self.make_normals()
        
    def evaluate(self, x, y):
        return self.f(x, y) # absolute evaluation
        # 
        #d = self.L-2
        #return (d/(self.zmax-self.zmin)) * (self.f(self.xmin+x*(self.xmax-self.xmin)/d, self.ymin+y*(self.ymax-self.ymin)/d)-self.zmin)

    def update_texture(self, texture):
        self.texture = texture

        for i in range(self.L*self.L):
            self.vertices[i].pos.y += 0.01 # move by small meter

        self.make_quads()
        scene.waitfor("redraw")
        print("Redraw complete")

    def make_quads(self):
        del self.quads[:] # delete all
        self.quads = []
        # Create the quad objects, based on the vertex objects already created.
        for x in range(self.L-2):
            for y in range(self.L-2):
                v0 = self.get_vertex(x,y)
                v1 = self.get_vertex(x+1,y)
                v2 = self.get_vertex(x+1, y+1)
                v3 = self.get_vertex(x, y+1)
                quad(vs=[v0, v1, v2, v3], texture=self.texture)
                self.quads.append(quad)
        
    def make_normals(self):
        # Set the normal for each vertex to be perpendicular to the lower left corner of the quad.
        # The vectors a and b point to the right and up around a vertex in the xy plance.
        for i in range(self.L*self.L):
            x = int(i/self.L)
            y = i % self.L
            if x == self.L-1 or y == self.L-1: continue
            v = self.vertices[i]
            a = self.vertices[i+self.L].pos - v.pos
            b = self.vertices[i+1].pos - v.pos
            v.normal = cross(a,b)
    
    def replot(self):
        for i in range(self.L*self.L):
            x = int(i/self.L)
            y = i % self.L
            v = self.vertices[i]
            v.pos.y = self.evaluate(x,y)
        self.make_normals()
                
    def make_vertex(self,x,y,tex_x, tex_y, value):        
        #texpos = vector(tex_x/self.L, tex_y/self.L, 0)
        texpos = vector(tex_y/self.L, 1.-tex_x/self.L, 0)
        return vertex(pos=vec(y,value,x), texpos=texpos, shininess=0.0, normal=vec(0,1,0))
        
    def get_vertex(self,x,y):
        return self.vertices[x*self.L+y]
        
    def get_pos(self,x,y):
        return self.get_vertex(x,y).pos

''' end of it '''