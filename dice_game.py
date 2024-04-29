import pygame
from sympy import nsolve, solve, symbols, Eq, diff
import sympy
from math import sin, cos, pi, sqrt, copysign
import random
import numpy as np
import multiprocessing as mp

def shift(points, displacement=(0,0,0)):
    output = []
    for p in points:
        output.append(tuple(p[i]+displacement[i] for i in range(3)))
    return output

def scale(points, factors=(1,1,1)):
    output = []
    if type(factors) == tuple or type(factors) == list:
        for p in points:
            output.append(tuple(p[i]*factors[i] for i in range(3)))
    else:
        for p in points:
            output.append(tuple(p[i]*factors for i in range(3)))
    return output

def rotate(points, angles=(0,0,0)):
    output = []
    for p in points:
        p = (p[0],p[1]*cos(angles[0])-p[2]*sin(angles[0]),p[1]*sin(angles[0])+p[2]*cos(angles[0]))
        p = (p[0]*cos(angles[1])-p[2]*sin(angles[1]),p[1],p[0]*sin(angles[1])+p[2]*cos(angles[1]))
        p = (p[0]*cos(angles[2])-p[1]*sin(angles[2]),p[0]*sin(angles[2])+p[1]*cos(angles[2]),p[2])
        output.append(p)
    return output

class Mass:
    def __init__(self, mass=1, pos=(0,0,0), vel=(0,0,0), acc=(0,0,0)):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.acc = acc
    def simulate(self, dt):
        acc = self.acc
        vel = self.vel
        pos = self.pos
        self.vel = (vel[0]+acc[0]*dt, vel[1]+acc[1]*dt, vel[2]+acc[2]*dt)
        self.pos = (pos[0]+vel[0]*dt, pos[1]+vel[1]*dt, pos[2]+vel[2]*dt)
    def zero_forces(self):
        self.acc = (0,0,0)
    def apply(self, force=(0,0,0)):
        acc = self.acc
        m = self.mass
        self.acc = (acc[0]+force[0]/m, acc[1]+force[1]/m, acc[2]+force[2]/m)
    def distance(self, other):
        return sqrt(sum((self.pos[i]-other.pos[i])**2 for i in range(3)))

def dot(vector1, vector2):
    return sum(x*vector2[i] for i,x in enumerate(vector1))
class Spring:
    def __init__(self, m1, m2, length=1, k=1):
        self.m1 = m1
        self.m2 = m2
        self.length = length
        self.k = k
    def apply(self):
        m1 = self.m1
        m2 = self.m2
        distance = m1.distance(m2)
        if self.k == float('inf'):
            diff = distance-self.length
            mass = m1.mass + m2.mass
            vec = (m2.pos[0]-m1.pos[0], m2.pos[1]-m1.pos[1], m2.pos[2]-m1.pos[2])
            mag = sqrt(dot(vec, vec))
            unit = (vec[0]/mag, vec[1]/mag, vec[2]/mag)
            m1.pos = (m1.pos[0]+unit[0]*diff*m2.mass/mass,m1.pos[1]+unit[1]*diff*m2.mass/mass,m1.pos[2]+unit[2]*diff*m2.mass/mass)
            unit = (-unit[0],-unit[1],-unit[2])
            m2.pos = (m2.pos[0]+unit[0]*diff*m1.mass/mass,m2.pos[1]+unit[1]*diff*m1.mass/mass,m2.pos[2]+unit[2]*diff*m1.mass/mass)
            return
        sign = copysign(1,distance-self.length)
        vector = (sign*self.k*(m2.pos[0]-m1.pos[0]), sign*self.k*(m2.pos[1]-m1.pos[1]), sign*self.k*(m2.pos[2]-m1.pos[2]))
        m1.apply(vector)
        vector = (-vector[0], -vector[1], -vector[2])
        m2.apply(vector)

def gravity(mass, dt):
    mass.apply((0,-9.8*mass.mass*dt,0))

def ground(mass, dt):
    if mass.pos[1] < 0:
        mass.pos = (mass.pos[0], 0, mass.pos[2])
        mass.vel = (mass.vel[0], -mass.vel[1], mass.vel[2])
        # friction
        mass.vel = (mass.vel[0]*0.5**dt, mass.vel[1], mass.vel[2]*0.5**dt)

def drag(mass, dt):
    mass.vel = (mass.vel[0]*0.995**dt, mass.vel[1]*0.995**dt, mass.vel[2]*0.995**dt)

class Camera:
    def __init__(self):
        self.origin = (0,0,0)
        self.x_vector = (1,0,0)
        self.y_vector = (0,1,0)
        self.focal = (0,0,-1)
        self.zoom = 1
    def project(self, point):
        o = self.origin
        x = self.x_vector
        y = self.y_vector
        f = self.focal
        p = point
        '''
        alpha, beta, gamma = symbols("alpha beta gamma")
        eqs = [Eq(o[i] + alpha*x[i] + beta*y[i], gamma*f[i] + (1-gamma)*p[i]) for i in range(3)]
        solution = solve(eqs, dict=True)[0]
        return (solution[alpha], solution[beta])
        '''
        #o[i] + alpha*x[i] + beta*y[i] = gamma*f[i] + (1-gamma)*p[i]
        #o[i] + alpha*x[i] + beta*y[i] = p[i] + gamma*f[i] - gamma*p[i]
        #o[i] + alpha*x[i] + beta*y[i] = p[i] + (f[i]-p[i])*gamma
        #x[i]*alpha + y[i]*beta + (p[i]-f[i])*gamma = p[i] - o[i]
        a = np.array([[x[i], y[i], p[i]-f[i]] for i in range(3)], dtype=np.float64)
        b = np.array([p[i]-o[i] for i in range(3)], dtype=np.float64)
        try:
            x = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return None, None
        x = x.tolist()
        return x[0]*self.zoom, x[1]*self.zoom
    def move(self, displacement):
        self.origin = tuple(self.origin[i]+displacement[i] for i in range(3))
        self.focal = tuple(self.focal[i]+displacement[i] for i in range(3))
    def rotate(self, angles):
        p = tuple(self.focal[i]-self.origin[i] for i in range(3))
        p, self.x_vector, self.y_vector = rotate([p, self.x_vector, self.y_vector], angles)
        self.focal = tuple(p[i]+self.origin[i] for i in range(3))
    def look(self, point):
        vector1 = (self.origin[0]-point[0], self.origin[1]-point[1], self.origin[2]-point[2])
        mag1 = sqrt(dot(vector1,vector1))
        unit = (vector1[0]/mag1,vector1[1]/mag1,vector1[2]/mag1)
        vector2 = (self.focal[0]-self.origin[0], self.focal[1]-self.origin[1], self.focal[2]-self.origin[2])
        mag2 = sqrt(dot(vector2,vector2))
        focal = (unit[0]*mag2+self.origin[0],unit[1]*mag2+self.origin[1],unit[2]*mag2+self.origin[2])
        theta = symbols("theta1 theta2 theta3")
        point = symbols("point1 point2 point3")
        focal[1], self.focal[1]
        f = vector2
        p1 = (f[0],f[1]*sympy.cos(theta[0])-f[2]*sympy.sin(theta[0]),f[1]*sympy.sin(theta[0])+f[2]*sympy.cos(theta[0]))
        p2 = (p1[0]*sympy.cos(theta[1])-p1[2]*sympy.sin(theta[1]),p1[1],p1[0]*sympy.sin(theta[1])+p1[2]*sympy.cos(theta[1]))
        p3 = (p2[0]*sympy.cos(theta[2])-p2[1]*sympy.sin(theta[2]),p2[0]*sympy.sin(theta[2])+p2[1]*sympy.cos(theta[2]),p2[2])
        eqs = [Eq(focal[i],p3[i]+self.origin[i]) for i in range(3)]
        try:
            solutions = nsolve(eqs, theta, (0.1,0.1,0.1), dict=True)
            solution = solutions[0]
        except (ValueError, ZeroDivisionError):
            print("error")
            return
        self.x_vector, self.y_vector = rotate([self.x_vector, self.y_vector], tuple(solution[theta[i]] for i in range(3)))
        self.focal = focal
    def forward_vector(self):
        vec = tuple(self.origin[i]-self.focal[i] for i in range(3))
        mag = sqrt(dot(vec,vec))
        return tuple(vec[i]/mag for i in range(3))

class Polyhedron:
    def __init__(self):
        self.verts = set()
        self.edges = set()
        self.faces = set()
    # return sequential points on the face
    def circuit(face):
        if not len(face):
            return []
        circuit = []
        previous = frozenset()
        edge_lookup = dict()
        for edge in face:
            for point in edge:
                if point not in edge_lookup:
                    edge_lookup[point] = set()
                edge_lookup[point].add(edge)
        start = list(face)[0]
        previous = start
        point = list(start)[0]
        circuit.append(point)
        current = list(edge_lookup[point] - set([start]))[0]
        while current != start:
            point = list(current - previous)[0]
            circuit.append(point)
            previous = current
            current = list(edge_lookup[point] - set([current]))[0]
        return circuit
    # returns any edge that intersects the line segment between p1 and p2
    def intersect(self, p1, p2):
        output = []
        for edge in self.edges:
            p3, p4 = tuple(edge)
            alpha, beta = symbols("alpha beta")
            eqs = [Eq(alpha*p1[i]+(1-alpha)*p2[i], beta*p3[i]+(1-beta)*p4[i]) for i in range(3)]
            solutions = solve(eqs, dict=True)
            if len(solutions):
                alpha, beta = solutions[0][alpha], solutions[0][beta]
                if alpha >= 0 and alpha <= 1 and beta >= 0 and beta <= 1:
                    output.append(edge)
        return output
    def inside_triangle(triangle, point):
        triangle = list(triangle)
        alpha, beta = symbols("alpha beta")
        print(triangle)
        exprs = [alpha*triangle[0][i]+(1-alpha)*triangle[1][i] for i in range(3)]
        exprs = [beta*x+(1-beta)*triangle[2][i] for i,x in enumerate(exprs)]
        eqs = [Eq(x,point[i]) for i,x in enumerate(exprs)]
        solutions = solve(eqs)
        if len(solutions):
            alpha, beta = solutions[0][alpha], solutions[0][beta]
            if alpha >= 0 and alpha <= 1 and beta >= 0 and beta <= 1:
                return True
        return False
    def triangulation(face):
        circuit = Polyhedron.circuit(face)
        output = []
        for i,x in enumerate(circuit):
            for j,y in enumerate(circuit):
                for k,z in enumerate(circuit):
                    points = set([x,y,z])
                    if len(points) != 3:
                        continue
                    '''
                    for w in circuit:
                        if w not in points and Polyhedron.inside_triangle(points, w):
                            continue
                    '''
                    output.append(points)
                    polygons = (circuit[k:]+circuit[:i+1], circuit[i:j+1], circuit[j:k+1])
                    print(polygons)
                    for polygon in polygons:
                        if len(polygon) == 3:
                            output.append(set(polygon))
                        else:
                            sub_face = frozenset([frozenset([w,polygon[(l+1)%len(polygon)]]) for l,w in enumerate(polygon)])
                            if len(sub_face) > 2:
                                output.extend(Polyhedron.triangulation(sub_face))
                    return output
        return output
    # Cast a ray from position in direction, return where on the polyhedron it hits
    def ray(self, position, direction):
        output = []
        print(self.faces, "faces")
        for face in self.faces:
            for triangle in Polyhedron.triangulation(face):
                triangle = list(triangle)
                alpha, beta, gamma = symbols("alpha beta gamma")
                exprs = [alpha*triangle[0][i]+(1-alpha)*triangle[1][i] for i in range(3)]
                exprs = [beta*x+(1-beta)*triangle[2][i] for i,x in enumerate(exprs)]
                eqs = [Eq(expr,position[i]+direction[i]*gamma) for i,expr in enumerate(exprs)]
                '''
                a,b,g = 0,0,0
                exprs = [(x-position[i]+direction[i]*gamma)**2 for i,x in enumerate(exprs)]
                for i in range(1000):
                    da = diff(sum(exprs), alpha).subs(alpha,a).subs(beta,b).subs(gamma,g)
                    db = diff(sum(exprs), beta).subs(alpha,a).subs(beta,b).subs(gamma,g)
                    dg = diff(sum(exprs), gamma).subs(alpha,a).subs(beta,b).subs(gamma,g)
                    a -= da*0.0001
                    b -= db*0.0001
                    g -= dg*0.0001
                    print(sum(exprs).subs(alpha,a).subs(beta,b).subs(gamma,g),a,b,g,da,db,dg)
                sys.exit()
                '''
                #beta*(alpha*triangle[0][i]+(1-alpha)*triangle[1][i])+(1-beta)*triangle[2][i]
                try:
                    solutions = nsolve(eqs, (alpha,beta,gamma), (1,1,1), dict=True)
                except ValueError:
                    continue
                alpha, beta, gamma = solutions[0][alpha], solutions[0][beta], solutions[0][gamma]
                #(beta*(alpha*triangle[0][i]+(1-alpha)*triangle[1][i])+(1-beta)*triangle[2][i]-position[i]+direction[i]*gamma)**2
                if alpha >= 0 and alpha <= 1 and beta >= 0 and beta <= 1:
                    point = tuple(alpha*triangle[0][i]+(1-alpha)*triangle[1][i] for i in range(3))
                    point = tuple(beta*x+(1-beta)*triangle[2][i] for i,x in enumerate(point))
                    output.append((gamma, point, face))
                    double_break = True
                    break
        return sorted(output)

    
def cross(vector1, vector2):
    return (vector1[1]*vector2[2]-vector1[2]*vector2[1], vector1[2]*vector2[0]-vector1[0]*vector2[2],vector1[0]*vector2[1]-vector1[1]*vector2[0])
class Body:
    def __init__(self, mass, pos=(0,0,0), polyhedron=None):
        self.mass = mass
        self.pos = pos
        self.vel = (0,0,0)
        self.acc = (0,0,0)
        self.rpos = (0,0,0)
        self.rvel = (0,0,0)
        self.racc = (0,0,0)
        self.polyhedron = polyhedron
    def apply(self, force=(0,0,0), location=(0,0,0)):
        pos = self.pos
        loc = location
        vec = (pos[0]-loc[0],pos[1]-loc[1],pos[2]-loc[2])
        mag = sqrt(dot(vec,vec))
        if mag != 0:
            unit = (vec[0]/mag,vec[1]/mag,vec[2]/mag)
            f = dot(force,unit)
            f = (f*unit[0], f*unit[1], f*unit[2])
            r = (force[0]-f[0],force[1]-f[1],force[2]-f[2])
        else:
            f = force
            r = (0,0,0)
        #r = ((r[1]+r[2])/2,(r[0]+r[2])/2,(r[1]+r[0])/2)
        #r = ((copysign(r[2],vec[1])-copysign(r[1],vec[2]))*-1*sqrt(vec[1]**2+vec[2]**2),(copysign(r[2],vec[0])-copysign(r[0],vec[2]))*-1*sqrt(vec[0]**2+vec[2]**2),(copysign(r[1],vec[0])-copysign(r[0],vec[1]))*-1*sqrt(vec[1]**2+vec[0]**2))
        #r = ((copysign(r[2],vec[1])-copysign(r[1],vec[2]))*-1*sqrt(vec[1]**2+vec[2]**2),0,(copysign(r[1],vec[0])-copysign(r[0],vec[1]))*-1*sqrt(vec[1]**2+vec[0]**2))
        #r = (r[0]*sqrt(vec[1]**2+vec[2]**2),r[1]*sqrt(vec[0]**2+vec[2]**2),r[2]**sqrt(vec[0]**2+vec[1]**2))
        r = cross(r,vec)
        acc = self.acc
        m = self.mass
        self.acc = (acc[0]+f[0]/m,acc[1]+f[1]/m,acc[2]+f[2]/m)
        Ix = sum((p[1]-pos[1])**2+(p[2]-pos[2])**2 for p in self.polyhedron.verts)*m/len(self.polyhedron.verts)
        Iy = sum((p[0]-pos[0])**2+(p[2]-pos[2])**2 for p in self.polyhedron.verts)*m/len(self.polyhedron.verts)
        Iz = sum((p[0]-pos[0])**2+(p[1]-pos[1])**2 for p in self.polyhedron.verts)*m/len(self.polyhedron.verts)
        racc = self.racc
        self.racc = (racc[0]+r[0]/Ix,racc[1]+r[1]/Iy,racc[2]+r[2]/Iz)
        print(force, location, acc, self.racc, Ix, Iy, Iz)
    def simulate(self, dt):
        acc = self.acc
        vel = self.vel
        pos = self.pos
        self.vel = (vel[0]+acc[0]*dt, vel[1]+acc[1]*dt, vel[2]+acc[2]*dt)
        self.pos = (pos[0]+vel[0]*dt, pos[1]+vel[1]*dt, pos[2]+vel[2]*dt)
        points = {p:(p[0]+vel[0]*dt,p[1]+vel[1]*dt,p[2]+vel[2]*dt) for p in self.polyhedron.verts}
        edges = {edge:frozenset([points[p] for p in edge]) for edge in self.polyhedron.edges}
        polyhedron = Polyhedron()
        polyhedron.verts = {points[key] for key in points}
        polyhedron.edges = {edges[key] for key in edges}
        polyhedron.faces = {frozenset([edges[edge] for edge in face]) for face in self.polyhedron.faces}
        racc = self.racc
        rvel = self.rvel
        rpos = self.rpos
        self.rvel = (rvel[0]+racc[0]*dt, rvel[1]+racc[1]*dt, rvel[2]+racc[2]*dt)
        self.rpos = (rpos[0]+rvel[0]*dt, rpos[1]+rvel[1]*dt, rpos[2]+rvel[2]*dt)
        points = list(polyhedron.verts)
        shifted = shift(points, (-self.pos[0], -self.pos[1], -self.pos[2]))
        rotated = rotate(shifted, (rvel[0]*dt, rvel[1]*dt, rvel[2]*dt))
        shifted = shift(rotated, self.pos)
        points = {x:shifted[i] for i,x in enumerate(points)}
        edges = {edge:frozenset([points[p] for p in edge]) for edge in polyhedron.edges}
        faces = {frozenset([edges[edge] for edge in face]) for face in polyhedron.faces}
        polyhedron = Polyhedron()
        polyhedron.verts = {points[key] for key in points}
        polyhedron.edges = {edges[key] for key in edges}
        polyhedron.faces = faces
        self.polyhedron = polyhedron
    def zero_forces(self):
        self.acc = (0,0,0)
        self.racc = (0,0,0)
    def gravity(self, dt):
        self.apply((0,-9.8*dt*self.mass,0), self.pos)
    def ground(self, dt):
        epsilon = .001
        points = sorted([(p[1], p) for p in self.polyhedron.verts])
        count = len([p for p in points if p[0]<epsilon])
        print("count: ", count)
        if count > 0:
            point = (sum(p[1][0] for p in points if p[0]<epsilon)/count,sum(p[1][1] for p in points if p[0]<epsilon)/count,sum(p[1][2] for p in points if p[0]<epsilon)/count)
        if points[0][0] < epsilon:
            self.apply((0,9.8*dt*self.mass,0), point)
            #self.vel = (self.vel[0]*0.1**dt,0, self.vel[2]*0.1**dt)
            self.vel = (self.vel[0]*0.1**dt,-self.vel[1]*0.5, self.vel[2]*0.1**dt)
            #self.vel = (self.vel[0]*0.1**dt,0, self.vel[2]*0.1**dt)
            #self.vel = (self.vel[0],0, self.vel[2])
            self.apply((-self.vel[0]*dt,-self.vel[1]*dt, -self.vel[2]*dt),point)
            #self.vel = (0,0,0)
            self.rvel = (self.rvel[0]*.5**dt, self.rvel[1]*.5**dt, self.rvel[2]*.5**dt)
            #self.rvel = (self.rvel[0]*.5**dt, self.rvel[1], self.rvel[2]*.5**dt)
            #self.rvel = (0,0,0)
            #self.vel = (0,self.vel[1],0)
        if points[0][0] < 0:
            pos = self.pos
            delta = -points[0][0]
            self.pos = (pos[0], pos[1]+delta, pos[2])
            polyhedron = Polyhedron()
            points = list(self.polyhedron.verts)
            shifted = shift(points, (0, delta, 0))
            points = {x:shifted[i] for i,x in enumerate(points)}
            edges = {edge:frozenset([points[p] for p in edge]) for edge in self.polyhedron.edges}
            faces = {frozenset([edges[edge] for edge in face]) for face in self.polyhedron.faces}
            polyhedron.verts = {points[key] for key in points}
            polyhedron.edges = {edges[key] for key in edges}
            polyhedron.faces = faces
            self.polyhedron = polyhedron

def coplanar(points):
    if len(points) < 4:
        return True
    points = list(points)
    '''
    alpha, beta = symbols("alpha beta")
    exprs = [alpha*(points[1][i]-points[0][i])+beta*(points[2][i]-points[0][i]) for i in range(3)]
    for p in points[3:]:
        eqs = [Eq(expr,p[i]-points[0][i]) for i,expr in enumerate(exprs)]
        if not len(solve(eqs)):
            return False
    '''
    for p in points[3:]:
        a = np.array([[points[1][i]-points[0][i],points[2][i]-points[0][i]] for i in range(3)])
        b = np.array([p[i]-points[0][i] for i in range(3)])
        x, res, rank, s = np.linalg.lstsq(a, b)
        #print(a, b, x, res)
        if not np.allclose(np.dot(a, x), b):
            return False
    return True

def construct_faces(edges):
    edge_lookup = dict()
    for edge in edges:
        for point in edge:
            if point not in edge_lookup:
                edge_lookup[point] = set()
            edge_lookup[point].add(edge)
    cycles = set()
    queue = [(edge,) for edge in edges]
    while len(queue):
        #print([len(x) for x in queue], [len(x) for x in cycles])
        path = queue.pop()
        for point in path[-1]:
            if len(path) == 1 or point not in path[-2]:
                for edge in edge_lookup[point]:
                    if edge not in path:
                        new_path = path + (edge,)
                        points = {point for edge in new_path for point in edge}
                        if len(new_path) > 2:
                            if coplanar(points):
                                if len(new_path[0] & new_path[-1]):
                                    cycles.add(frozenset(new_path))
                                else:
                                    queue.append(new_path)
                        else:
                            queue.append(new_path)
    return cycles

def get_cube(displacement=(0,0,0), factors=(1,1,1), angles=(0,0,0)):
    points = [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
    points.extend([(p[0],p[1],1) for p in points])
    center = (0.5,0.5,0.5)
    points = shift(points, (-center[0], -center[1], -center[2]))
    points = scale(points, factors)
    points = rotate(points, angles)
    points = shift(points, center)
    points = shift(points, displacement)
    edges = [(points[i],points[(i+1)%4]) for i in range(4)]
    edges.extend([(points[4+i],points[4+(i+1)%4]) for i in range(4)])
    edges.extend([(points[i],points[i+4]) for i in range(4)])
    polyhedron = Polyhedron()
    polyhedron.verts = set(points)
    polyhedron.edges = set([frozenset(x) for x in edges])
    polyhedron.faces = construct_faces(polyhedron.edges)
    print(len(polyhedron.faces), "len faces")
    return polyhedron

class Crosshair():
    def __init__(self):
        self.pos = (0,0)
    def draw(self, pygame, screen):
        screen_width, screen_height = screen.get_size()
        p = [(self.pos[0]+screen_width/2+x, -self.pos[1]+screen_height/2) for x in (-10,10)]
        pygame.draw.line(screen, "white", p[0], p[1])
        p = [(self.pos[0]+screen_width/2, -self.pos[1]+screen_height/2+x) for x in (-10,10)]
        pygame.draw.line(screen, "white", p[0], p[1])


class Grid():
    def __init__(self):
        self.origin = (0,0,0)
        self.w_vector = (1,0,0)
        self.h_vector = (0,0,1)
        self.width = 10
        self.height = 10
    def draw(self, pygame, screen):
        screen_width, screen_height = screen.get_size()
        '''
        for i in range(self.width):
            for j in range(self.height):
                p = [tuple() for k in range(4)]
                p[0] = tuple(self.origin[k]+self.w_vector[k]*i+self.h_vector[k]*j for k in range(3))
                p[1] = tuple(self.origin[k]+self.w_vector[k]*(i+1)+self.h_vector[k]*j for k in range(3))
                p[2] = tuple(self.origin[k]+self.w_vector[k]*(i+1)+self.h_vector[k]*(j+1) for k in range(3))
                p[3] = tuple(self.origin[k]+self.w_vector[k]*i+self.h_vector[k]*(j+1) for k in range(3))
                p = [camera.project(x) for x in p]
                p = [(x[0]+screen_width/2,-x[1]+screen_height/2) for x in p]
                for k in range(4):
                    pygame.draw.line(screen, "white", p[k], p[(k+1)%4])
        '''
        for i in range(self.width):
            p = [tuple() for k in range(2)]
            p[0] = tuple(self.origin[k]+self.w_vector[k]*i+self.h_vector[k] for k in range(3))
            p[1] = tuple(self.origin[k]+self.w_vector[k]*i+self.h_vector[k]*self.height for k in range(3))
            p = [camera.project(x) for x in p]
            p = [(x[0]+screen_width/2,-x[1]+screen_height/2) for x in p]
            pygame.draw.line(screen, "white", p[0], p[1])
        for i in range(self.height):
            p = [tuple() for k in range(2)]
            p[0] = tuple(self.origin[k]+self.w_vector[k]+self.h_vector[k]*i for k in range(3))
            p[1] = tuple(self.origin[k]+self.w_vector[k]*self.width+self.h_vector[k]*i for k in range(3))
            p = [camera.project(x) for x in p]
            try:
                p = [(x[0]+screen_width/2,-x[1]+screen_height/2) for x in p]
            except TypeError:
                continue
            pygame.draw.line(screen, "white", p[0], p[1])




camera = Camera()
camera.focal = (0,0,-10)
camera.zoom = 100
print(camera.project((1,1,1)))


print(rotate([(1,1,1)], (pi,0,0)))
cube = get_cube()
print('hi',cube.verts)

#angles = [0,0,0]
#masses = [Mass(mass=100, pos=(500,500,500)), Mass(mass=100, pos=(600,500,500))]
#springs = [Spring(masses[0], masses[1], length=99.99, k=10)]

#cube = get_cube(displacement=(5000,5000,100000), factors=10000, angles=tuple(2*pi*random.random() for i in range(3)))
#cube = get_cube(displacement=(500,500,100), factors=50, angles=tuple(2*pi*random.random() for i in range(3)))
#masses = [Mass(mass=100000,pos=x) for x in cube.verts]

#for mass in masses:
#    mass.vel = (100,-1000,0)
#links = [(x,y) for x in masses for y in masses if frozenset((x.pos,y.pos)) in cube.edges]
#for mass in masses:
#    mass.vel = (1000,-1000,0)
'''
for i in range(10):
    i1, i2, i3  = 0,0,0
    while i1 == i2 or i2 == i3 or i3 == i1:
        i1, i2, i3 = random.randrange(8), random.randrange(8), random.randrange(8)
    alpha = random.random()
    beta = random.random()
    pos = (alpha*masses[i1].pos[0]+(1-alpha)*masses[i2].pos[0],alpha*masses[i1].pos[1]+(1-alpha)*masses[i2].pos[1],alpha*masses[i1].pos[2]+(1-alpha)*masses[i2].pos[2])
    pos = (beta*pos[0]+(1-beta)*masses[i3].pos[0],beta*pos[1]+(1-beta)*masses[i3].pos[1],beta*pos[2]+(1-beta)*masses[i3].pos[2])
    masses.append(Mass(mass=100,pos=pos))
#masses2 = [Mass(mass=100, pos=( (x.pos[0]+y.pos[0])/2,(x.pos[1]+y.pos[1])/2,(x.pos[2]+y.pos[2])/2 )) for x in masses for y in masses if x != y]
#masses.extend(masses2)
'''
#springs = [Spring(masses[i], masses[j], length=masses[i].distance(masses[j]), k=1000) for i in range(len(masses)) for j in range(len(masses)) if i != j]
#springs = [Spring(masses[i], masses[j], length=masses[i].distance(masses[j]), k=float('inf')) for i in range(len(masses)) for j in range(len(masses)) if i != j]

cube = get_cube(displacement=(0,5,1), factors=2, angles=(2*pi*random.random(),2*pi*random.random(),2*pi*random.random()))
#cube = get_cube(displacement=(5,5,1), factors=2, angles=(0,0,2*pi*random.random()))
center = (sum(p[0] for p in cube.verts)/len(cube.verts),sum(p[1] for p in cube.verts)/len(cube.verts),sum(p[2] for p in cube.verts)/len(cube.verts))
body = Body(1, center, polyhedron=cube)
#body.apply((1000,0,0), list(cube.verts)[0])
body.rvel = (0,0,1)
#body.vel = (0,0,100)
grid = Grid()
camera_angular = [0,0,0]
camera_velocity = [0,0,0]
force = 0
space_down = False
crosshair = Crosshair()

pygame.init()
screen_width = 1280
screen_height = 720
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
pygame.mouse.set_visible(False)
pygame.mouse.set_pos((screen_width/2, screen_height/2))
running = True
framerate = 60
while running:
    body.zero_forces()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                camera_angular[1] = 0.025
            if event.key == pygame.K_d:
                camera_angular[1] = -0.025
            if event.key == pygame.K_w:
                camera_angular[0] = -0.025
            if event.key == pygame.K_s:
                camera_angular[0] = 0.025
            if event.key == pygame.K_e:
                camera_angular[2] = -0.025
            if event.key == pygame.K_q:
                camera_angular[2] = 0.025
            if event.key == pygame.K_UP:
                forward = camera.forward_vector()
                for i in range(3):
                    camera_velocity[i] = forward[i]
            if event.key == pygame.K_DOWN:
                forward = camera.forward_vector()
                for i in range(3):
                    camera_velocity[i] = -forward[i]
            if event.key == pygame.K_LEFT:
                for i in range(3):
                    camera_velocity[i] = -camera.x_vector[i]
            if event.key == pygame.K_RIGHT:
                for i in range(3):
                    camera_velocity[i] = camera.x_vector[i]
            if event.key == pygame.K_LSHIFT:
                for i in range(3):
                    camera_velocity[i] = camera.y_vector[i]
            if event.key == pygame.K_LCTRL:
                for i in range(3):
                    camera_velocity[i] = -camera.y_vector[i]
            if event.key == pygame.K_SPACE:
                space_down = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a:
                camera_angular[1] = 0
            if event.key == pygame.K_d:
                camera_angular[1] = -0
            if event.key == pygame.K_w:
                camera_angular[0] = -0
            if event.key == pygame.K_s:
                camera_angular[0] = 0
            if event.key == pygame.K_e:
                camera_angular[2] = -0
            if event.key == pygame.K_q:
                camera_angular[2] = 0
            if event.key == pygame.K_UP:
                camera_velocity = [0,0,0]
            if event.key == pygame.K_DOWN:
                camera_velocity = [0,0,0]
            if event.key == pygame.K_LEFT:
                camera_velocity = [0,0,0]
            if event.key == pygame.K_RIGHT:
                camera_velocity = [0,0,0]
            if event.key == pygame.K_LSHIFT:
                camera_velocity = [0,0,0]
            if event.key == pygame.K_LCTRL:
                camera_velocity = [0,0,0]
            if event.key == pygame.K_SPACE:
                space_down = False
                pos = list(camera.origin)
                for i in range(3):
                    pos[i] += camera.x_vector[i]*crosshair.pos[0]/camera.zoom
                    pos[i] += camera.y_vector[i]*crosshair.pos[1]/camera.zoom
                vector = tuple(pos[i]-camera.focal[i] for i in range(3))
                mag = sqrt(dot(vector,vector))
                vector = tuple(vector[i]/mag for i in range(3))
                hits = body.polyhedron.ray(pos,vector)
                if len(hits):
                    print(hits)
                    force_vector = tuple(vector[i]*force for i in range(3))
                    body.apply(force_vector, hits[0][1])
                force = 0
        if event.type == pygame.MOUSEMOTION:
            x,y = pygame.mouse.get_pos()
            x -= screen_width/2
            y = -(y-screen_height/2)
            crosshair.pos = (x,y)
        if event.type == pygame.MOUSEBUTTONDOWN:
            pass


    #print("rel", pygame.mouse.get_rel(), "pos", pygame.mouse.get_pos())
    #pygame.draw.circle(screen, "red", (0,0), 40)
    screen.fill("black")
    crosshair.draw(pygame, screen)
    for edge in body.polyhedron.edges:
        p1, p2 = tuple(edge)
        print(p1,p2)
        p1, p2 = camera.project(p1), camera.project(p2)
        p1 = (p1[0]*1+screen_width/2,p1[1]*-1+screen_height/2)
        p2 = (p2[0]*1+screen_width/2,p2[1]*-1+screen_height/2)
        pygame.draw.line(screen, "white", p1, p2)
    grid.draw(pygame,screen)
    pygame.display.flip()
    dT = clock.tick(60) / 1000
    dt = 1/framerate 
    body.gravity(dt*5)
    body.ground(dt*5)
    body.simulate(dt*5)
    camera.rotate(tuple(x*dt*100 for x in camera_angular))
    camera.move(tuple(x*dt*100 for x in camera_velocity))
    #camera.look(body.pos)
    if space_down:
        force += dt*100
    print("dT", dT, force)
    if dt > 0.5:
        sys.exit()

pygame.quit()
