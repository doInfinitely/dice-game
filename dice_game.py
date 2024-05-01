import pygame
from sympy import nsolve, solve, symbols, Eq, diff
import sympy
from math import sin, cos, pi, sqrt, copysign
import random
import numpy as np
import multiprocessing as mp
import queue
from PIL import Image

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

def dot(vector1, vector2):
    return sum(x*vector2[i] for i,x in enumerate(vector1))

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
            if not np.allclose(np.dot(a, x), b, atol=0.1):
                return False
        return True
    def construct_faces(self):
        edge_lookup = dict()
        for edge_index, edge in enumerate(self.edges):
            for index in edge:
                point = self.verts[index]
                if point not in edge_lookup:
                    edge_lookup[point] = set()
                edge_lookup[point].add(edge_index)
        cycles = set()
        queue = [(index,) for index,edge in enumerate(self.edges)]
        while len(queue):
            #print([len(x) for x in queue], [len(x) for x in cycles])
            path = queue.pop()
            for index in self.edges[path[-1]]:
                if len(path) == 1 or self.verts[index] not in self.edges[path[-2]]:
                    point = self.verts[index]
                    for edge in edge_lookup[point]:
                        if edge not in path:
                            new_path = path + (edge,)
                            points = {self.verts[index] for edge_index in new_path for index in self.edges[edge_index]}
                            if len(new_path) > 2:
                                if Polyhedron.coplanar(points):
                                    if len(self.edges[new_path[0]] & self.edges[new_path[-1]]):
                                        cycles.add(frozenset(new_path))
                                    else:
                                        queue.append(new_path)
                            else:
                                queue.append(new_path)
        self.faces = list(cycles)
    def circuit(self, face_index):
        face = self.faces[face_index]
        if not len(face):
            return []
        circuit = []
        previous = frozenset()
        edge_lookup = dict()
        for edge_index in face:
            edge = self.edges[edge_index]
            for index in self.edges[edge_index]:
                point = self.verts[index]
                if point not in edge_lookup:
                    edge_lookup[point] = set()
                edge_lookup[point].add(edge)
        start = self.edges[list(face)[0]]
        previous = start
        point = self.verts[list(start)[0]]
        circuit.append(point)
        current = list(edge_lookup[point] - set([start]))[0]
        while current != start:
            point = self.verts[list(current - previous)[0]]
            circuit.append(point)
            previous = current
            current = list(edge_lookup[point] - set([current]))[0]
        return circuit
    # returns any edge that intersects the line segment between p1 and p2
    def intersect(self, p1, p2):
        output = []
        for edge in self.edges:
            p3, p4 = tuple(self.verts[index] for index in edge)
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
    def triangulation(circuit):
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
                    for polygon in polygons:
                        if len(polygon) == 3:
                            output.append(set(polygon))
                        else:
                                output.extend(Polyhedron.triangulation(polygon))
                    return output
        return output
    # Cast a ray from position in direction, return where on the polyhedron it hits
    def ray(self, position, direction):
        output = []
        for face_index,face in enumerate(self.faces):
            circuit = self.circuit(face_index)
            for triangle in Polyhedron.triangulation(circuit):
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

def get_cube(displacement=(0,0,0), factors=(1,1,1), angles=(0,0,0)):
    points = [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
    points.extend([(p[0],p[1],1) for p in points])
    center = (0.5,0.5,0.5)
    points = shift(points, (-center[0], -center[1], -center[2]))
    points = scale(points, factors)
    points = rotate(points, angles)
    points = shift(points, center)
    points = shift(points, displacement)
    edges = [(i,(i+1)%4) for i in range(4)]
    edges.extend([(4+i,4+(i+1)%4) for i in range(4)])
    edges.extend([(i,i+4) for i in range(4)])
    polyhedron = Polyhedron()
    polyhedron.verts = points
    polyhedron.edges = [frozenset(x) for x in edges]
    polyhedron.construct_faces()
    print(len(polyhedron.faces), "len faces")
    print(polyhedron.faces)
    return polyhedron

def get_octahedron(displacement=(0,0,0), factors=(1,1,1), angles=(0,0,0)):
    points = [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
    points.extend([(0.5,0.5,sqrt(1-sqrt(2*0.5**2))),(0.5,0.5,-sqrt(1-sqrt(2*0.5**2)))])
    center = (0.5,0.5,0)
    points = shift(points, (-center[0], -center[1], -center[2]))
    points = scale(points, factors)
    points = rotate(points, angles)
    points = shift(points, center)
    points = shift(points, displacement)
    edges = [(i,(i+1)%4) for i in range(4)]
    edges.extend([(i,4) for i in range(4)])
    edges.extend([(i,5) for i in range(4)])
    polyhedron = Polyhedron()
    polyhedron.verts = points
    polyhedron.edges = [frozenset(x) for x in edges]
    polyhedron.construct_faces()
    polyhedron.faces = [x for x in polyhedron.faces if len(x)==3]
    print(len(polyhedron.faces), "len faces")
    print(polyhedron.faces)
    return polyhedron

def get_tetrahedron(displacement=(0,0,0), factors=(1,1,1), angles=(0,0,0)):
    points = [(0,0.5,0),(0,-0.5,0),(sqrt(1-0.5**2),0,0),(sqrt(1-0.5**2)/2,0,sqrt(1-(1-0.5**2)))]
    center = (sqrt(1-0.5**2)/2,0,sqrt(1-(1-0.5**2))/2)
    points = shift(points, (-center[0], -center[1], -center[2]))
    points = scale(points, factors)
    points = rotate(points, angles)
    points = shift(points, center)
    points = shift(points, displacement)
    edges = [(i,j) for i in range(4) for j in range(i+1,4)]
    polyhedron = Polyhedron()
    polyhedron.verts = points
    polyhedron.edges = [frozenset(x) for x in edges]
    polyhedron.construct_faces()
    print(len(polyhedron.faces), "len faces")
    print(polyhedron.faces)
    return polyhedron

def get_equilateral_triangle():
    return [(0,0,0),(-0.5,-sqrt(1-0.5**2),0),(0.5,-sqrt(1-0.5**2),0)]
def get_pentagonal_angle():
    distance = float("inf")
    angle = 0
    delta = 0.1
    while delta > 0.0001:
        triangles = [get_equilateral_triangle()]
        triangles[0] = rotate(triangles[0], (angle+delta,0,0))
        triangles.append(rotate(triangles[-1],(0,0,2*pi/5)))
        new_distance = sum((triangles[0][1][i]-triangles[1][2][i])**2 for i in range(3))
        if new_distance < distance:
            distance = new_distance
            angle += delta
        else:
            delta /= 2
    return angle
def get_pentagonal_pyramid():
    points = [(0,0,0),(0,-1,0)]
    angle = get_pentagonal_angle()
    points = rotate(points, (angle,0,0))
    for i in range(4):
        points.extend(rotate(points[-1:],(0,0,2*pi/5)))
    edges = [(0,i) for i in range(1,6)]
    edges.extend([(1+i,1+(i+1)%5) for i in range(5)])
    polyhedron = Polyhedron()
    polyhedron.verts = points
    polyhedron.edges = [frozenset(x) for x in edges]
    polyhedron.construct_faces()
    return polyhedron
def get_icosahedron(displacement=(0,0,0), factors=(1,1,1), angles=(0,0,0)):
    pyramid = get_pentagonal_pyramid()
    points = pyramid.verts
    print(points)
    points.extend([(p[0],p[1],-p[2]+points[5][2]*2-sqrt(1-0.5**2)) for p in rotate(points,(0,0,2*pi/10))])
    center = (0,0,points[5][2]+sqrt(1-0.5**2)/2)
    points = shift(points, (-center[0], -center[1], -center[2]))
    points = scale(points, factors)
    points = rotate(points, angles)
    points = shift(points, center)
    points = shift(points, displacement)
    edges = [tuple(edge) for edge in pyramid.edges]
    edges.extend([(e[0]+6,e[1]+6) for e in edges])
    edges.extend([(1+i,1+(i-1)%5+6) for i in range(5)])
    edges.extend([(1+i,1+i+6) for i in range(5)])
    polyhedron = Polyhedron()
    polyhedron.verts = points
    polyhedron.edges = [frozenset(edge) for edge in edges]
    polyhedron.construct_faces()
    polyhedron.faces = [face for face in polyhedron.faces if len(face) == 3]
    return polyhedron

def get_dodecahedron(displacement=(0,0,0), factors=(1,1,1), angles=(0,0,0)):
    icosahedron = get_icosahedron()
    points = []
    edges = []
    for face in icosahedron.faces:
        triangle = list({icosahedron.verts[index] for edge_index in face for index in icosahedron.edges[edge_index]})
        points.append(tuple((triangle[0][i]+triangle[1][i]+triangle[2][i])/3 for i in range(3)))
    print(points)
    center = (0,0,icosahedron.verts[5][2]+sqrt(1-0.5**2)/2)
    points = shift(points, (-center[0], -center[1], -center[2]))
    points = scale(points, factors)
    points = rotate(points, angles)
    points = shift(points, center)
    points = shift(points, displacement)
    for i,x in enumerate(icosahedron.faces):
        for j,y in enumerate(icosahedron.faces):
            if i < j and len(x & y):
                edges.append((i,j))
    length = sqrt(sum((points[edges[0][0]][i]-points[edges[0][1]][i])**2 for i in range(3)))
    points = [(p[0]/length,p[1]/length,p[2]/length) for p in points]
    length = sum((points[edges[0][0]][i]-points[edges[0][1]][i])**2 for i in range(3))
    polyhedron = Polyhedron()
    polyhedron.verts = points
    polyhedron.edges = [frozenset(edge) for edge in edges]
    polyhedron.construct_faces()
    polyhedron.faces = [face for face in polyhedron.faces if len(face) == 5]
    return polyhedron
    
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
        #print(force, location, acc, self.racc, Ix, Iy, Iz)
    def simulate(self, dt):
        acc = self.acc
        vel = self.vel
        pos = self.pos
        self.vel = (vel[0]+acc[0]*dt, vel[1]+acc[1]*dt, vel[2]+acc[2]*dt)
        self.pos = (pos[0]+vel[0]*dt, pos[1]+vel[1]*dt, pos[2]+vel[2]*dt)
        self.polyhedron.verts = [(p[0]+vel[0]*dt,p[1]+vel[1]*dt,p[2]+vel[2]*dt) for p in self.polyhedron.verts]
        racc = self.racc
        rvel = self.rvel
        rpos = self.rpos
        self.rvel = (rvel[0]+racc[0]*dt, rvel[1]+racc[1]*dt, rvel[2]+racc[2]*dt)
        self.rpos = (rpos[0]+rvel[0]*dt, rpos[1]+rvel[1]*dt, rpos[2]+rvel[2]*dt)
        points = self.polyhedron.verts
        points = shift(points, (-self.pos[0], -self.pos[1], -self.pos[2]))
        points = rotate(points, (rvel[0]*dt, rvel[1]*dt, rvel[2]*dt))
        points = shift(points, self.pos)
        self.polyhedron.verts = points
    def zero_forces(self):
        self.acc = (0,0,0)
        self.racc = (0,0,0)
    def gravity(self, dt):
        self.apply((0,-9.8*dt*self.mass,0), self.pos)
    def ground(self, dt):
        epsilon = .001
        points = sorted([(p[1], p) for p in self.polyhedron.verts])
        count = len([p for p in points if p[0]<epsilon])
        #print("count: ", count)
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
            points = self.polyhedron.verts
            points = shift(points, (0, delta, 0))
            self.polyhedron.verts = points




class Crosshair():
    def __init__(self):
        self.pos = (0,0)
    def draw(self, pygame, screen):
        screen_width, screen_height = screen.get_size()
        p = [(self.pos[0]+screen_width/2+x, -self.pos[1]+screen_height/2) for x in (-10,10)]
        pygame.draw.line(screen, "white", p[0], p[1])
        p = [(self.pos[0]+screen_width/2, -self.pos[1]+screen_height/2+x) for x in (-10,10)]
        pygame.draw.line(screen, "white", p[0], p[1])

def load_texture(file_path):
    image = Image.open(file_path)
    bitmap = [[] for i in range(image.size[1])]
    for j in range(image.size[1]):
        for i in range(image.size[0]):
            pixel = image.getpixel((i,j))
            bitmap[j].append(int(sum(pixel)/3 < 128))
    return bitmap

def process_textures(input_queue, output_queue, screen_size):
    while True:
        textures = input_queue.get()
        output = []
        for texture in textures:
            output.extend(texture.draw_offline(screen_size))
        output_queue.put(output)
class Texture():
    def __init__(self, bitmap, polyhedron, face_index,  center=None):
        self.bitmap = bitmap
        self.polyhedron = polyhedron
        self.center = center
        points = polyhedron.circuit(face_index)
        if center is None:
            self.center = tuple(sum([point[i] for point in points])/len(points) for i in range(3))
        self.x_vector = (1,0,0)
        self.x_vector = tuple((points[1][i]-points[0][i])/len(self.bitmap[0]) for i in range(3))
        self.y_vector = (0,1,0)
        self.y_vector = tuple((points[3][i]-points[0][i])/len(self.bitmap) for i in range(3))
    def draw(self, pygame, screen):
        screen_width, screen_height = screen.get_size()
        bitmap_center = (len(self.bitmap)/2+0.5, len(self.bitmap[0])/2+0.5)
        polygon_center = self.center
        output = []
        for i,x in enumerate(self.bitmap):
            for j,y in enumerate(x):
                if y == 1:
                    pass
                    pos = (-(i+0.5-bitmap_center[0]),j+0.5-bitmap_center[1])
                    pos = tuple(self.x_vector[k]*pos[1]+self.y_vector[k]*pos[0] for k in range(3))
                    pos = tuple(pos[k]+polygon_center[k] for k in range(3))
                    pos = camera.project(pos)
                    pos = (pos[0]+screen_width/2,-pos[1]+screen_height/2)
                    output.append(pos)
                    #pygame.draw.circle(screen, "white", pos, 1)
        return output
    def draw_offline(self, screen_size):
        screen_width, screen_height = screen_size
        bitmap_center = (len(self.bitmap)/2+0.5, len(self.bitmap[0])/2+0.5)
        polygon_center = self.center
        output = []
        for i,x in enumerate(self.bitmap):
            for j,y in enumerate(x):
                if y == 1:
                    pass
                    pos = (-(i+0.5-bitmap_center[0]),j+0.5-bitmap_center[1])
                    pos = tuple(self.x_vector[k]*pos[1]+self.y_vector[k]*pos[0] for k in range(3))
                    pos = tuple(pos[k]+polygon_center[k] for k in range(3))
                    pos = camera.project(pos)
                    pos = (pos[0]+screen_width/2,-pos[1]+screen_height/2)
                    output.append(pos)
                    #pygame.draw.circle(screen, "white", pos, 1)
        return output
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


def apply_function(input_queue, output_queue):
    while True:
        f, args = input_queue.get()
        output_queue.put(f(*args))


camera = Camera()
camera.focal = (0,0,-100)
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

#cube = get_cube(displacement=(0,5,1), factors=2, angles=(2*pi*random.random(),2*pi*random.random(),2*pi*random.random()))
#cube = get_cube(displacement=(5,5,1), factors=2, angles=(0,0,2*pi*random.random()))
#center = (sum(p[0] for p in cube.verts)/len(cube.verts),sum(p[1] for p in cube.verts)/len(cube.verts),sum(p[2] for p in cube.verts)/len(cube.verts))
#body = Body(1, center, polyhedron=cube)
#body.apply((1000,0,0), list(cube.verts)[0])
#octahedron = get_octahedron(displacement=(0,5,1), factors=2, angles=(2*pi*random.random(),2*pi*random.random(),2*pi*random.random()))
#center = (sum(p[0] for p in octahedron.verts)/len(octahedron.verts),sum(p[1] for p in octahedron.verts)/len(octahedron.verts),sum(p[2] for p in octahedron.verts)/len(octahedron.verts))
#body = Body(1, center, polyhedron=octahedron)
#tetrahedron = get_tetrahedron(displacement=(0,5,1), factors=2, angles=(2*pi*random.random(),2*pi*random.random(),2*pi*random.random()))
#center = (sum(p[0] for p in tetrahedron.verts)/len(tetrahedron.verts),sum(p[1] for p in tetrahedron.verts)/len(tetrahedron.verts),sum(p[2] for p in tetrahedron.verts)/len(tetrahedron.verts))
#body = Body(1, center, polyhedron=tetrahedron)
#icosahedron = get_icosahedron(displacement=(0,5,1), factors=2, angles=(2*pi*random.random(),2*pi*random.random(),2*pi*random.random()))
#center = (sum(p[0] for p in icosahedron.verts)/len(icosahedron.verts),sum(p[1] for p in icosahedron.verts)/len(icosahedron.verts),sum(p[2] for p in icosahedron.verts)/len(icosahedron.verts))
#body = Body(1, center, polyhedron=icosahedron)
dodecahedron = get_dodecahedron(displacement=(0,5,1), factors=2, angles=(2*pi*random.random(),2*pi*random.random(),2*pi*random.random()))
center = (sum(p[0] for p in dodecahedron.verts)/len(dodecahedron.verts),sum(p[1] for p in dodecahedron.verts)/len(dodecahedron.verts),sum(p[2] for p in dodecahedron.verts)/len(dodecahedron.verts))
body = Body(1, center, polyhedron=dodecahedron)
body.rvel = (0,0,1)
#body.vel = (0,0,100)
grid = Grid()
camera_angular = [0,0,0]
camera_velocity = [0,0,0]
force = 0
space_down = False
crosshair = Crosshair()
bitmap6 = load_texture('six.png')
bitmap1 = load_texture('one.png')
if __name__ == "__main__":
    hit_input_queue = mp.Queue()
    hit_output_queue = mp.Queue()
    hit_process = mp.Process(target=apply_function, args=(hit_input_queue,hit_output_queue))
    hit_process.start()
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    screen_width, screen_height = screen.get_size()
    #texture_input_queue = mp.Queue()
    #texture_output_queue = mp.Queue()
    #texture_process = mp.Process(target=process_textures, args=(texture_input_queue, texture_output_queue, screen.get_size()))
    #texture_process.start()
    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)
    pygame.mouse.set_pos((screen_width/2, screen_height/2))
    running = True
    framerate = 60
    while running:
        body.zero_forces()
        #textures = []
        #textures.append(Texture(bitmap6, body.polyhedron, 0))
        #textures.append(Texture(bitmap1, body.polyhedron, 4))
        #texture_input_queue.put(textures)
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
                    hit_input_queue.put((body.polyhedron.ray, (pos,vector)))
                    '''
                    hits = body.polyhedron.ray(pos,vector)
                    if len(hits):
                        print(hits)
                        force_vector = tuple(vector[i]*force for i in range(3))
                        body.apply(force_vector, hits[0][1])
                    force = 0
                    '''
            if event.type == pygame.MOUSEMOTION:
                x,y = pygame.mouse.get_pos()
                x -= screen_width/2
                y = -(y-screen_height/2)
                crosshair.pos = (x,y)
            if event.type == pygame.MOUSEBUTTONDOWN:
                pass
        try:
            hits = hit_output_queue.get(False)
            if len(hits):
                print('hits',hits)
                force_vector = tuple(vector[i]*force for i in range(3))
                body.apply(force_vector, hits[0][1])
            force = 0
        except queue.Empty:
            pass


        #print("rel", pygame.mouse.get_rel(), "pos", pygame.mouse.get_pos())
        #pygame.draw.circle(screen, "red", (0,0), 40)
        screen.fill("black")
        crosshair.draw(pygame, screen)
        for edge in body.polyhedron.edges:
            p1, p2 = tuple(body.polyhedron.verts[index] for index in edge)
            #print(p1,p2)
            p1, p2 = camera.project(p1), camera.project(p2)
            p1 = (p1[0]*1+screen_width/2,p1[1]*-1+screen_height/2)
            p2 = (p2[0]*1+screen_width/2,p2[1]*-1+screen_height/2)
            pygame.draw.line(screen, "white", p1, p2)
        '''
        try:
            dots = []
            while True:
                dots = texture_output_queue.get(False)
        except queue.Empty:
            if len(dots):
                print('dots',dots)
            for pos in dots:
                pygame.draw.circle(screen, "white", pos, 1)
        '''
        #grid.draw(pygame,screen)
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
        #print("dT", dT, force)
    hit_process.kill()
    #texture_process.kill()
    pygame.quit()
