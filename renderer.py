# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:40:09 2019

@author: herrm
"""
import numpy as np
import cv2
import sys


class Proj3d(object):
    def __init__(self, height, width):
        self.height,self.width = height, width
        self.hhalf, self.whalf = int(height/2),int(width/2)
           
        self.fov = (np.pi/180) * 60
        self.ratio = width/height
        self.nearClip = 1
        self.farClip = 10
        
        
    def project(self, object3d):
        self.image = np.zeros([self.height,self.width,1],dtype = np.uint8)
        self.P = np.array([[1/np.tan(self.fov/2),0,0,0],
                            [0, self.ratio/np.tan(self.fov/2),0,0],
                            [0,0, (self.nearClip+self.farClip)/(self.nearClip-self.farClip),(2* self.nearClip * self.farClip)/(self.nearClip-self.farClip)],
                            [0,0,-1,0]])
        self.projectet_vertex = []
        for v in object3d.world_coord:
            self.p = (np.matmul(self.P, v) / (-v[2]))
            self.p[0] *= self.whalf
            self.p[1] *= self.hhalf
            self.projectet_vertex.append(( int(self.p[1]+self.whalf), int(self.p[0]+self.hhalf)))
            x = int(self.p[0]+self.hhalf)
            y = int(self.p[1]+self.whalf)

            #cv2.circle(self.image,(y,x),3,255,thickness = 1)
            
        for f in object3d.faces:
            cv2.line(self.image, self.projectet_vertex[f[0]-1], self.projectet_vertex[f[1]-1], 255,  1)
            cv2.line(self.image, self.projectet_vertex[f[1]-1], self.projectet_vertex[f[2]-1], 255,  1)
            cv2.line(self.image, self.projectet_vertex[f[2]-1], self.projectet_vertex[f[0]-1], 255,  1)
               
        cv2.imshow("render3d{}".format(self),self.image)    
        
        
    def testCube(self):
        b1 = P3dObject(np.array([0,0,0]))
        a = 0.0
        while 1:
            b1.set_pos(np.array([0, 0, 10 ]))
            b1.rotate([0.03, 0.0, 0.0])
            self.project(b1)
            if cv2.waitKey(25) == 27:
                break
            a += 0.04 
 
        
class P3dObject():
    def __init__(self,pos , orientation = [0.0, 0.0, 3.2*  np.pi/2] ):
        self.pos = pos
        self.orientation = orientation
        path = "obj/teapot.obj"
        f = open(path, 'r')
        obj_data = f.readlines()
        f.close()  
        
        vertex = []
        faces = []
        for line in obj_data:            
            if "v " in line:
                numbs = line.lstrip("v ").rstrip("\n").split(" ")
                vertex.append([float(numbs[0]), float(numbs[1]), float(numbs[2])])
                
            if "f " in line:
                numbs = line.lstrip("f ").rstrip("\n").split(" ")
                faces.append([int(numbs[0]), int(numbs[1]), int(numbs[2])])
                
        self.vertex = np.array(vertex)
        self.vertex = self.vertex * 0.03
        self.faces = np.array(faces) 
        self.world_coord =  [np.array( [v[0] + self.pos[0] , v[1] + self.pos[1] , v[2] + self.pos[2], 1]) for v in self.vertex]
        
        
    def rotate(self, rotXYZ):
        for n in range(3):
            self.orientation[n] += rotXYZ[n]
        
        self.rotX = np.array([[1 , 0 , 0],
                           [0,np.cos(self.orientation[0]), np.sin(self.orientation[0])],
                           [0 , -np.sin(self.orientation[0]) , np.cos(self.orientation[0])]])     

        self.rotY = np.array([[np.cos(self.orientation[1]) , 0 , np.sin(self.orientation[1])],
                           [0, 1, 0],
                           [ -np.sin(self.orientation[1]) , 0 , np.cos(self.orientation[1])]])
        
        self.rotZ = np.array([[np.cos(self.orientation[2]) , -np.sin(self.orientation[2]) , 0],
                           [np.sin(self.orientation[2]) ,np.cos(self.orientation[2]), 0],
                           [0 , 0 , 1]])                                            
                                                             
        self.r = []    
        for v in self.vertex:
            self.r.append(self.rotX @ self.rotY @ self.rotZ @ v)
            
            
        self.world_coord =  [np.array( [x[0] + self.pos[0] , x[1] + self.pos[1] , x[2] + self.pos[2], 1]) for x in self.r]        
        
    def set_pos(self,pos):
        self.pos = pos
        
        

        
        
if __name__ == "__main__":
    
    p3ds = Proj3d(800,1400)
    
    p3ds.testCube()
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()      