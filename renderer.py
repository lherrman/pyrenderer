# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:40:09 2019

@author: herrm
"""
import numpy as np
import cv2


        

class Proj3d(object):
    def __init__(self, height, width):
        self.height,self.width = height, width
        self.hhalf, self.whalf = int(height/2),int(width/2)
           
        self.fov = (np.pi/180) * 60
        self.ratio = width/height
        self.nearClip = 1
        self.farClip = 100
        
        
    def project(self, vertex):
        self.image = np.zeros([self.height,self.width,1],dtype = np.uint8)
        self.P = np.array([[1/np.tan(self.fov/2),0,0,0],
                            [0, self.ratio/np.tan(self.fov/2),0,0],
                            [0,0, (self.nearClip+self.farClip)/(self.nearClip-self.farClip),(2* self.nearClip * self.farClip)/(self.nearClip-self.farClip)],
                            [0,0,-1,0]])
        self.pix = []
        for v in vertex:
            self.p = (np.matmul(self.P, v) / (-v[2]))
            self.p[0] *= self.whalf
            self.p[1] *= self.hhalf
            self.pix.append(self.p)
            x = int(self.p[0]+self.hhalf)
            y = int(self.p[1]+self.whalf)
            try:
                #self.image[int(self.p[0])+self.hhalf:int(self.p[0])+5+self.hhalf,int(self.p[1])+self.whalf:int(self.p[1]) +5 +self.whalf] = 255
                cv2.circle(self.image,(y,x),3,255,thickness = -5)
            except:
                pass
            
        cv2.imshow("proj3d{}".format(self),self.image)    
        
        
        
        
    def testCube(self):
        b1 = P3dObject(np.array([0,0,0]))
        a = 0.0
        t = True
        while 1:
            b1.set_pos(np.array([0, 0, 10 ]))
            b1.rotate(0.03)
            self.project(b1.world_coord)
            t = False
            if cv2.waitKey(25) == 27:
                break
            a += 0.04 
 
        
class P3dObject():
    def __init__(self,pos , angle = 0.0):
        self.pos = pos
        self.angle = angle
        path = "obj/teapot.obj"
        f = open(path, 'r')
        obj_data = f.readlines()
        f.close()        
        vertex = []
        for line in obj_data:            
            if "v  " in line:
                print(line)
                numbs = line.lstrip("v ").rstrip("\n").split("  ")
                
                try:
                    vertex.append([float(numbs[0]), float(numbs[1]), float(numbs[2])])
                except:
                    pass
        
        self.vertex = np.array(vertex)
        
        self.vertex = self.vertex * 0.02
        
        
        self.world_coord =  [np.array( [v[0] + self.pos[0] , v[1] + self.pos[1] , v[2] + self.pos[2], 1]) for v in self.vertex]
        
        
    def rotate(self,a):
        self.angle += a
        
        self.rotX = np.array([[1 , 0 , 0],
                           [0,np.cos(self.angle), np.sin(self.angle)],
                           [0 , -np.sin(self.angle) , np.cos(self.angle)]])    
        self.r = []    
        for v in self.vertex:

            self.r.append(np.matmul(self.rotX , v))
            
        self.world_coord =  [np.array( [x[0] + self.pos[0] , x[1] + self.pos[1] , x[2] + self.pos[2], 1]) for x in self.r]        
        
    def set_pos(self,pos):
        self.pos = pos
        
        

        
        
if __name__ == "__main__":
    
    p3ds = Proj3d(800,1400)
    
    p3ds.testCube()
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()      