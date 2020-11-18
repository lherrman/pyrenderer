# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:40:09 2019

@author: herrm
"""
import numpy as np
import cv2
import sys
import time


class Proj3d(object):
    def __init__(self, height, width):
        self.height, self.width = height, width
        self.__hhalf, self.__whalf = int(height/2), (width/2)
        self.fov = (np.pi/180) * 60

        ratio = width / height

        self.P = np.array([[1/np.tan(self.fov/2), 0, 0, 0],
                           [0, ratio/np.tan(self.fov/2), 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 1, 0]])

        self.image = np.zeros([self.height, self.width, 1], dtype=np.uint8)

    def render(self, object3d, mode="solid"):
        self.image *= 0

        obj_vertecies = np.array(object3d.world_coord)

        # project vertices to pixelspace
        stretch_factor = np.array([self.__whalf, self.__hhalf, 1, 1]).reshape(1, 4)
        offset_factor = np.array([self.__hhalf, self.__whalf, 0, 0]).reshape(1, 4)

        projected_vertices = (self.P @ obj_vertecies.T).T
        projected_vertices[:, :3] /= projected_vertices[:, 2].reshape(-1, 1)
        projected_vertices *= stretch_factor
        projected_vertices += offset_factor

        if mode == "point":
            obj_vertecies_int = np.ceil(projected_vertices).astype(np.int32)
            for v in obj_vertecies_int:
                cv2.circle(self.image, (v[1], v[0]), 3, 255, thickness=-1)

        elif mode == "wireframe":
            obj_vertecies_int = np.ceil(projected_vertices[:, 0:2]).astype(np.int32)
            for f in object3d.faces:
                cv2.line(self.image, (obj_vertecies_int[f[0]-1][1], obj_vertecies_int[f[0]-1][0]),
                                     (obj_vertecies_int[f[1]-1][1], obj_vertecies_int[f[1]-1][0]),
                         255, 1)
                cv2.line(self.image, (obj_vertecies_int[f[1]-1][1], obj_vertecies_int[f[1]-1][0]),
                                     (obj_vertecies_int[f[2]-1][1], obj_vertecies_int[f[2]-1][0]),
                         255, 1)
                cv2.line(self.image, (obj_vertecies_int[f[2]-1][1], obj_vertecies_int[f[2]-1][0]),
                                     (obj_vertecies_int[f[0]-1][1], obj_vertecies_int[f[0]-1][0]),
                         255, 1)
        elif mode == "solid":
            obj_vertecies_int = np.ceil(projected_vertices[:, 0:2]).astype(np.uint32)

            # sorting faces from farthest away to nearest
            depth = np.zeros(len(object3d.faces))
            for idx, f in enumerate(object3d.faces):
                depth[idx] = projected_vertices[f[0]-1][3] + projected_vertices[f[1]-1][3] + projected_vertices[f[2]-1][3]
            arg_depth_sort = np.argsort(depth)[::-1]

            # draw faces
            face = np.zeros((3, 1, 2), dtype=np.int32)
            for idx, f in enumerate(object3d.faces):
                face[0, 0, ::-1] = obj_vertecies_int[object3d.faces[arg_depth_sort[idx]][0]-1]
                face[1, 0, ::-1] = obj_vertecies_int[object3d.faces[arg_depth_sort][idx][1]-1]
                face[2, 0, ::-1] = obj_vertecies_int[object3d.faces[arg_depth_sort][idx][2]-1]
                self.image = cv2.fillPoly(self.image, [face], idx//6, 8)

        cv2.imshow("render3d{}".format(self), self.image)

    def render_test_object(self):
        b1 = P3dObject(np.array([0, 0, 0]), [0.0, 0.0, 3.2 * np.pi/2], 0.03)
        a = 0.0
        while 1:
            t1 = time.time()
            b1.set_pos(np.array([0, 0, 10]))
            b1.rotate([0.03, 0.0, 0.0])
            self.render(b1)
            if cv2.waitKey(1) == 27:
                break
            a += 0.04
            t2 = time.time()
            print(t2-t1, "ms  ", 1/(t2-t1), " FPS")


class P3dObject():
    def __init__(self, pos, orientation=[0.0, 0.0, 0.0], scale=0.03):
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
        self.vertex = self.vertex * scale
        self.faces = np.array(faces)
        self.world_coord =  [np.array([v[0] + self.pos[0], v[1] + self.pos[1], v[2] + self.pos[2], 1]) for v in self.vertex]

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

    p3ds.render_test_object()

    cv2.waitKey(0)
    cv2.destroyAllWindows()