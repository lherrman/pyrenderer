# -*- coding: utf-8 -*-
"""
@author: herrm
"""
import numpy as np
import cv2
import time
from IPython import get_ipython

class Renderer(object):
    def __init__(self, height, width):
        self.height, self.width = height, width
        self.__hhalf, self.__whalf = int(height/2), (width/2)
        self.fov = (np.pi/180) * 60
        cv2.namedWindow("render3d_{}".format(id(self)))
        ratio = width / height

        self.P = np.array([[1/np.tan(self.fov/2), 0, 0],
                           [0, ratio/np.tan(self.fov/2), 0],
                           [0, 0, 1]])

        self.image = np.zeros([self.height, self.width, 3], dtype=np.uint8)

        self.__input_setup()

    def render(self, object3d, mode="solid"):
        self.image *= 0

        obj_vertecies = np.array(object3d.world_coord)

        # project vertices to pixelspace
        stretch_factor = np.array([self.__whalf, self.__hhalf, 1]).reshape(1, 3)
        offset_factor = np.array([self.__hhalf, self.__whalf, 0]).reshape(1, 3)

        projected_vertices = (self.P @ obj_vertecies.T).T
        projected_vertices[:, :2] /= -projected_vertices[:, 2].reshape(-1, 1)
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

            # calcualte indices that sorts faces from farthest away to nearest
            depth = np.zeros(len(object3d.faces))
            for idx, f in enumerate(object3d.faces):
                depth[idx] = projected_vertices[f[0]-1][2] + projected_vertices[f[1]-1][2] + projected_vertices[f[2]-1][2]
            arg_depth_sort = np.argsort(depth)[::-1]

            # draw faces
            face = np.zeros((3, 1, 2), dtype=np.int32)
            for idx, f in enumerate(object3d.faces):
                face[0, 0, ::-1] = obj_vertecies_int[object3d.faces[arg_depth_sort[idx]][0]-1]
                face[1, 0, ::-1] = obj_vertecies_int[object3d.faces[arg_depth_sort][idx][1]-1]
                face[2, 0, ::-1] = obj_vertecies_int[object3d.faces[arg_depth_sort][idx][2]-1]
                col = ((idx//6) % 256)
                self.image = cv2.fillPoly(self.image, [face], (col, col, col), 1)

        cv2.imshow("render3d_{}".format(id(self)), self.image)

    def render_test_orbit_control(self):
        b1 = P3dObject(np.array([0, 0, 0]), [0.0, 0.0, 3.2 * np.pi/2], 0.03)

        t0 = time.time()
        dist = 20.
        distgoal = 7.

        while 1:
            t1 = time.time()

            dist = dist + 0.1 * (distgoal - dist) + 0.01 * np.sign(distgoal - dist)*(distgoal - dist)**2
            rotate_mouse = self.mouse_xy_diff / 1000
            self.mouse_xy_diff *= 0.8

            b1.set_pos(np.array([0, 0, dist]))
            b1.rotate([rotate_mouse[0], np.cos(b1.orientation[0]) * rotate_mouse[1], np.sin(b1.orientation[0]) * rotate_mouse[1]])

            self.render(b1)

            # Keyboard input
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord("w"):
                distgoal += 2
                self.render(b1)
            elif key == ord("s"):
                distgoal -= 2

            # (clear console and) print FPS every second
            t2 = time.time()
            if t2 - t0 > 1:
                t0 = t2
                # print("\033[H\033[J")
                print("{:.3} ms {:2.3} FPS".format((t2-t1) * 1000, 1/(t2-t1)))

    def __mouse_update(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            last = self.mouse_xy.copy()

            self.mouse_xy[:] = [x, y]
            if self.mouse_btns[0]:
                self.mouse_xy_diff += (self.mouse_xy - last)
                self.mouse_xy_diff *= 1

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_btns[0] = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_btns[0] = False

        elif event == cv2.EVENT_MOUSEHWHEEL:
            self.mouse_wheel += 1
            print(self.mouse_wheel)

    def __input_setup(self):
        cv2.setMouseCallback("render3d_{}".format(id(self)), self.__mouse_update)
        self.mouse_xy = np.array([0., 0])
        self.mouse_xy_diff = np.array([0., 0])
        self.mouse_btns = np.array([False, False, False])
        self.mouse_wheel = 0.0
        # print([i for i in dir(cv2) if 'EVENT' in i])


class P3dObject():
    def __init__(self, pos, orientation=[0.0, 0.0, 0.0], scale=1, path="obj/teapot.obj"):
        self.pos = pos
        self.orientation = orientation

        f = open(path, 'r')
        obj_data = f.readlines()
        f.close()

        vertices = []
        faces = []

        for line in obj_data:
            if "v " in line:
                numbs = line.lstrip("v ").rstrip("\n").split(" ")
                vertices.append([float(numbs[0]), float(numbs[1]), float(numbs[2])])

            if "f " in line:
                numbs = line.lstrip("f ").rstrip("\n").split(" ")
                faces.append([int(numbs[0]), int(numbs[1]), int(numbs[2])])

        self.vertices = np.array(vertices)
        self.vertices = self.vertices * scale
        self.faces = np.array(faces)
        self.world_coord = self.vertices + self.pos

    def rotate(self, rotXYZ):
        for n in range(3):
            self.orientation[n] += rotXYZ[n]

        rotX = np.array([[1, 0, 0],
                         [0, np.cos(self.orientation[0]), np.sin(self.orientation[0])],
                         [0, -np.sin(self.orientation[0]), np.cos(self.orientation[0])]])

        rotY = np.array([[np.cos(self.orientation[1]), 0, np.sin(self.orientation[1])],
                         [0, 1, 0],
                         [-np.sin(self.orientation[1]), 0, np.cos(self.orientation[1])]])

        rotZ = np.array([[np.cos(self.orientation[2]), -np.sin(self.orientation[2]), 0],
                         [np.sin(self.orientation[2]), np.cos(self.orientation[2]), 0],
                         [0, 0, 1]])

        R = rotX @ rotY @ rotZ

        self.world_coord = ((R @ self.vertices.T).T) + self.pos

    def set_pos(self, pos):
        self.pos = pos


if __name__ == "__main__":

    p3ds = Renderer(600, 1000)
    p3ds.render_test_orbit_control()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
