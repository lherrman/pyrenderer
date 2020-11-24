# -*- coding: utf-8 -*-
"""
@author: herrm
"""
import numpy as np
import cv2
import time


class Renderer(object):
    def __init__(self, height, width):
        self.height, self.width = height, width
        self.__hhalf, self.__whalf = height // 2, width // 2
        self.__stretch_factor = np.array([self.__whalf, self.__hhalf, 1]).reshape(1, 3)
        self.__offset_factor = np.array([self.__hhalf, self.__whalf, 0]).reshape(1, 3)
        self.__fov = (np.pi/180) * 60
        ratio = width / height

        self.__P = np.array([[1 / np.tan(self.__fov / 2), 0, 0],
                             [0, ratio / np.tan(self.__fov / 2), 0],
                             [0, 0, 1]])

        self.image = np.zeros([self.height, self.width, 3], dtype=np.uint8)

        cv2.namedWindow("render3d_{}".format(id(self)))
        self.__input_setup()

    def render(self, object3d, mode="shaded"):
        self.image *= 0

        object_vertices_world = object3d.world_coord

        # project vertices to pixelspace
        projected_vertices = (self.__P @ object_vertices_world.T).T
        projected_vertices[:, :2] /= -projected_vertices[:, 2].reshape(-1, 1)
        projected_vertices *= self.__stretch_factor
        projected_vertices += self.__offset_factor

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

        elif mode == "shaded1":
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

                col = (object3d.world_normal[arg_depth_sort[idx]][1] * 200 + 20, 0, - object3d.world_normal[arg_depth_sort[idx]][1] * 200 + 20)
                self.image = cv2.fillPoly(self.image, [face], col, 1)

        elif mode == "shaded2":
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

                col = (-object3d.world_normal[arg_depth_sort[idx]][1] * 200 + 20,
                       0,
                       -object3d.world_normal[arg_depth_sort[idx]][0] * 100)

                self.image = cv2.fillPoly(self.image, [face], col, 1)

        cv2.imshow("render3d_{}".format(id(self)), self.image)

    def render_test_orbit_control(self):
        b1 = P3dObject([0, 0, 0], [0.0, 0.0, 3.2 * np.pi/2], scale=0.03)

        t0 = time.time()
        mode_n, modes = 2, ["point", "wireframe", "solid", "shaded1", "shaded2"]
        dist = 14.
        distgoal = 7.
        while 1:
            t1 = time.time()

            dist_diff = distgoal - dist
            dist = dist + 0.1 * dist_diff + 0.01 * np.sign(dist_diff)*(dist_diff)**2
            rotate_mouse = self.__mouse_xy_diff / 1000
            if self.__mouse_btns[0]:
                self.__mouse_xy_diff *= 0.6
            else:
                self.__mouse_xy_diff *= 0.99

            b1.set_pos(np.array([0, 0, dist]))
            b1.rotate([rotate_mouse[0],
                       np.cos(b1.orientation[0]) * rotate_mouse[1],
                       np.sin(b1.orientation[0]) * rotate_mouse[1]])

            self.render(b1, modes[mode_n])

            # Keyboard input
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord("w"):
                distgoal += 2
                self.render(b1)
            elif key == ord("s"):
                distgoal -= 2
            elif key == ord("d"):
                mode_n += 1
                mode_n = mode_n % len(modes)
            elif key == ord("a"):
                mode_n -= 1
                mode_n = mode_n % len(modes)

            # print FPS every second
            t2 = time.time()
            if t2 - t0 > 1:
                t0 = t2
                print("{:.3} ms {:2.4} FPS".format((t2-t1) * 1000, 1/(t2-t1)))

    def __mouse_update(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            last = self.__mouse_xy.copy()
            self.__mouse_xy[:] = [x, y]
            if self.__mouse_btns[0]:
                self.__mouse_xy_diff += (self.__mouse_xy - last)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.__mouse_btns[0] = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.__mouse_btns[0] = False

    def __input_setup(self):
        cv2.setMouseCallback("render3d_{}".format(id(self)), self.__mouse_update)
        self.__mouse_xy = np.array([0., 0])
        self.__mouse_xy_diff = np.array([0., 0])
        self.__mouse_btns = np.array([False, False, False])
        # print([i for i in dir(cv2) if 'EVENT' in i])


class P3dObject():
    def __init__(self, pos, orientation=[0.0, 0.0, 0.0], scale=1, path="obj/teapot.obj"):
        self.pos = np.array(pos)
        self.orientation = np.array(orientation)

        # read .obj file and extract vertices and faces
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
        self.vertices *= scale
        self.faces = np.array(faces)
        self.world_coord = self.vertices + self.pos
        self.__calculate_normals()

    def __calculate_normals(self):
        self.normal = np.zeros((len(self.faces), 3))
        for idx, f in enumerate(self.faces):
            vec1 = self.world_coord[f[0]-1] - self.world_coord[f[1]-1]
            vec2 = self.world_coord[f[0]-1] - self.world_coord[f[2]-1]
            normal = np.cross(vec1, vec2)
            self.normal[idx, :] = normal / np.linalg.norm(normal)



    def rotate(self, rotXYZ):
        self.orientation += rotXYZ

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

        self.world_normal = ((R @ self.normal.T).T)
        self.world_coord = ((R @ self.vertices.T).T) + self.pos

    def set_pos(self, pos):
        self.pos = pos


if __name__ == "__main__":

    p3ds = Renderer(600, 1000)
    p3ds.render_test_orbit_control()

    cv2.destroyAllWindows()
