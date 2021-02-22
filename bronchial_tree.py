#!/usr/bin/env python
""" UNSW, COMP9991/COMP9993, 20T2/T3
    Oleg Kufirin - z5216713
    Bronchial Tree class implementation"""

import fnmatch
import os
import time
import random

from datetime import datetime
from string import Template
from scipy.io import loadmat
from scipy import interpolate
from sortedcontainers import SortedList
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import cv2

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------

DICOM_DIRECTORY = r"F:\BronchialDetectionProject\DICOMDATA\\"\
                                "crosssectionalimageData\\"
DICOM_FILE_NAME_PREFIX = Template("Bronchial_cross_sectional_"+
                                  "image_data_Study_$st")

# ---------------------------

class CrossSection():
    """Cross section class"""
    def __init__(self, image_number, p_t):
        self.image_number = image_number
        self.p_t = p_t
        self.detections = []
        self.annotations = []
        self.prediction_centrepoint = []

    def __eq__(self, other):
        return self.p_t == other.p_t

    def __lt__(self, other):
        return self.p_t < other.p_t

# ---------------------------

class Edge():
    """Edge class"""
    def __init__(self, edge_number):
        self.edge_number = edge_number
        self.cross_sections = SortedList()

    def __eq__(self, other):
        return self.edge_number == other.edge_number

    def __lt__(self, other):
        return self.edge_number < other.edge_number

# check if the given cross section already exists in this edge
    def __cs_exist(self, p_t_check):
        if p_t_check in self.cross_sections:
            return True
        return False

    def add(self, image_number_insert, p_t_insert):
        """Add new cross section to the edge"""
        p_t_insert_obj = CrossSection(image_number_insert, p_t_insert)
        if not self.__cs_exist(p_t_insert_obj):
            self.cross_sections.add(p_t_insert_obj)
        else:
            index = self.cross_sections.index(p_t_insert_obj)
            cs = self.cross_sections[index]
            f = open("error_report.txt", "a+")
            error_msg = str(datetime.now().time()) + \
                    f" Image number {image_number_insert} " + \
                    f"is the same as {cs.image_number}. Skipping...\n"
            f.write(error_msg)
            print(error_msg)
            f.close()

# ---------------------------

class ConnectedComponent():
    """Connected Component class"""
    def __init__(self, cc_number):
        self.cc_number = cc_number
        self.edges = SortedList()

    def __eq__(self, other):
        return self.cc_number == other.cc_number

    def __lt__(self, other):
        return self.cc_number < other.cc_number

# check if the given edge already exists in this connected component
    def __edge_exist(self, egde_num_check):
        if egde_num_check in self.edges:
            return True
        return False

    def add(self, edge_insert, image_number_insert, p_t_insert):
        """Add new edge->cross section"""
        edge_insert_obj = Edge(edge_insert)
        if not self.__edge_exist(edge_insert_obj):
            self.edges.add(edge_insert_obj)
        index = self.edges.index(edge_insert_obj)
        edge = self.edges[index]
        edge.add(image_number_insert, p_t_insert)

# ---------------------------



class BronchialTree():
    """Bronchial Tree class"""
    def __init__(self, dicom_directory, dicom_file_name_prefix):
        self.study_number = None
        self.dicom_directory = dicom_directory
        self.dicom_file_name_prefix = Template(dicom_file_name_prefix)
        self.__connected_components = SortedList()


    # checking if given connected component exists in the tree
    def __cc_exist(self, cc_num_check):
        if cc_num_check in self.__connected_components:
            return True
        return False


    # insert new image into the tree
    def __add(self, cc_insert, edge_insert, image_number_insert,
              p_t_insert):
        cc_insert_obj = ConnectedComponent(cc_insert)
        if not self.__cc_exist(cc_insert_obj):
            self.__connected_components.add(cc_insert_obj)
        index = self.__connected_components.index(cc_insert_obj)
        cc = self.__connected_components[index]
        cc.add(edge_insert, image_number_insert, p_t_insert)


    # add detections for cross section
    def __add_detections(self, det, image_number, scale):

        # checking if detection already exists
        def __detection_in_list(detection, list_detections):
            return next((True for elem in list_detections
                         if np.array_equal(elem, detection)), False)

        for cc in self.__connected_components:
            for e in cc.edges:
                for cs in e.cross_sections:
                    if image_number == cs.image_number:
                        for d in det:
                            d = np.round(d/scale, 2)
                            if not __detection_in_list(d, cs.detections):
                                cs.detections.append(d)


    # add centrepoint prediction for cross section
    def __add_prediction_centrepoints(self, cpt, image_number):
        for cc in self.__connected_components:
            for e in cc.edges:
                for cs in e.cross_sections:
                    if image_number == cs.image_number:
                        cs.prediction_centrepoint.append(cpt)


    # add annotations for cross section
    def __add_annotations(self, ann, image_number):
        for cc in self.__connected_components:
            for e in cc.edges:
                for cs in e.cross_sections:
                    if image_number == cs.image_number:
                        cs.annotations.append(ann)
                        

    # delete current annotations
    def __clear_annotations(self):
        for cc in self.__connected_components:
            for e in cc.edges:
                for cs in e.cross_sections:
                    cs.annotations = []
                    
                    
    # delete current centrepoint predictions
    def __clear_prediction_centrepoints(self):
        for cc in self.__connected_components:
            for e in cc.edges:
                for cs in e.cross_sections:
                    cs.prediction_centrepoint = []   


    # extract connected component
    def __extract_cc(self, cc_in):
        cc_in_obj = ConnectedComponent(cc_in)
        if cc_in_obj not in self.__connected_components:
            raise ValueError(f"Component number {cc_in} does not " +
                             f"exist in study {self.study_number}")
        index = self.__connected_components.index(cc_in_obj)
        cc = self.__connected_components[index]
        return cc


    def __extract_edge(self, cc, edge_in):
        edge_in_obj = Edge(edge_in)
        if edge_in_obj not in cc.edges:
            raise ValueError(f"Edge number {edge_in} does not " +
                             f"exist in connected component {cc.cc_number} " +
                             f"in study {self.study_number}")
        index = cc.edges.index(edge_in_obj)
        edge = cc.edges[index]
        return edge


    # drawing vessel centreline in 3D
    def __draw_vessel_centreline(self, edge, plt3d):
        number_of_cs = len(edge.cross_sections)
        prev_point = ()
        for cs in range(number_of_cs):
            file_path = self.dicom_directory + \
                        self.dicom_file_name_prefix. \
                        substitute(st=self.study_number) + \
                        '_image_number_' + str(edge.cross_sections[cs].\
                                               image_number) + ".mat"
            matstruct_contents = loadmat(file_path)
            
            shape_x = matstruct_contents['image'][0, 0]['xi_4'].shape[1]
            shape_y = matstruct_contents['image'][0, 0]['xi_4'].shape[0]
            
            centre_point_2d_x = shape_x // 2
            centre_point_2d_y = shape_y // 2

            centre_point_3d_x = matstruct_contents['image'][0, 0]['xi_4']\
                [centre_point_2d_x][centre_point_2d_y]
            centre_point_3d_y = matstruct_contents['image'][0, 0]['yi_4']\
                [centre_point_2d_x][centre_point_2d_y]
            centre_point_3d_z = matstruct_contents['image'][0, 0]['zi_4']\
                [centre_point_2d_x][centre_point_2d_y]

            # plot points - first and last are red
            if cs == 0 or cs == number_of_cs-1:
                point_colour = "red"
            else:
                point_colour = "black"

            plt3d.plot([centre_point_3d_x], [centre_point_3d_y], 
                       [centre_point_3d_z], color=point_colour, marker='o',
                       markersize=3, alpha=1)

            # plot lines connecting points
            if not cs == 0:
                plt3d.plot([centre_point_3d_x, prev_point[0]],
                           [centre_point_3d_y, prev_point[1]],
                           [centre_point_3d_z, prev_point[2]], 
                           color="blue", linewidth=1)
            # storing coordinates of the previous point
            prev_point = (centre_point_3d_x, centre_point_3d_y, 
                          centre_point_3d_z)
    

    # drawing gt centreline in 3D
    def __draw_annotations(self, cc_in, edge_in, plt3d, 
                  source="annotations.txt", drawing_scope="edge"):
        # reading the file with annotations
        f = open(source, "r")
        while True:
            # reading next line from the file
            line = f.readline()

            # exit if reached the end
            if not line:
                break

            # skip if line starts with #
            if line[0] == '#':
                continue

            # split study no., cc no., edge no. and bounding boxes
            whole_line = line.rsplit(";")
            
            serial_number = int(whole_line[0])

            # checking scope of drawing
            if drawing_scope == "edge":
                if serial_number != self.study_number or\
                    int(whole_line[1]) != cc_in or\
                    int(whole_line[2]) != edge_in:
                        continue
            elif drawing_scope == "cc":
                if int(whole_line[0]) != self.study_number or\
                    int(whole_line[1]) != cc_in:
                        continue
            else:
                if int(whole_line[0]) != self.study_number:
                        continue

            # split bounding boxes
            bounding_boxes = whole_line[-1].rstrip().rsplit(",")
            # draw bronchi centrelines
            _draw_bronchi_centreline(serial_number, bounding_boxes, plt3d)  
            
            # draw vessel centreline
            cc = self.__extract_cc(int(whole_line[1]))
            edge = self.__extract_edge(cc, int(whole_line[2]))
            self.__draw_vessel_centreline(edge, plt3d)
            
        f.close()


    def get_bronchial_tree(self, study_number):
        """Read all the files from disc and create structure"""
        self.study_number = study_number
        file_name_prefix = self.dicom_file_name_prefix.\
            substitute(st=study_number)
        file_name_suffix = r"*[0-9].mat"
        start = time.time() # start measuring processing time

        # limit = 1000 # limiting number of files to process
        for file in os.listdir(self.dicom_directory):
            # if limit == 0: break
            if fnmatch.fnmatch(file, file_name_prefix + file_name_suffix):
                matstruct_contents = loadmat(self.dicom_directory + file)

                # extracting info from .mat files
                cc = matstruct_contents['info'][0, 0]['cc_number'][0, 0]
                edge_number = matstruct_contents['info'][0, 0] \
                                                    ['edge_number'][0, 0]
                image_number = matstruct_contents['info'][0, 0] \
                                                    ['image_number'][0, 0]
                p_t = matstruct_contents['info'][0, 0]['p_t'][0, 0]

                # add new image
                self.__add(cc, edge_number, image_number, p_t)
                # limit -= 1;

        end = time.time()
        print("Processing time: " + str((end - start)/60) + " min.")


    def load_detections(self, source_file, scale=1):
        """Loading original detections"""
        file_path = self.dicom_directory + source_file
        matstruct_contents = loadmat(file_path)
        number_of_records = matstruct_contents['tData_struct']['data']\
                                                            [0, 0][0, 0].size
        for i in range(number_of_records):
            record = matstruct_contents['tData_struct']['data'][0, 0]\
                                                            [0, 0][i, 0][0]
            record_split = record.split('_')
            if int(record_split[-4]) == self.study_number:
                image_number = int(record_split[-1].split('.')[0])
                det = matstruct_contents['tData_struct']['data']\
                                                            [0, 0][0, 1][i, 0]
                self.__add_detections(det, image_number, scale)


    def load_annotations(self, cc_in, edge_in, source="annotations.txt"):
        """Loading annotations for edge"""
        self.__clear_annotations()
        f = open(source, "r")
        while True:
            # reading next line from the file
            line = f.readline()

            # exit if reached the end
            if not line:
                break

            # skip if line starts with #
            if line[0] == '#':
                continue

            # split study no., cc no., edge no. and bounding boxes
            whole_line = line.rsplit(";")

            # check if it is the current study, cc and edge
            if int(whole_line[0]) != self.study_number or\
                    int(whole_line[1]) != cc_in or\
                    int(whole_line[2]) != edge_in:
                continue

            # split bounding boxes
            bounding_boxes = whole_line[-1].rstrip().rsplit(",")

            #draw each bounding box
            for bb in bounding_boxes:
                single_bb = bb.rsplit(" ")
                self.__add_annotations([int(single_bb[1]),
                                        int(single_bb[2]),
                                        int(single_bb[3]),
                                        int(single_bb[4])],
                                       int(single_bb[0]))
        f.close()
        
        
    def load_predictions2d_centrepoints(self, cc_in, edge_in, look_back):
        """Loading annotations for edge"""
        self.__clear_prediction_centrepoints()
        f = open("pre_processed_data/"+
                 f"look_back_{look_back}/inter_Pred_2d.txt", "r")
        while True:
            # reading next line from the file
            line = f.readline()
        
            # exit if reached the end
            if not line:
                break
        
            # skip if line starts with #
            if line[0] == '#':
                continue
        
            # split study no., cc no., edge no. and bounding boxes
            whole_line = line.rsplit(";")
        
            # check if it is the current study, cc and edge
            if int(whole_line[0]) != self.study_number or\
                    int(whole_line[1]) != cc_in or\
                    int(whole_line[2]) != edge_in:
                continue
        
            # split bounding boxes
            centrepoint = whole_line[3].rstrip().rsplit(" ")
        
            self.__add_prediction_centrepoints([int(centrepoint[1]),
                                                int(centrepoint[2])],
                                                int(centrepoint[0]))
        f.close()
    

    def extract_candidate_edges(self, threshold=0.5):
        """Extract candidate edges"""
        if not os.path.exists("candidate_edges"):
            os.makedirs("candidate_edges")

        f = open(f"candidate_edges/edges_{self.study_number}.txt", "w")
        f.write("Study Number : " + str(self.study_number) + '\n')
        edges_count = 0
        for connected_component in self.__connected_components:
            for edge in connected_component.edges:
                detected_cs = 0
                for _cs in edge.cross_sections:
                    if len(_cs.detections) > 0:
                        detected_cs += 1
                percent_detected = detected_cs / len(edge.cross_sections)
                # print(percent_detected)
                if percent_detected > threshold:
                    edges_count += 1
                    f.write("Connected component: " +
                            str(connected_component.cc_number) + '\n')
                    f.write("\tEdge: " + str(edge.edge_number) +
                            f" | {round(percent_detected*100, 1)}% detection" + '\n')
                    for cs in edge.cross_sections:
                        f.write("\t\tCrossSection:\t" + str(cs.p_t) +
                                "\tImage Number:" + str(cs.image_number) +
                                "\t Detection:" + str(len(cs.detections)) + '\n')
        f.write("\nNumber of detected edges : " + str(edges_count) + '\n')
        f.close()


    def export_bt(self):
        """Printing a list of connected components"""
        if not os.path.exists("full_tree_export"):
            os.makedirs("full_tree_export")

        f = open(f"full_tree_export/bt_{self.study_number}.txt", "w")
        f.write("Study Number : " + str(self.study_number) + '\n')
        edges_count = 0
        for connected_component in self.__connected_components:
            f.write("Connected component: " + \
                    str(connected_component.cc_number) + '\n')
            # print("Connected component: ", str(connected_component.cc_number))
            for edge in connected_component.edges:
                edges_count += 1
                f.write("\tEdge: " + str(edge.edge_number) + '\n')
                for cs in edge.cross_sections:
                    f.write("\t\tCrossSection:\t" + str(cs.p_t) +
                            "\tImage Number:" + str(cs.image_number) +
                            "\t Detection:" + str(len(cs.detections)) + '\n')
        f.write("\nTotal number of edges : " + str(edges_count) + '\n')
        f.close()


    def show_edge(self, cc_in, edge_in, detections=True, annotations=True,
                  centrepoint_prediction=True):
        """Showing edge on the screen"""
        # extract connected component
        cc = self.__extract_cc(cc_in)
        # extract edge
        edge = self.__extract_edge(cc, edge_in)
        
        # printing cross sectional images in ascending order
        for cs in reversed(edge.cross_sections):
            file_path = self.dicom_directory + \
                        self.dicom_file_name_prefix. \
                            substitute(st=self.study_number) + \
                        '_image_number_' + str(cs.image_number) + ".mat"
            matstruct_contents = loadmat(file_path)
            img = matstruct_contents['image'][0, 0]['vi_4']

            fig, ax = plt.subplots(1)
            fig.subplots_adjust(0, 0, 1, 1)
            plt.gcf().set_facecolor('black')

            if annotations:
                for a in cs.annotations:
                    rect = patches.Rectangle((a[0], a[1]), a[2], a[3],
                                             linewidth=2, edgecolor='y',
                                             facecolor='none')
                    ax.add_patch(rect)
                    ax.plot((a[0]+a[2]/2), ((a[1]+a[3]/2)), 'ro')
                    ax.patch.set_edgecolor('black')

            if detections:
                for d in cs.detections:
                    rect = patches.Rectangle((d[0], d[1]), d[2], d[3],
                                             linewidth=2, edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)
                    ax.plot((d[0]+d[2]/2), ((d[1]+d[3]/2)), 'yo')
                    ax.patch.set_edgecolor('black')
                    
            if centrepoint_prediction:
                for d in cs.prediction_centrepoint:
                    ax.plot((d[0]), ((d[1])), 'go')
                    ax.patch.set_edgecolor('black')

            ax.axis('off')

            plt.text(2, -6, "Study: " + str(self.study_number),
                     bbox=dict(fill=True, edgecolor='none', linewidth=2))
            plt.text(40, -6, "Img: " + str(cs.image_number),
                     bbox=dict(fill=True, edgecolor='none', linewidth=2))

            ax.imshow(img, cmap="gray")
            plt.pause(0.1)


    def visualise_edge_3d(self, cc_in, edge_in, show_first=None):
        """Show edge in 3D"""
        # extract connected component
        cc = self.__extract_cc(cc_in)        
        # extract edge
        edge = self.__extract_edge(cc, edge_in)
        
        if show_first == None:
            show_first = len(edge.cross_sections)
        
        # setup plot
        plt3d = plt.figure(figsize=(7, 7)).gca(projection='3d')
        plt3d.set_xlabel('x')
        plt3d.set_ylabel('y')
        plt3d.set_zlabel('z')

        # printing cross sectional images in 3D
        for cs in reversed(edge.cross_sections):
            if show_first == 0:
                break
            file_path = self.dicom_directory + \
                        self.dicom_file_name_prefix. \
                        substitute(st=self.study_number) + \
                        '_image_number_' + str(cs.image_number) + ".mat"
            matstruct_contents = loadmat(file_path)

            img = matstruct_contents['image'][0, 0]['vi_4']
            xx = matstruct_contents['image'][0, 0]['xi_4']
            yy = matstruct_contents['image'][0, 0]['yi_4']
            zz = matstruct_contents['image'][0, 0]['zi_4']

            img_shape_x = img.shape[1]
            img_shape_y = img.shape[0]

            # point 1 - base: top left
            x0 = xx[0][0]
            y0 = yy[0][0]
            z0 = zz[0][0]
            # point 2: top right
            x_global = xx[0][img_shape_y-1]
            y_global = yy[0][img_shape_y-1]
            z1 = zz[0][img_shape_y-1]
            # point 3: bottom left
            x2 = xx[img_shape_x-1][0]
            y2 = yy[img_shape_x-1][0]
            z2 = zz[img_shape_x-1][0]

            ux, uy, uz = [x_global-x0, y_global-y0, z1-z0]
            vx, vy, vz = [x2-x0, y2-y0, z2-z0]

            u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]
            normal = np.array(u_cross_v)

            # image normalisation between 0 and 1 for RBGA format
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

            # forming RGBA image
            img_rgba = np.empty([img_shape_x, img_shape_y, 4])
            for dim_1 in range(img_rgba.shape[0]):                              # pylint: disable=unsubscriptable-object
                for dim_2 in range(img_rgba.shape[1]):                          # pylint: disable=unsubscriptable-object
                    img_rgba[dim_1][dim_2][0] = img_rgba[dim_1][dim_2][1] = \
                                    img_rgba[dim_1][dim_2][2] = img[dim_1][dim_2]
                    img_rgba[dim_1][dim_2][3] = 1

            # plot the surface
            plt3d.plot_surface(xx, yy, zz, alpha=1, rstride=5,
                               cstride=2, facecolors=img_rgba)

            # plot the base point
            plt3d.plot([x0], [y0], [z0], color='red', marker='o',
                       markersize=3, alpha=1)

            # set the normal vector to start at the point on the plane
            startX = x0
            startY = y0
            startZ = z0

            # plot the normal vector
            plt3d.quiver([startX], [startY], [startZ], [normal[0]],
                         [normal[1]], [normal[2]], length=0.003,
                         linewidths=(2,), edgecolor="red")
            # plot the vector 1 -> 2 : green
            plt3d.quiver([x0], [y0], [z0], [x_global-x0], [y_global-y0], [z1-z0],
                         linewidths=(2,), length=0.1, edgecolor="lime")
            # plot the vector 1 -> 3 : ornage
            plt3d.quiver([x0], [y0], [z0], [x2-x0], [y2-y0], [z2-z0],
                         linewidths=(2,), length=0.1, edgecolor="orange")
            show_first -= 1

        plt.show()


    def visualise_vessel_centreline_3d(self, cc_in, edge_in, ds="edge"):
        """Show vessel(s) centreline(s) in 3D"""
        # setup plot
        plt3d = plt.figure(figsize=(7, 7)).gca(projection='3d')
        plt3d.set_xlabel('x')
        plt3d.set_ylabel('y')
        plt3d.set_zlabel('z')

        if ds == "edge":
            # extract connected component
            cc = self.__extract_cc(cc_in)
            # extract edge
            edge = self.__extract_edge(cc, edge_in)
            self.__draw_vessel_centreline(edge, plt3d)
        elif ds == "cc":
            # extract connected component
            cc = self.__extract_cc(cc_in)
            for edge in cc.edges:
                self.__draw_vessel_centreline(edge, plt3d)
        elif ds == "bt":
            for cc in self.__connected_components:
                for edge in cc.edges:
                    self.__draw_vessel_centreline(edge, plt3d)

        plt.show()


    def visualise_annotations_3d(self, cc_in, edge_in, ds="edge"):
        """Show ground truth for edge/cc/bt in 3D"""
        # setup plot
        plt3d = plt.figure(figsize=(7, 7)).gca(projection='3d')
        plt3d.set_xlabel('x')
        plt3d.set_ylabel('y')
        plt3d.set_zlabel('z')
        
        self.__draw_annotations(cc_in, edge_in, plt3d, drawing_scope=ds)
        
        plt.show()
        

    # annotate edges
    def annotate_edge(self, cc_in, edge_in):
        """Annotate edge"""
        # handling mouse events ----------------------------------------------
        def mouse_events_(event, x, y, flags, param):                           #pylint: disable=unused-argument
            nonlocal drawing, x_global, y_global, img_curr, img_prev
            nonlocal bounding_box

            # click left mouse button down
            if event == cv2.EVENT_LBUTTONDOWN:
                # allowing only one bounding box
                if bounding_box:
                    return
                drawing = True
                x_global, y_global = x, y
                cv2.rectangle(img_curr, (x, y), (x, y), (0, 255, 255), 1)

            # mouse move
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    img_curr = img_prev.copy()
                    cv2.rectangle(img_curr, (x_global, y_global), (x, y),
                                  (0, 255, 255), 1)
                    cv2.circle(img_curr,
                               (int((x+x_global)/2), int((y+y_global)/2)),
                               radius=0, color=(0, 0, 255), thickness=2)
                    area = abs(x-x_global) * abs(y-y_global)
                    cv2.displayStatusBar(window_name, "Area: " + str(area),
                                         1000)

            # click left mouse button up
            elif event == cv2.EVENT_LBUTTONUP:
                # allowing only one bounding box
                if bounding_box:
                    return
                drawing = False
                img_prev = img_curr.copy()
                cv2.rectangle(img_curr, (x_global, y_global), (x, y),
                              (0, 255, 255), 1)
                bounding_box.append(x_global)
                bounding_box.append(y_global)
                bounding_box.append(x-x_global)
                bounding_box.append(y-y_global)
                print("Bounding Box coordinates", bounding_box)
                print("Area:", abs(x-x_global)*abs(y-y_global))
        # --------------------------------------------------------------------

        # action on pressing keyboard buttons---------------------------------
        def action_on_key_(key):
            nonlocal img_rgb_orig, img_curr, img_prev, bounding_box
            nonlocal annotations, cs

             # c for clear
            if key == 99:
                img_curr = img_rgb_orig.copy()
                img_prev = img_rgb_orig.copy()
                bounding_box = []
                print("Cleaning...done")
                return 0

            # escape for exit
            if key == 27:
                print("Exiting...done")
                return 1

            # space for next image
            if key == 32:
                # adding image number and bounding box
                if bounding_box:
                    annotations.append([cs.image_number, *bounding_box])            #pylint: disable=undefined-loop-variable
                bounding_box = []
                print("Going to next image...\n")
                return 2

            return 0
        # --------------------------------------------------------------------

        # drawing bounding box and centre oint--------------------------------
        def draw_bounding_box_and_centre_point(img, detections,
                                               border_col, cp_col):
            for d in detections:
                d = [round(_) for _ in d]

                # draw horizontal lines
                if d[2] >= 0:
                    step = 1
                else:
                    step = -1

                for pix in range(d[0], d[0]+d[2]+step, step):
                    if d[1] in range(0, img.shape[0]) and\
                                        pix in range(0, img.shape[1]):
                        img[d[1]][pix][0] = border_col[0]
                        img[d[1]][pix][1] = border_col[1]
                        img[d[1]][pix][2] = border_col[2]
                    if d[1]+d[3] in range(0, img.shape[0]) and\
                                        pix in range(0, img.shape[1]):
                        img[d[1]+d[3]][pix][0] = border_col[0]
                        img[d[1]+d[3]][pix][1] = border_col[1]
                        img[d[1]+d[3]][pix][2] = border_col[2]

                # draw vertical lines
                if d[3] >= 0:
                    step = 1
                else:
                    step = -1

                for pix in range(d[1], d[1]+d[3]+step, step):
                    if d[0] in range(0, img.shape[1]) and\
                                        pix in range(0, img.shape[0]):
                        img[pix][d[0]][0] = border_col[0]
                        img[pix][d[0]][1] = border_col[1]
                        img[pix][d[0]][2] = border_col[2]
                    if d[0]+d[2] in range(0, img.shape[1]) and\
                                        pix in range(0, img.shape[0]):
                        img[pix][d[0]+d[2]][0] = border_col[0]
                        img[pix][d[0]+d[2]][1] = border_col[1]
                        img[pix][d[0]+d[2]][2] = border_col[2]

                # draw the centre poround as a cross
                cp_x = round(d[1]+d[3]/2)
                cp_y = round(d[0]+d[2]/2)

                if cp_x in range(0, img.shape[1]) and\
                    cp_y in range(0, img.shape[0]):

                    img[cp_x][cp_y][0] = cp_col[0]
                    img[cp_x][cp_y][1] = cp_col[1]
                    img[cp_x][cp_y][2] = cp_col[2]

                if cp_x+1 in range(0, img.shape[1]) and\
                    cp_y in range(0, img.shape[0]):

                    img[cp_x+1][cp_y][0] = cp_col[0]
                    img[cp_x+1][cp_y][1] = cp_col[1]
                    img[cp_x+1][cp_y][2] = cp_col[2]

                if cp_x-1 in range(0, img.shape[1]) and\
                    cp_y in range(0, img.shape[0]):

                    img[cp_x-1][cp_y][0] = cp_col[0]
                    img[cp_x-1][cp_y][1] = cp_col[1]
                    img[cp_x-1][cp_y][2] = cp_col[2]

                if cp_x in range(0, img.shape[1]) and\
                    cp_y+1 in range(0, img.shape[0]):

                    img[cp_x][cp_y+1][0] = cp_col[0]
                    img[cp_x][cp_y+1][1] = cp_col[1]
                    img[cp_x][cp_y+1][2] = cp_col[2]

                if cp_x in range(0, img.shape[1]) and\
                    cp_y-1 in range(0, img.shape[0]):

                    img[cp_x][cp_y-1][0] = cp_col[0]
                    img[cp_x][cp_y-1][1] = cp_col[1]
                    img[cp_x][cp_y-1][2] = cp_col[2]
            return img
        # --------------------------------------------------------------------

        # drawing manual annotations -----------------------------------------
        def draw_manual_annotations(img, cc_in, edge_in, image_number):
            f = open("annotations.txt", "r")
            while True:
                # reading next line from the file
                line = f.readline()

                # exit if reached the end
                if not line:
                    break

                # skip if line starts with #
                if line[0] == '#':
                    continue

                # split study no., cc no., edge no. and bounding boxes
                whole_line = line.rsplit(";")

                # check if it is the current study, cc and edge
                if int(whole_line[0]) != self.study_number or\
                        int(whole_line[1]) != cc_in or\
                        int(whole_line[2]) != edge_in:
                    continue

                # split bounding boxes
                bounding_boxes = whole_line[-1].rstrip().rsplit(",")

                #draw each bounding box
                for bb in bounding_boxes:
                    single_bb = bb.rsplit(" ")
                    if int(single_bb[0]) == image_number:
                        bb_list = [[int(single_bb[1]), int(single_bb[2]),
                                    int(single_bb[3]), int(single_bb[4])]]
                        img = draw_bounding_box_and_centre_point\
                                (img, bb_list,
                                 border_col=[0, 255, 255], cp_col=[0, 0, 255])
            f.close()
            return img
        # --------------------------------------------------------------------

        # extract connected component
        cc_in_obj = ConnectedComponent(cc_in)
        if cc_in_obj not in self.__connected_components:
            raise ValueError(f"Component number {cc_in} does not " +
                             f"exist in study {self.study_number}")
        index = self.__connected_components.index(cc_in_obj)
        cc = self.__connected_components[index]

        # extract edge
        edge_in_obj = Edge(edge_in)
        if edge_in_obj not in cc.edges:
            raise ValueError(f"Edge number {edge_in} does not " +
                             f"exist in connected component {cc_in} " +
                             f"in study {self.study_number}")
        index = cc.edges.index(edge_in_obj)
        edge = cc.edges[index]

        # list of annotation for the current edge
        annotations = []
        # window header
        window_name = 'Annotate: LB-bb, space-next, c-clean, ESC-exit'
        # file for writing annotations
        f = open("annotations.txt", "a")

        # annotate cross-sections one by one
        for cs in reversed(edge.cross_sections):

            print("Showing study:", self.study_number, " cc:", cc_in,
                  " edge:", edge_in, " cs: ", cs.p_t,
                  " image number:", cs.image_number)

            # getting the data file
            file_path = self.dicom_directory + \
                        self.dicom_file_name_prefix. \
                            substitute(st=self.study_number) + \
                        '_image_number_' + str(cs.image_number) + ".mat"
            matstruct_contents = loadmat(file_path)

            # getting raw image in HU
            img_raw = matstruct_contents['image'][0, 0]['vi_4']

            # normalise values
            img_norm = cv2.normalize(img_raw, None, alpha=0, beta=1,
                                     norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_32F)

            # forming RGB image
            img_rgb_orig = np.empty([img_norm.shape[0], img_norm.shape[1], 3])
            for dim_1 in range(img_norm.shape[0]):
                for dim_2 in range(img_norm.shape[1]):
                    img_rgb_orig[dim_1][dim_2][0] = \
                    img_rgb_orig[dim_1][dim_2][1] = \
                        img_rgb_orig[dim_1][dim_2][2] = img_norm[dim_1][dim_2]

            # draw automatic detections
            img_rgb_orig = draw_bounding_box_and_centre_point(\
                img_rgb_orig, cs.detections, border_col=[0, 0, 255],
                cp_col=[0, 255, 255])

            # draw manual annotations
            img_rgb_orig = draw_manual_annotations(img_rgb_orig,
                                                   cc_in, edge_in,
                                                   cs.image_number)

            # init nonlocal variables
            drawing = False
            x_global = 0
            y_global = 0
            bounding_box = []

            # init images
            img_curr = img_rgb_orig.copy()
            img_prev = img_rgb_orig.copy()

            # window settings
            cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
            cv2.setMouseCallback(window_name, mouse_events_)

            # infinite loop while showing
            while True:

                cv2.resizeWindow(window_name, 400, 400)
                cv2.moveWindow(window_name, 0, 50)
                # cv2.moveWindow(window_name, -1800,-100)
                cv2.imshow(window_name, img_curr)

                key_result = action_on_key_(cv2.waitKey(10))
                if key_result:
                    break

            cv2.destroyAllWindows()

            # if ESC pressed stop annotating the edge
            if key_result == 1:
                break

        if key_result != 1 and annotations:
            print("\nFinishing... Writing...")
            coordinates_outer = ""
            # forming string of coordinates to write
            f = open("annotations.txt", "a")
            for a in annotations:
                print(f"Img {a[0]} area:", abs(a[3]) * abs(a[4]))
                coordinates_inner = " ".join([str(a[0]), str(a[1]), str(a[2]),
                                              str(a[3]), str(a[4])])
                coordinates_outer = ",".join([coordinates_outer,
                                              coordinates_inner])

            f.write(";".join([str(self.study_number), str(cc_in), str(edge_in),
                              coordinates_outer[1:len(coordinates_outer)]]) + '\n')
            f.close()


def _get_bronchus_centrepoint_coordinate_3d(sn, bounding_boxes, bb_index):
    """Extract 3D centrepoint coordinate given bounding box in 2D image patch.
        sn - study number - integer
        bounding_boxes - list of coordinates in 2D image patch: 
                        [0] - image number within a study,
                        [1] - x-coordinate of the top left corner point,
                        [2] - y-coordinate of the top left corner point,
                        [3] - width (x-axis) of the bounding box
                        [4] - height (y-axis) of the bounding box
        bb_index - index of bounding box from list bounding_boxes
    """
    # extract single bounding box from the list and split by elements
    single_bb = bounding_boxes[bb_index].rsplit(" ")
    
    # extract image number
    image_number = single_bb[0]
    
    # setting up file path and extract the content from Matlab file
    file_path = DICOM_DIRECTORY + \
                DICOM_FILE_NAME_PREFIX. \
                substitute(st=sn) + \
                '_image_number_' + image_number + ".mat"
    matstruct_contents = loadmat(file_path)
    
    # getting x coordinate
    delta_x_x = matstruct_contents['image'][0, 0]['xi_4'][0][1] -\
                matstruct_contents['image'][0, 0]['xi_4'][0][0]
    
    delta_x_y = matstruct_contents['image'][0, 0]['xi_4'][1][0] -\
                matstruct_contents['image'][0, 0]['xi_4'][0][0]
                
    centre_point_3d_x = matstruct_contents['image'][0, 0]['xi_4']\
                        [int(single_bb[2])][int(single_bb[1])] +\
                        delta_x_x * int(single_bb[3]) / 2 +\
                        delta_x_y * int(single_bb[4]) / 2
    
    # getting y coordinate
    delta_y_x = matstruct_contents['image'][0, 0]['yi_4'][0][1] -\
                matstruct_contents['image'][0, 0]['yi_4'][0][0]
    
    delta_y_y = matstruct_contents['image'][0, 0]['yi_4'][1][0] -\
                matstruct_contents['image'][0, 0]['yi_4'][0][0]
                
    centre_point_3d_y = matstruct_contents['image'][0, 0]['yi_4']\
                        [int(single_bb[2])][int(single_bb[1])] +\
                        delta_y_x * int(single_bb[3]) / 2 +\
                        delta_y_y * int(single_bb[4]) / 2
    
    # getting z coordinate
    delta_z_x = matstruct_contents['image'][0, 0]['zi_4'][0][1] -\
                matstruct_contents['image'][0, 0]['zi_4'][0][0]
    
    delta_z_y = matstruct_contents['image'][0, 0]['zi_4'][1][0] -\
                matstruct_contents['image'][0, 0]['zi_4'][0][0]
                
    centre_point_3d_z = matstruct_contents['image'][0, 0]['zi_4']\
                        [int(single_bb[2])][int(single_bb[1])] +\
                        delta_z_x * int(single_bb[3]) / 2 +\
                        delta_z_y * int(single_bb[4]) / 2
    
    return centre_point_3d_x, centre_point_3d_y, centre_point_3d_z, image_number


def _draw_bronchi_centreline(sn, bounding_boxes, plt3d):
    """Draw broncus centreline in 3D
        sn - study number - integer
        bounding_boxes - list of coordinates in 2D image patch: 
                        [0] - image number within a study,
                        [1] - x-coordinate of the top left corner point,
                        [2] - y-coordinate of the top left corner point,
                        [3] - width (x-axis) of the bounding box
                        [4] - height (y-axis) of the bounding box
        plt3d - matplotlib object
    """
    number_of_bb = len(bounding_boxes)
    prev_point = ()
    for bb in range(number_of_bb):
        centre_point_3d_x, centre_point_3d_y, centre_point_3d_z, image_number =\
            _get_bronchus_centrepoint_coordinate_3d(sn, bounding_boxes,bb)

        # plot points - first and last are orange
        if bb == 0 or bb == number_of_bb-1:
            point_colour = "orange"
        else:
            point_colour = "black"

        plt3d.plot([centre_point_3d_x], [centre_point_3d_y], 
                   [centre_point_3d_z], color=point_colour, marker='o',
                   markersize=3, alpha=1)

        # plot lines connecting points
        if not bb == 0:
            plt3d.plot([centre_point_3d_x, prev_point[0]],
                       [centre_point_3d_y, prev_point[1]],
                       [centre_point_3d_z, prev_point[2]], 
                       color="lime", linewidth=1)
            
        # storing coordinates of the previous point
        prev_point = (centre_point_3d_x, centre_point_3d_y, 
                      centre_point_3d_z)


def _draw_all_annotations(annotations_source, plt3d):
    """Draw all ground truth data points in one 3D plot"""
    # reading the file with annotations
    f = open(annotations_source, "r")
    while True:
        # reading next line from the file
        line = f.readline()

        # exit if reached the end
        if not line:
            break

        # skip if line starts with #
        if line[0] == '#':
            continue

        # split study no., cc no., edge no. and bounding boxes
        whole_line = line.rsplit(";")

        # split bounding boxes
        bounding_boxes = whole_line[-1].rstrip().rsplit(",")
        
        # skip edges where there are only 1 or 2 data points
        if len(bounding_boxes) < 3:
            continue
        
        study_number = int(whole_line[0])
        
        # draw bronchi centrelines
        _draw_bronchi_centreline(study_number, bounding_boxes, plt3d)
        
    f.close()


def _read_gt_from_files(from_file, look_back):
    # open files with with preprocessed X and Y
    f_x = open("pre_processed_data/"+
                   f"look_back_{look_back}/{from_file}_X.txt", "r")
    f_y = open("pre_processed_data/"+
                   f"look_back_{look_back}/{from_file}_Y.txt", "r")
    
    # init lists of data and lables
    all_x = []
    all_y = []
    
    # read data and lables in
    while True:
        # reading next line from the files
        line_x_all = f_x.readline()
        line_y_all = f_y.readline()

        # exit if reached the end
        if not line_x_all or not line_y_all:
            break

        # process X data
        all_x_line = []
        whole_line_x = line_x_all.rstrip().rsplit(",")
        for time_point in whole_line_x:
            single_point = time_point.rsplit(" ")
            all_x_line.append(list(map(np.float32 ,single_point)))
        all_x.append(all_x_line)    

        # process Y lables
        whole_line_y = line_y_all.rstrip().rsplit(" ")
        all_y.append(list(map(np.float32, whole_line_y)))
        
    # convert into np arrays
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    f_x.close()
    f_y.close()
    
    return all_x, all_y


def _export_shuffled_data(look_back, suffix, X, y):
    # open files for writing
    f_x = open("pre_processed_data/"+
               f"look_back_{look_back}/{suffix}_X.txt", "w")
    f_y = open("pre_processed_data/"+
               f"look_back_{look_back}/{suffix}_Y.txt", "w")
    
    # data shapes
    n_rec = X.shape[0]
    n_row = X.shape[1]
    
    # export to txt files
    for rec in range(n_rec):
        # export X data
        str_export = ""
        for row in range(n_row):
            str_point = " ".join((str(X[rec][row][0]), str(X[rec][row][1]),
                                    str(X[rec][row][2])))
            str_export += str_point
            if row != n_row-1:
                str_export += ','
        str_export += '\n'
        f_x.write(str_export) 
        
        # export Y lables
        str_export = " ".join((str(y[rec][0]), str(y[rec][1]), 
                               str(y[rec][2]))) + '\n'
        f_y.write(str_export) 
        
    f_x.close()
    f_y.close()


def _line_plane_collision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    """Find intersection point of a line and a plane in 3D"""
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
 
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


def shuffle_and_split_original_annotations(source="annotations.txt", 
                                           train_size=0.65, test_size=0.2,
                                           val_size=0.15, random_state=78):
        """Read, shuffle and split the original annotations into training,
        validation and test data sets"""
        # open file with original annotations
        f_annotations = open(source, "r")
        
        # reading all lines into a list
        file_lines = []
        while True:
            # reading next line from the file
            line = f_annotations.readline()

            # exit if reached the end
            if not line:
                break

            # skip if line starts with #
            if line[0] == '#':
                continue
            
            # split into [study no., cc no., edge no. and bounding boxes]
            whole_line = line.rstrip().rsplit(";")
            
            # split bounding boxes
            bounding_boxes = whole_line[-1].rsplit(",")
            
            # skip edges where there are only 1 or 2 data points
            if len(bounding_boxes) < 3:
                continue
            
            file_lines.append(line)

        # shuffle the list
        random.seed(random_state)        
        file_lines = random.sample(file_lines, k=len(file_lines))
        
        # find last indicies of train, val and test sets
        last_train_element_index = int(len(file_lines) * train_size)
        last_val_element_index = int(len(file_lines) * 
                                     (train_size + val_size))
        last_test_element_index = len(file_lines)
        
        
        f_train = open("annotations_train.txt", "w")
        f_val = open("annotations_val.txt", "w")
        f_test = open("annotations_test.txt", "w")
        
        def write_list_to_file(list_of_strings, file):
            for line in list_of_strings:
                file.write(line)
        
        write_list_to_file(file_lines[:last_train_element_index], f_train)
        write_list_to_file(file_lines[last_train_element_index:
                                      last_val_element_index], f_val)
        write_list_to_file(file_lines[last_val_element_index:
                                      last_test_element_index], f_test)
                        

def preprocess_annotations_centrepoints(look_back=5, random_state=42):
    """Read 2D annotations and convert to sequences of centrepoints of 
    length look_back in 3D space. Assume original annotations have
    already been split into train/val/test sets"""
    
    def __preprocess_single_set(set_name):
        f_annotations = open(f"annotations_{set_name}.txt", "r")
        # open files for writing preprocssed data
        f_x = open("pre_processed_data/"+
                   f"look_back_{look_back}/{set_name}_X.txt", "w")
        f_y = open("pre_processed_data/"+
                   f"look_back_{look_back}/{set_name}_Y.txt", "w")
    
        while True:
            # reading next line from the file
            line = f_annotations.readline()

            # exit if reached the end
            if not line:
                break

            # split into [study no., cc no., edge no. and bounding boxes]
            whole_line = line.rstrip().rsplit(";")
            study_number = whole_line[0]
        
            # split bounding boxes
            bounding_boxes = whole_line[-1].rsplit(",")
                    
            # extract centrepoints of bounding boxes in 3D space
            centrepoints_3d_coordinates = []
            for bb in range(len(bounding_boxes)):
                centre_point_3d_x, centre_point_3d_y, centre_point_3d_z, sn =\
                    _get_bronchus_centrepoint_coordinate_3d(study_number,
                                                            bounding_boxes,
                                                            bb)
                centrepoints_3d_coordinates.append([centre_point_3d_x, 
                                                    centre_point_3d_y, 
                                                    centre_point_3d_z])
                # to be deleted
                # print(round(centre_point_3d_x,5))
                # if round(centre_point_3d_x,4) == 282.4686:
                #     print("SN:", study_number, " cc:", whole_line[1],
                #           " edge:", whole_line[2])
                #     print(round(centre_point_3d_y,5))
                #     print(bounding_boxes[bb])
                # -------------
                
        
            # generate sequences of look_back length
            for num in range (1, len(centrepoints_3d_coordinates)-1):
                # check if padding is needed
                if (num + 1) - look_back >= 0:
                    X = centrepoints_3d_coordinates[(num+1)-look_back:num+1]
                else:
                    # padding with zeroes
                    number_of_zeroes = look_back - num - 1
                    X = [[0, 0, 0]] * number_of_zeroes
                    X += centrepoints_3d_coordinates[0:num+1]
                Y = centrepoints_3d_coordinates[num+1]
                
                # write data X
                for i in range(len(X)):
                    f_x.write(str(X[i][0]) + ' ')
                    f_x.write(str(X[i][1]) + ' ')
                    f_x.write(str(X[i][2]))
                    if i != len(X)-1:
                        f_x.write(",")
                f_x.write('\n')
                
                # write data Y
                for y in Y:
                    f_y.write(str(y) + ' ')
                f_y.write('\n')
            
        f_annotations.close()
        f_x.close()
        f_y.close()
        
    def __shuffle_set(set_name):
        set_x, set_y = _read_gt_from_files(set_name, look_back)
        set_x, set_y = shuffle(set_x, set_y, random_state=random_state)
        _export_shuffled_data(look_back, set_name, set_x, set_y)
        
    # checking if relevant folders exist or create ones
    if not os.path.exists("pre_processed_data"):
            os.makedirs("pre_processed_data")
    if not os.path.exists(f"pre_processed_data/look_back_{look_back}"):
            os.makedirs(f"pre_processed_data/look_back_{look_back}")
            
    # preprocessing sets: extract 3D coordinates
    __preprocess_single_set("train")
    __preprocess_single_set("val")
    __preprocess_single_set("test")
    __preprocess_single_set("test_special")

    # shuffling sets
    __shuffle_set("train")
    __shuffle_set("val")
    __shuffle_set("test")


def draw_input_histograms(look_back):
    all_x, all_y = _read_gt_from_files("train", look_back)
    x_flatten = all_x[:,:,0].flatten()
    y_flatten = all_x[:,:,1].flatten()
    z_flatten = all_x[:,:,2].flatten()
    
    plt.style.use('default')
    fig, axs = plt.subplots(3, 1)
    ax0, ax1, ax2 = axs.flatten()
    
    ax0.hist(x_flatten, bins=50, density=True, range=(100, x_flatten.max()))
    ax0.set_xlabel('x')
    
    ax1.hist(y_flatten, bins=50, density=True, range=(50, y_flatten.max()))
    ax1.set_xlabel('y')
    
    ax2.hist(z_flatten, bins=50, density=True, range=(50, z_flatten.max()))
    ax2.set_xlabel('z')
    
    fig.tight_layout()
    plt.show()


def visualise_all_annotations(annotations_source):
    """Visualise all annotations in 3D"""
    # setup plot
    plt.style.use('dark_background')
    plt3d = plt.figure(figsize=(7, 7)).gca(projection='3d')
    plt3d.set_xlabel('x')
    plt3d.set_ylabel('y')
    plt3d.set_zlabel('z')
    
    _draw_all_annotations(annotations_source, plt3d)
    plt.show()


def network_centreline(look_back, input_scaler, output_scaler):

    class Centrepoint_Net(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim, 
                     num_layers, drop_prob=0.2):
            super(Centrepoint_Net, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.lstm = nn.LSTM(input_size =self.input_dim, 
                                hidden_size=self.hidden_dim,
                                num_layers=self.num_layers,
                                batch_first=True)
            # self.dropout = nn.Dropout(drop_prob)
            self.fc = nn.Linear(in_features=self.hidden_dim, 
                                 out_features=self.output_dim)
            # self.fc2 = nn.Linear(in_features=self.hidden_dim, 
            #                      out_features=self.output_dim)
            
            self.relu = nn.CELU()

        def forward(self, x):
            lstm_out, hidden = self.lstm(x)
            fc_out = self.fc(hidden[0][0,:,:])
            relu_out = self.relu(fc_out)
            return relu_out

    def __scale_inputs(i_sc, tr_x, vl_x, ts_x, tsp_x):
        """Scale inputs"""
        for axis in range(3):
            i_sc.fit(tr_x[:,:,axis])
            tr_x[:,:,axis] = input_scaler.transform(tr_x[:,:,axis])
            vl_x[:,:,axis] = input_scaler.transform(vl_x[:,:,axis])
            ts_x[:,:,axis] = input_scaler.transform(ts_x[:,:,axis])
            tsp_x[:,:,axis] = input_scaler.transform(tsp_x[:,:,axis])
        return tr_x, vl_x, ts_x, tsp_x

    def __scale_outputs(o_sc_0, o_sc_1, o_sc_2, tr_y, vl_y, ts_y, tsp_y):
        """Scale outputs"""
        # reshaping outputs to 2-dim arrays
        tr_y = tr_y.reshape(tr_y.shape[0], tr_y.shape[1], 1)
        vl_y = vl_y.reshape(vl_y.shape[0], vl_y.shape[1], 1)
        ts_y = ts_y.reshape(ts_y.shape[0], ts_y.shape[1], 1)
        tsp_y = tsp_y.reshape(tsp_y.shape[0], tsp_y.shape[1], 1)
        
        # scaling first dimension / feature
        o_sc_0.fit(tr_y[:,0,:])
        tr_y[:,0,:] = o_sc_0.transform(tr_y[:,0,:])
        vl_y[:,0,:] = o_sc_0.transform(vl_y[:,0,:])
        ts_y[:,0,:] = o_sc_0.transform(ts_y[:,0,:])       
        tsp_y[:,0,:] = o_sc_0.transform(tsp_y[:,0,:])
        
        # scaling second dimension / feature
        o_sc_1.fit(tr_y[:,1,:])
        tr_y[:,1,:] = o_sc_1.transform(tr_y[:,1,:])
        vl_y[:,1,:] = o_sc_1.transform(vl_y[:,1,:])
        ts_y[:,1,:] = o_sc_1.transform(ts_y[:,1,:])
        tsp_y[:,1,:] = o_sc_1.transform(tsp_y[:,1,:])

        # scaling third dimension / feature
        o_sc_2.fit(tr_y[:,2,:])
        tr_y[:,2,:] = o_sc_2.transform(tr_y[:,2,:])
        vl_y[:,2,:] = o_sc_2.transform(vl_y[:,2,:])
        ts_y[:,2,:] = o_sc_2.transform(ts_y[:,2,:])
        tsp_y[:,2,:] = o_sc_2.transform(tsp_y[:,2,:])

        # reshaping back to 1-dim array
        tr_y = tr_y.reshape(tr_y.shape[0], tr_y.shape[1])
        vl_y = vl_y.reshape(vl_y.shape[0], vl_y.shape[1])
        ts_y = ts_y.reshape(ts_y.shape[0], ts_y.shape[1])
        tsp_y = tsp_y.reshape(tsp_y.shape[0], tsp_y.shape[1])
        return tr_y, vl_y, ts_y, tsp_y, o_sc_0, o_sc_1, o_sc_2
    
    def __unscale_outputs(out, o_sc_0, o_sc_1, o_sc_2):
        """Unscale outputs"""
        out = out.reshape(out.shape[0], out.shape[1], 1)
        out[:,0,:] =  o_sc_0.inverse_transform(out[:,0,:])
        out[:,1,:] =  o_sc_1.inverse_transform(out[:,1,:])
        out[:,2,:] =  o_sc_2.inverse_transform(out[:,2,:])
        out = out.reshape(out.shape[0], out.shape[1])
        return out
        
    # reading ground truth from text files
    train_x, train_y = _read_gt_from_files("train", look_back)
    val_x, val_y = _read_gt_from_files("val", look_back)
    test_x, test_y = _read_gt_from_files("test", look_back)
    test_sp_x, test_sp_y = _read_gt_from_files("test_special", look_back)
    
    # scaling inputs
    if input_scaler != None:
        train_x, val_x, test_x, test_sp_x = __scale_inputs(input_scaler, 
                                                           train_x, val_x, 
                                                           test_x, test_sp_x)

    # init output scalers
    output_scaler_0 = output_scaler
    output_scaler_1 = output_scaler
    output_scaler_2 = output_scaler
    
    # scaling outputs
    if output_scaler != None:
        train_y, val_y, test_y, test_sp_y, output_scaler_0, output_scaler_1, \
        output_scaler_2 = __scale_outputs(output_scaler_0, output_scaler_1, 
                                          output_scaler_2, 
                                          train_y, val_y, test_y, test_sp_y)

    # storing set sizes
    train_size = train_x.shape[0]
    val_size = val_x.shape[0]
    test_size = test_x.shape[0]
    test_sp_size = test_sp_x.shape[0]
    print("Train size:", train_size, "Validation size:", val_size, 
          "Test size", test_size, "Test special size", test_sp_size)
    
    # converting to tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), 
                               torch.from_numpy(train_y))
    val_data = TensorDataset(torch.from_numpy(val_x), 
                             torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), 
                              torch.from_numpy(test_y))
    test_sp_data = TensorDataset(torch.from_numpy(test_sp_x),
                                 torch.from_numpy(test_sp_y))
    
    # seeting batch size
    batch_size = 32
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    test_sp_loader = DataLoader(test_sp_data, shuffle=True, batch_size=batch_size)
    
    # choosing CPU vs GPU
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    
    # ******************************
    # setting up NN parameters
    input_dim = 3
    output_dim = 3
    hidden_dim = 32
    num_layers = 1
    epochs = 600
    learning_rate = 0.005
    # ******************************
    
    model = Centrepoint_Net(input_dim, output_dim, 
                            hidden_dim, num_layers).to(device)
    print(model)
    criterion = nn.MSELoss(reduction="sum")
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # train and validate network
    train_losses = []
    train_losses_scaled = []
    val_losses = []
    val_losses_scaled = []
    
    for epoch in range(epochs):
        # ---------------------------------
        # train model
        train_running_loss = 0
        train_running_loss_scaled = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # PyTorch calculates gradients by accumulating contributions to them 
            # Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()
            # Forward pass through the network.
            output = model(inputs)
            loss = criterion(output, labels)
            # Calculate gradients.
            loss.backward()
            # Minimise the loss according to the gradient.
            optimiser.step()
            train_running_loss += loss.item()
            # scale back if output scaler was applied
            if output_scaler != None:
                output = __unscale_outputs(output.to("cpu").detach().numpy(), 
                                           output_scaler_0,
                                           output_scaler_1,
                                           output_scaler_2)
                labels = __unscale_outputs(labels.to("cpu").detach().numpy(), 
                                           output_scaler_0,
                                           output_scaler_1,
                                           output_scaler_2)
                output = torch.Tensor(output)
                labels = torch.Tensor(labels)
                # calculate and accumulate loss
                loss_scaled = criterion(output, labels)
                train_running_loss_scaled += loss_scaled.item()
        # calculate average training loss
        train_avg_loss = round(train_running_loss/train_size, 5)
        train_losses.append(train_avg_loss)

        # ---------------------------------
        # validate model
        with torch.no_grad():
            val_running_loss = 0
            val_running_loss_scaled = 0
            # switch model to evaluation mode
            model.eval()
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                # calculate and accumulate loss
                loss = criterion(output, labels)
                val_running_loss += loss.item()
                # scale back if output scaler was applied
                if output_scaler != None:
                    output = __unscale_outputs(np.array(output.to("cpu")), 
                                                      output_scaler_0,
                                                      output_scaler_1,
                                                      output_scaler_2)
                    
                    labels = __unscale_outputs(np.array(labels.to("cpu")), 
                                                      output_scaler_0,
                                                      output_scaler_1,
                                                      output_scaler_2)
                    output = torch.Tensor(output)
                    labels = torch.Tensor(labels)
                    # calculate and accumulate loss
                    loss_scaled = criterion(output, labels)
                    val_running_loss_scaled += loss_scaled.item()
                # print out true and predicted values
                print("O", output[0])
                print("L", labels[0])
            # switch model back to training mode
            model.train()
        # calculate average validation loss
        val_avg_loss = round(val_running_loss/val_size, 5)
        val_losses.append(val_avg_loss)
 
        if output_scaler == None:
            print(f"Epoch:{epoch}, Loss:", train_avg_loss, 
                  "Val_loss:", val_avg_loss)
        else:
            # calculate average trainig loss_scaled
            train_avg_loss_scaled = round(train_running_loss_scaled/train_size, 5)
            train_losses_scaled.append(train_avg_loss_scaled)
            # calculate average validation loss_scaled
            val_avg_loss_scaled = round(val_running_loss_scaled/val_size, 5)
            val_losses_scaled.append(val_avg_loss_scaled)
            print(f"Epoch:{epoch}, Loss_scaled:", train_avg_loss_scaled, 
                  "Val_loss_scaled:", val_avg_loss_scaled)
      
    # ---------------------------------
    # evaluate model on the test data
    with torch.no_grad():
        # init losses
        test_running_loss = 0
        test_running_loss_scaled = 0
        # switch model to evaluation mode
        model.eval()
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            # calculate and accumulate loss
            loss = criterion(output, labels)
            test_running_loss += loss.item()
            # scale back if output scaler was applied
            if output_scaler != None:
                output = __unscale_outputs(np.array(output.to("cpu")), 
                                                    output_scaler_0,
                                                    output_scaler_1,
                                                    output_scaler_2)
                    
                labels = __unscale_outputs(np.array(labels.to("cpu")), 
                                                    output_scaler_0,
                                                    output_scaler_1,
                                                    output_scaler_2)
                output = torch.Tensor(output)
                labels = torch.Tensor(labels)
                # calculate and accumulate loss
                loss_scaled = criterion(output, labels)
                test_running_loss_scaled += loss_scaled.item()
        # switch model back to training mode
        model.train()
    # calculate average validation loss
    test_loss = round(test_running_loss/test_size, 5)
    
    # ---------------------------------
    # evaluate model on the special test data
    with torch.no_grad():
        # init losses
        test_sp_running_loss = 0
        test_sp_running_loss_scaled = 0
        # switch model to evaluation mode
        model.eval()
        for inputs, labels in test_sp_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            # calculate and accumulate loss
            loss = criterion(output, labels)
            test_sp_running_loss += loss.item()
            # scale back if output scaler was applied
            if output_scaler != None:
                output = __unscale_outputs(np.array(output.to("cpu")), 
                                                    output_scaler_0,
                                                    output_scaler_1,
                                                    output_scaler_2)
                    
                labels = __unscale_outputs(np.array(labels.to("cpu")), 
                                                    output_scaler_0,
                                                    output_scaler_1,
                                                    output_scaler_2)
                output = torch.Tensor(output)
                labels = torch.Tensor(labels)
                # calculate and accumulate loss
                loss_scaled = criterion(output, labels)
                test_sp_running_loss_scaled += loss_scaled.item()
        # switch model back to training mode
        model.train()
    # calculate average validation loss
    test_sp_loss = round(test_sp_running_loss/test_sp_size, 5)
    
    
    if output_scaler == None:
        train_mean_str = "Last 20 ep train mean:" + \
                    str(round(np.mean(train_losses[-20:]), 2))
        val_mean_str = "Last 20 ep val mean:" + \
                    str(round(np.mean(val_losses[-20:]), 2))
        test_str = "Test loss:" + str(round(test_loss, 2))
        test_sp_str = "Test special loss:" + str(round(test_sp_loss, 2))
    else:
        train_mean_str = "Last 20 ep train mean:" + \
                    str(round(np.mean(train_losses_scaled[-20:]), 2))
        val_mean_str = "Last 20 ep val mean:" + \
                    str(round(np.mean(val_losses_scaled[-20:]), 2))
        test_loss_scaled = round(test_running_loss_scaled/test_size, 5)
        test_str = "Test loss:" + str(round(test_loss_scaled, 2))            
        test_sp_loss_scaled = round(test_sp_running_loss_scaled/test_sp_size, 5)
        test_sp_str = "Test special loss:" + str(round(test_sp_loss_scaled, 2))
    print(train_mean_str)
    print(val_mean_str)
    print(test_str)
    print(test_sp_str)

    # plot loss-epoch graph
    plt.style.use('dark_background')
    if output_scaler != None:
        plt.plot(train_losses_scaled, color="red")
        plt.plot(val_losses_scaled, color="green")
        plt.plot([epochs-1], [test_loss_scaled], marker='o', 
                 markersize=3, color="yellow")
    else:
        plt.plot(train_losses, color="red")
        plt.plot(val_losses, color="green")
        plt.plot(epochs, test_loss, color="blue")
        plt.plot([epochs-1], [test_loss], marker='o', 
                 markersize=3, color="yellow")
    plt.ylabel("Avg MSE Loss")
    plt.xlabel("Epoch")
    plt.gca().legend(('Train', 'Validation', 'Test'))
    plt.ylim(0, 7)
    text_line = f"Look back:{look_back}\n" +\
                f"Batch size:{batch_size}\n" +\
                f"LR:{learning_rate}\n" +\
                f"Input scaler:{str(input_scaler)}\n" +\
                f"Output scaler:{str(output_scaler)}"
    # plt.text(-0.5, 95, text_line, horizontalalignment="left",
    #           verticalalignment="top", multialignment="left", fontsize=7)
    # plt.text(-0.5, 20, str(model), horizontalalignment="left",
    #           verticalalignment="top", multialignment="left", fontsize=7)
    # plt.text(-0.5, 25, mean_str, horizontalalignment="left",
    #           verticalalignment="top", multialignment="left", fontsize=7)
    
    
    plt.show()
    

def spline_interpolation_centreline(dataset, look_back, pol_deg):
    set_x, set_y = _read_gt_from_files(dataset, look_back)
    set_len = len(set_x)
    
    running_loss = 0
    for edge in range(0, set_len):
        set_x_zero_reduced = (set_x[edge][set_x[edge] != 
                                          [[0.0,0.0,0.0]]]).reshape(-1,3)
        x_sample = set_x_zero_reduced[:, 0]
        y_sample = set_x_zero_reduced[:, 1]
        z_sample = set_x_zero_reduced[:, 2]
        if pol_deg >= set_x_zero_reduced.shape[0]:
            pol_deg_adj = set_x_zero_reduced.shape[0] - 1
        else:
            pol_deg_adj = pol_deg
        tck, u = interpolate.splprep([x_sample,y_sample,z_sample], 
                                     s=1, k=pol_deg_adj)
        u_pred = np.array([1 + 1/set_x_zero_reduced.shape[0]])
        x_pred, y_pred, z_pred = interpolate.splev(u_pred, tck)
        point_pred = np.array([x_pred, y_pred, z_pred])
        point_loss = mean_squared_error(set_y[edge], point_pred)
        running_loss += point_loss
        if point_loss > 10:
            print(f"Point {edge} loss:", point_loss)
            print(set_y[edge])
        
    avg_loss = running_loss / set_len
    print("Avg MSE Loss", round(avg_loss, 4))


def spline_interpolation_full(source, look_back, pol_deg):
    """Prediction of next point of trajectory with spline
    interpolation in 3D"""
    
    def _write_data3d_out(file, sn, ccn, en, data, im_number, err=None):
        """Writing out 3D prediction"""
        s = str(sn) + ';' + str(ccn) + ';' + str(en) + ';'
        for i in range(data.shape[0]):
            s += str(im_number[i]) + ' ' + str(data[i][0]) + ' ' + \
            str(data[i][1]) + ' ' + str(data[i][2])
            if i != data.shape[0]-1:
                s += ','
        if err != None:
            s += ' ' + err
        file.write(s + '\n')
 
        
    def _write_data2d_out(file, sn, ccn, en, data, im_number, err):
        """Writing out 2D prediction"""
        s = str(sn) + ';' + str(ccn) + ';' + str(en) + ';'
        s += str(im_number) + ' ' + str(data[0]) + ' ' + str(data[1])
        s += ' ' + err
        print(s)
        file.write(s + '\n')


    # source data file
    f_source = open(source, "r")
    # open files for writing
    # files for pre-processed sequences as inputs for the model
    f_x = open("pre_processed_data/"+
               f"look_back_{look_back}/inter_X.txt", "w")
    f_y = open("pre_processed_data/"+
               f"look_back_{look_back}/inter_Y.txt", "w")
    # output files with predictions with 3D and 2D coordinates
    f_pred3d = open("pre_processed_data/"+
                    f"look_back_{look_back}/inter_Pred_3d.txt", "w")
    f_pred2d = open("pre_processed_data/"+
                    f"look_back_{look_back}/inter_Pred_2d.txt", "w")
    
    f_big_errors = open("pre_processed_data/"+
                        f"look_back_{look_back}/inter_big_errors.txt", "w")
    
    # running sum of losses and number of predictions made
    running_loss = 0
    prediction_count = 0
    
    while True:
        # reading next line from the file
        line = f_source.readline()
    
        # exit if reached the end
        if not line:
            break
    
        # skip if line starts with #
        if line[0] == '#':
            continue
        
        # stop reading the file when reach &
        if line[0] == '&':
            break
        
        # split into [study no., cc no., edge no. and bounding boxes]
        whole_line = line.rstrip().rsplit(";")
        
        study_number = whole_line[0]
        cc_number = whole_line[1]
        edge_number = whole_line[2]
        
        # form string as edge ID for the dict
        edge_id = study_number + ' ' + cc_number + ' ' + edge_number
        
        # split bounding boxes
        bounding_boxes = whole_line[-1].rsplit(",")
        
        # skip edges where there are only 1 or 2 data points
        if len(bounding_boxes) < 3:
            continue
        
        # init list of bb centrepoints coordinates in 3D and corresponding
        # image numbers
        centrepoints_3d_coordinates = []
        image_numbers = []
        
        # extract centrepoints of bounding boxes in 3D space
        for bb in range(len(bounding_boxes)):    
            centre_point_3d_x, centre_point_3d_y,\
            centre_point_3d_z, image_number =\
                _get_bronchus_centrepoint_coordinate_3d(study_number,
                                                        bounding_boxes,
                                                        bb)
            centrepoints_3d_coordinates.append([centre_point_3d_x, 
                                                centre_point_3d_y, 
                                                centre_point_3d_z])
            image_numbers.append(image_number)
        
        # generate sequences and predict the next point
        for num in range (1, len(centrepoints_3d_coordinates)-1):
            # sample at most look_back consequetive points
            if (num + 1) - look_back >= 0:
                sample_data = centrepoints_3d_coordinates[(num+1)- \
                                                          look_back:num+1]
                sample_in = image_numbers[(num+1)-look_back:num+1]
            else:
                sample_data = centrepoints_3d_coordinates[0:num+1]
                sample_in = image_numbers[0:num+1]
            # get next "true" data point
            true_data = centrepoints_3d_coordinates[num+1]
            true_in = str(image_numbers[num+1])
            
            # getting x,y,z coordinates of the sampling points
            sample_data = np.array(sample_data)
            x_sample = sample_data[:,0]
            y_sample = sample_data[:,1]
            z_sample = sample_data[:,2]
            
            # getting x,y,z coordinates of the predicting point
            true_data = np.array(true_data)
            x_true = true_data[0]
            y_true = true_data[1]
            z_true = true_data[2]
            point_true = [x_true, y_true, z_true]
            
            # writing out pre-processed sequence and true point
            _write_data3d_out(f_x, study_number, cc_number, 
                              edge_number, sample_data, sample_in)

            _write_data3d_out(f_y, study_number, cc_number, 
                              edge_number, true_data.reshape(-1, 3), [true_in])
            
            # interpolate the sample points
            tck, u = interpolate.splprep([x_sample, y_sample, z_sample], 
                                         s=2, k=pol_deg)
            # last_node = len(tck[1][0]) - 1
            
            # get plane of the next point in the sequence
            
            # file path to the original data of the next point
            next_point_file_path = DICOM_DIRECTORY + \
                                    DICOM_FILE_NAME_PREFIX. \
                                    substitute(st=int(study_number)) + \
                                    '_image_number_' + str(true_in) + ".mat"
            
            matstruct_contents = loadmat(next_point_file_path)
    
            # load image content and x,y,z coordinates of each voxel
            img = matstruct_contents['image'][0, 0]['vi_4']
            xx = matstruct_contents['image'][0, 0]['xi_4']
            yy = matstruct_contents['image'][0, 0]['yi_4']
            zz = matstruct_contents['image'][0, 0]['zi_4']

            # get shape of the cross-sectional image 
            img_shape_x = img.shape[1]
            img_shape_y = img.shape[0]

            # point 1 - base: top left
            x0 = xx[0][0]
            y0 = yy[0][0]
            z0 = zz[0][0]
            # point 2: top right
            x1 = xx[0][img_shape_y-1]
            y1 = yy[0][img_shape_y-1]
            z1 = zz[0][img_shape_y-1]
            # point 3: bottom left
            x2 = xx[img_shape_x-1][0]
            y2 = yy[img_shape_x-1][0]
            z2 = zz[img_shape_x-1][0]

            # get vectors
            ux, uy, uz = [x1-x0, y1-y0, z1-z0]
            vx, vy, vz = [x2-x0, y2-y0, z2-z0]

            # find cross-product
            u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]
            
            # define plane
            plane_normal = np.array(u_cross_v)
            plane_point = np.array([x0, y0, z0])
            
            # define ray
            ray_direction = np.array([tck[1][0][1]-tck[1][0][0],
                                      tck[1][1][1]-tck[1][1][0],
                                      tck[1][2][1]-tck[1][2][0]])
            ray_point = np.array([tck[1][0][1], tck[1][1][1], tck[1][2][1]])
            
            # find intersection between the fitted line and cross-sectional
            # plane of the next point
            inter_point = _line_plane_collision(plane_normal, plane_point, 
                                                ray_direction, ray_point)
            
            # calculate MSE
            error = mean_squared_error(inter_point, point_true)
            
            # record edge if error greater than treshold
            if error > 2.22:
                 s = study_number + ' ' + cc_number + ' ' + edge_number + \
                     ' ' + true_in + ' ' + str(error)
                 f_big_errors.write(s + '\n') 
            
            # accumulate loss and count of predictions made
            running_loss += error
            prediction_count += 1
            
            _write_data3d_out(f_pred3d, study_number, cc_number, 
                              edge_number, inter_point.reshape(-1, 3), 
                              [true_in], str(round(error, 4)))
            
            
            cm = np.zeros((img_shape_y, img_shape_x))
            for i in range(img_shape_y):
                for j in range(img_shape_x):
                    original_point = [xx[j][i], yy[j][i], zz[j][i]]
                    cm[i][j] = mean_squared_error(inter_point, 
                                                    original_point)
            
            idx = np.where(cm == cm.min())
            coordinates_2d = [idx[0][0], idx[1][0]]
            
            _write_data2d_out(f_pred2d, study_number, cc_number, 
                              edge_number, coordinates_2d, 
                              true_in, str(round(error, 4)))
            
    print("Average MSE Loss:", round(running_loss/prediction_count, 4))
   
    f_big_errors.close()    
    f_x.close()
    f_y.close()
    f_pred3d.close()
    f_pred2d.close()
            
    
    
    
    