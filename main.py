#!/usr/bin/env python
"""Driver code"""

import bronchial_tree

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# *** SETTING UP PATHS ***

# external drive path
DICOM_DIR = r"F:\BronchialDetectionProject\DICOMDATA\\"\
                                                "crosssectionalimageData\\"

# laptop path
# DICOM_DIR = r"D:\UNSW\Research Project\BronchialDetectionProject\\"\
#                                         "DICOMDATA\crosssectionalimageData\\"

DICOM_FILE_NAME_PREFIX = "Bronchial_cross_sectional_image_data_Study_$st"

DETECTION_SOURCE_FILE = "trainingdata_struct.mat"
DETECTION_SOURCE_FILE_UPDATE = "trainingdata_reevaulated_struct.mat"

# *************************

## init bronchial tree object
bt = bronchial_tree.BronchialTree(DICOM_DIR, DICOM_FILE_NAME_PREFIX)            #pylint: disable=invalid-name

## load bronchial tree by study number
bt.get_bronchial_tree(203)

## load detections
bt.load_detections(DETECTION_SOURCE_FILE)
bt.load_detections(DETECTION_SOURCE_FILE_UPDATE, scale=300/161)

# ---- uncomment necessary actions below ----

## extract bronchial tree structure into a text file
# bt.export_bt()

## load manual annotations
# bt.load_annotations(cc_in=4, edge_in=118)
## load predicted coordinates in 2D
# bt.load_predictions2d_centrepoints(cc_in=2, edge_in=314, look_back=3)

## visualise an edge as a series of 2D images
## run in consol: %matplotlib inline
# bt.show_edge(4, 118)

## visualise an edge in 3D
## run in consol: %matplotlib auto
# bt.visualise_edge_3d(4, 118)

## visualise edge centrelines
# bt.visualise_vessel_centreline_3d(cc_in=4, edge_in=118, ds="edge")
## visualise centrelines of a connected component
# bt.visualise_vessel_centreline_3d(cc_in=4, edge_in=118, ds="cc")
## visualise centrelines of a study
# bt.visualise_vessel_centreline_3d(cc_in=4, edge_in=118, ds="bt")

## extract candidate edges with detection rate > threshold
# bt.extract_candidate_edges(threshold=0.5)

## run annotation tool
# bt.annotate_edge(cc_in=4, edge_in=118)

# bronchial_tree.shuffle_and_split_original_annotations()
# bronchial_tree.preprocess_annotations_centrepoints(look_back=5)
# bronchial_tree.visualise_all_annotations("annotations.txt")
# bronchial_tree.draw_input_histograms(5)

## running NN with various look_back and input/output scalers
# bronchial_tree.network_centreline(look_back=5, input_scaler=StandardScaler(), 
#                                   output_scaler = MinMaxScaler())

## running B-spline interpolation algorithm
# bronchial_tree.spline_interpolation_centreline("train", 5, 1)
# bronchial_tree.spline_interpolation_full("annotations.txt", 5, 1)