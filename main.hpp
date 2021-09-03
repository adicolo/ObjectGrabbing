// Copyright (c) 2021 Adrien CRIDELAUZE
#ifndef MAIN_HPP
#define MAIN_HPP

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/gripper.h>

#include <opencv2/core/persistence.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_sensor.hpp>
#include <librealsense2/hpp/rs_processing.hpp>
#include <librealsense2/rsutil.h>

#include "examples_common.h"

using namespace std;
using namespace cv;
using namespace dnn;
using namespace rs2;

// Initialize the parameters
float confThreshold = 0.5;		// Confidence threshold, confidence under which detected object are ignored
float nmsThreshold = 0.4;		// Non-maximum suppression threshold
int inpWidth = 416;				// Width of network's input image for object detection
int inpHeight = 416;			// Height of network's input image for object detection
vector<string> classes;			// Array of classes of detectable objects
double depth_precision = 10.0;	// Accepted depth variation between two measurements
double error = 0.00001;			// Value under which the elements of the matrices for the robot control are set to zero
double z_threshold = 300;		// Depth under which the robot control goes straight to the object position

// Initialize the paths of the files
string data_folder = "../data/";							// Relative path to the data folder
string calibParams_filepath = data_folder + "params.xml";	// Extrinsic and intrisic calibration parameters, generated automatically
string gTc_filepath = data_folder + "g_T_c.xml";			// Hand-eye matrix
string classesFile = data_folder + "coco.names";			// Classes names for object detection
String modelConfiguration = data_folder + "yolov3.cfg";		// Configuration file for the dnn model
String modelWeights = data_folder + "yolov3.weights";		// Weight file for the dnn model

// Get the (X, Y, Z) position of the object in the camera frame
// from its bounding box edges and the depth frame of the camera
void getPos(int left, int top, int right, int bottom, double* px, double* py, double* pz, Mat& depth_frame) {
	// Read the matrices from file
	Mat depth_intrinsic_matrix, color_intrinsic_matrix, rotation_extrinsic_matrix, translation_extrinsic_matrix;
	FileStorage fs(calibParams_filepath, FileStorage::READ);
	fs["color_intrinsic_matrix"] >> color_intrinsic_matrix;					// Intrinsic matrix for the RGB frame
	// fs["depth_intrinsic_matrix"] >> depth_intrinsic_matrix;				// Intrinsic matrix for the depth frame, here identical to the color_intrinsic_matrix
	// fs["rotation_extrinsic_matrix"] >> rotation_extrinsic_matrix;		// Rotation submatrix of the transformation from depth frame to RGB frame, here identity matrix
	// fs["translation_extrinsic_matrix"] >> translation_extrinsic_matrix;	// Translation submatrix of the transformation from depth frame to RGB frame, here null vector

	// Create a 3D point from the center of the bounding box and express it in the camera frame
	double u = (left + right)/2;				// Horizontal coordinate of the center of the box
	double v = (top + bottom)/2;				// Vertical coordinate of the center of the box
	double z = depth_frame.at<ushort>(u, v);	// Depth of the center of the box measured by the camera
	double p[3] = {u, v, 1};
	Mat point_image = Mat(3, 1, CV_64F, p);								// Center of the box expressed in the image frame
	Mat point_camera = color_intrinsic_matrix.inv() * point_image * z;	// Center of the box expressed in the camera frame

	// Assign the values
	*px = point_camera.at<double>(0,0);
	*py = point_camera.at<double>(1,0);
	*pz = point_camera.at<double>(2,0);
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, Mat& depth_frame, double* px, double* py, double* pz) {
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty()) {
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);

	// Display the coordinates at the center of the box
	getPos(left, top, right, bottom, px, py, pz, depth_frame);
	string position = "(" + to_string(*px) + ", " + to_string(*py) + ", " + to_string(*pz) + ")";
	putText(frame, position, Point((right+left)/2, (top+bottom)/2), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,255,0),1);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, Mat& depth_frame, const vector<Mat>& outs, double* px, double* py, double* pz) {
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i) {
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold) {
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
	vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i) {
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, depth_frame, px, py, pz);
	}
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net) {
	static vector<String> names;
	if (names.empty()) {
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
		names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

// Save the intrinsic and extrinsic parameters in a file
void getParams(depth_frame depth, video_frame color) {
	stream_profile depth_profile = depth.get_profile();
	stream_profile color_profile = color.get_profile();
	video_stream_profile cvsprofile(color_profile);
	video_stream_profile dvsprofile(depth_profile);
	rs2_intrinsics color_intrinsic = cvsprofile.get_intrinsics();				// Calculated intrinsic parameters for the color profile
	rs2_intrinsics depth_intrinsic = dvsprofile.get_intrinsics();				// Calculated intrinsic parameters for the depth profile
	rs2_extrinsics extrinsic = depth_profile.get_extrinsics_to(color_profile);	// Calculated transformation parameters from the depth frame to the RGB frame

	FileStorage fs(calibParams_filepath, FileStorage::WRITE);

	// For each parameter :
	// - Express the parameters in a matrix : camera matrix for the intrinsics and rotation matrix and translation vector for the extrinsics
	// - Construct a Mat object from a array of arrays
	// - Save the matrices in a file
	double md[3][3] = {{depth_intrinsic.fx, 0, depth_intrinsic.ppx}, {0, depth_intrinsic.fy, depth_intrinsic.ppy}, {0, 0, 1}};
	Mat depth_intrinsic_matrix = Mat(3, 3, CV_64F, md);
	fs << "depth_intrinsic_matrix" << depth_intrinsic_matrix;				// Intrinsic matrix for the depth frame, here identical to the color_intrinsic_matrix

	double mc[3][3] = {{color_intrinsic.fx, 0, color_intrinsic.ppx}, {0, color_intrinsic.fy, color_intrinsic.ppy}, {0, 0, 1}};
	Mat color_intrinsic_matrix = Mat(3, 3, CV_64F, mc);
	fs << "color_intrinsic_matrix" << color_intrinsic_matrix;				// Intrinsic matrix for the RGB frame

	double mR[3][3] = {{extrinsic.rotation[0], extrinsic.rotation[3], extrinsic.rotation[6]}, {extrinsic.rotation[1], extrinsic.rotation[4], extrinsic.rotation[7]}, {extrinsic.rotation[2], extrinsic.rotation[5], extrinsic.rotation[8]}};
	Mat rotation_extrinsic_matrix = Mat(3, 3, CV_64F, mR);
	fs << "rotation_extrinsic_matrix" << rotation_extrinsic_matrix;			// Rotation submatrix of the transformation from depth frame to RGB frame, here identity matrix

	double mT[3] = {extrinsic.translation[0], extrinsic.translation[1], extrinsic.translation[2]};
	Mat translation_extrinsic_matrix = Mat(3, 1, CV_64F, mT);
	fs << "translation_extrinsic_matrix" << translation_extrinsic_matrix;	// Translation submatrix of the transformation from depth frame to RGB frame, here null vector
}

// Perform a matrix multiplication where each of the two 4x4 transformation matrices are represented by a column-major array of 16 elements
array<double, 16> dot(array<double, 16>& a, array<double, 16>& b) {
	array<double, 16> res;

	double res11 = a[0]*b[0] + a[4]*b[1] + a[8]*b[2] + a[12]*b[3];
	double res21 = a[1]*b[0] + a[5]*b[1] + a[9]*b[2] + a[13]*b[3];
	double res31 = a[2]*b[0] + a[6]*b[1] + a[10]*b[2] + a[14]*b[3];
	double res41 = a[3]*b[0] + a[7]*b[1] + a[11]*b[2] + a[15]*b[3];

	double res12 = a[0]*b[4] + a[4]*b[5] + a[8]*b[6] + a[12]*b[7];
	double res22 = a[1]*b[4] + a[5]*b[5] + a[9]*b[6] + a[13]*b[7];
	double res32 = a[2]*b[4] + a[6]*b[5] + a[10]*b[6] + a[14]*b[7];
	double res42 = a[3]*b[4] + a[7]*b[5] + a[11]*b[6] + a[15]*b[7];

	double res13 = a[0]*b[8] + a[4]*b[9] + a[8]*b[10] + a[12]*b[11];
	double res23 = a[1]*b[8] + a[5]*b[9] + a[9]*b[10] + a[13]*b[11];
	double res33 = a[2]*b[8] + a[6]*b[9] + a[10]*b[10] + a[14]*b[11];
	double res43 = a[3]*b[8] + a[7]*b[9] + a[11]*b[10] + a[15]*b[11];

	double res14 = a[0]*b[12] + a[4]*b[13] + a[8]*b[14] + a[12]*b[15];
	double res24 = a[1]*b[12] + a[5]*b[13] + a[9]*b[14] + a[13]*b[15];
	double res34 = a[2]*b[12] + a[6]*b[13] + a[10]*b[14] + a[14]*b[15];
	double res44 = a[3]*b[12] + a[7]*b[13] + a[11]*b[14] + a[15]*b[15];

	res = {res11, res21, res31, res41, res12, res22, res32, res42, res13, res23, res33, res43, res14, res24, res34, res44};

	return res;
}

#endif
