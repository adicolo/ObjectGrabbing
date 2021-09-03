// Copyright (c) 2021 Adrien CRIDELAUZE
#include "main.hpp.h"

int main(int argc, char** argv) {
	// Stop if the wrong number of arguments is given
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <robot-hostname> "
				<< "<total duration in seconds>" << std::endl;
		return -1;
	}

	try {
		// Configure the robot and the gripper
		franka::Robot robot(argv[1]);
		setDefaultBehavior(robot);
		franka::Gripper gripper(argv[1]);

		// Set the numer of via-points for every command and get the total duration
		const int nb_points = 3;
		double total_duration = std::stod(argv[2]);

		// Initialize the arrays to calculate the trajectories
		std::array<double, nb_points> via_times;								// Times at which the end-effector will be at the desired position
		std::array<std::array<double, 6>, nb_points> via_positions;				// Positions at which the end-effector will be at the desired time
		std::array<std::array<double, 6>, nb_points> via_velocities;			// Calculated velocities of each via-point
		std::array<std::array<double, 6>, nb_points> via_accelerations;			// Calculated accelerations of each via-point
		std::array<std::array<std::array<double, 6>, 6>, nb_points> via_coeff;	// Coefficients (six) for polynomial trajectory for each coordinate (position and orientation) and for each segment between via-points

		// First move the robot to a suitable joint configuration
		std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
		MotionGenerator motion_generator(0.2, q_goal);
		std::cout << "WARNING: This example will move the robot! "
				<< "Please make sure to have the user stop button at hand!" << std::endl
				<< "Press Enter to continue..." << std::endl;
		std::cin.ignore();
		robot.control(motion_generator);
		gripper.homing();
		std::cout << "Finished moving to initial joint configuration." << std::endl;

		// Set additional parameters always before the control loop, NEVER in the control loop!
		// Set collision behavior.
		robot.setCollisionBehavior(
			{{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
			{{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
			{{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
			{{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});

		// Load names of classes for the model
		ifstream ifs(classesFile.c_str());
		string line;
		while (getline(ifs, line)) classes.push_back(line);

		// Load the network
		Net net = readNetFromDarknet(modelConfiguration, modelWeights);

		// Configure the camera
		pipeline pipeline;
		config config;
		config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 60);	// Configure the format of the color stream
		config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 60);	// Configure the format of the depth stream
		pipeline.start(config);

		// Align the color frame to the depth frame
		rs2::align align(RS2_STREAM_COLOR);
		frameset frames = pipeline.wait_for_frames();
		frameset aligned_frames = align.process(frames);
		depth_frame depth = aligned_frames.get_depth_frame();
		video_frame color = aligned_frames.get_color_frame();

		// Calculate the camera parameters and store them in a file
		getParams(depth, color);
		Mat depth_frame, color_frame, blob;

		// Create a window
		static const string kWinName = "Object Detection";
		namedWindow(kWinName, WINDOW_AUTOSIZE);

		// Initialize the coordinates of the object in the camera frame
		double x, y, z;

		do {
			// Get the coordinates of the object from the camera
			double deltaz = depth_precision;
			double zprev;	// Depth of previous measure
			z = 0;
			do {
				// Get current frame
				frames = pipeline.wait_for_frames();
				aligned_frames = align.process(frames);
				depth = aligned_frames.get_depth_frame();
				color = aligned_frames.get_color_frame();

				// Convert the depth frame into a matrix
				int depth_width = depth.get_width();
				int depth_height = depth.get_height();
				depth_frame = Mat(Size(depth_width, depth_height), CV_16UC1, (void*)depth.get_data(), Mat::AUTO_STEP);

				// Convert the color frame into a matrix
				int color_width = color.get_width();
				int color_height = color.get_height();
				color_frame = Mat(Size(color_width, color_height), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);

				// Create a 4D blob from a frame.
				blobFromImage(color_frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

				//Sets the input to the network
				net.setInput(blob);

				// Runs the forward pass to get output of the output layers
				vector<Mat> outs;
				net.forward(outs, getOutputsNames(net));

				// Set the previous z
				if (z == 0)				// wrong measure
					zprev = -2*deltaz;	// far enough from zero to measure again
				else
					zprev = z;

				// Remove the bounding boxes with low confidence
				// Calculates the values of x, y, z of the object in the camera frame
				postprocess(color_frame, depth_frame, outs, &x, &y, &z);

				// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
				vector<double> layersTimes;
				double freq = getTickFrequency() / 1000;
				double t = net.getPerfProfile(layersTimes) / freq;
				string label = format("Inference time for a frame : %.2f ms", t);
				putText(color_frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

				imshow(kWinName, color_frame);
				imwrite(data_folder + "capture.png", color_frame);	// Save the last view of the camera for debugging
			} while (fabs(z - zprev) > deltaz);	// Continue until two successive measurements are close

			std::array<double, 16> initial_position;
			double time = 0.0;
			robot.control([&initial_position, &via_positions, &via_velocities, &via_accelerations, &time, &total_duration, &via_times, &via_coeff, &x, &y, &z](const franka::RobotState& robot_state, franka::Duration period) -> franka::CartesianPose {
				// Update time
				time += period.toSec();

				// Any number below this value will be set to zero
				const double err = error;

				// At time zero only
				if (time == 0.0) {
					// Get the current position
					initial_position = robot_state.O_T_EE_c;
					for (size_t i = 0; i < 16; i++)
						if (fabs(initial_position[i]) < err)
							initial_position[i] = 0;

					// Initialize the positions and orientations as the initial configuration
					for (size_t i = 0; i < nb_points; i++) {
						// Initialize the via-positions to the initial position
						for (size_t k = 0; k < 3; k++) {
							via_positions[i][k] = initial_position[12+k];
						}

						// Initialize the via-orientations to the initial orientation using roll-pitch-yaw convention
						double a = atan2(initial_position[1], initial_position[0]);
						double b = asin(-initial_position[2]);
						double c = atan2(initial_position[6], initial_position[10]);
						via_positions[i][3] = c;
						via_positions[i][4] = b;
						via_positions[i][5] = a;
					}

					// Read the hand-eye matrix from file
					FileStorage fs(gTc_filepath, FileStorage::READ);
					Mat gRc, gtc;
					fs["g_R_c"] >> gRc;	// Rotation submatrix of the transformation from the gripper (end-effector) frame to the camera frame
					fs["g_t_c"] >> gtc;	// Translation submatrix of the transformation from the gripper (end-effector) frame to the camera frame
					// Hand-eye matrix as a column-major array
					array<double, 16> gTc = {gRc.at<double>(0,0), gRc.at<double>(1,0), gRc.at<double>(2,0), 0, gRc.at<double>(0,1), gRc.at<double>(1,1), gRc.at<double>(2,1), 0, gRc.at<double>(0,2), gRc.at<double>(1,2), gRc.at<double>(2,2), 0, gtc.at<double>(0,0)*1e-03, gtc.at<double>(1,0)*1e-03, gtc.at<double>(2,0)*1e-03, 1};

					// Construct the matrix cTo from the coordinates calculated from the camera
					// Configuration of the object in the camera frame as a column-major array
					// The rotation is set to identity and the translation to the calculated position
					array<double, 16> cTo = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x*1e-03, y*1e-03, z*1e-03, 1};

					// Calculate the wanted position for the end effector
					array<double, 16> gTo = dot(gTc, cTo);				// Transformation matrix from the gripper to the object
					array<double, 16> OTo = dot(initial_position, gTo);	// Transformation matrix from the robot base to the object

					// Set the wanted position as the final point for the command
					if (z < z_threshold) {	// If the camera is close enough, the end-effector goes straight to the object
						via_positions[2][0] = OTo[12]; via_positions[2][1] = OTo[13]; via_positions[2][2] = OTo[14];
					} else {				// Else, the object could move so the end-effector goes halfway
						via_positions[2][0] = (OTo[12] + initial_position[12])/2; via_positions[2][1] = (OTo[13] + initial_position[13])/2; via_positions[2][2] = (OTo[14] + initial_position[14])/2;
					}
					// via_positions[2][3] = 0; via_positions[2][4] = 0; via_positions[2][5] = M_PI_2;	// Custom orientation for tests

					// Correct the altitude
					double zmax = -0.09;	// Depth value under which the end-effector cannot go
					if (via_positions[2][2] < zmax)
						via_positions[2][2] = zmax;

					// Set the middle point to be between the starting point and the end point for a linear trajectory
					for (size_t k = 0; k < 3; k++) {
						via_positions[1][k] = (via_positions[2][k] + via_positions[0][k]) / 2;
					}
					// via_positions[1][3] += 0; via_positions[1][4] += 0; via_positions[1][5] += 0;	// Custom orientation for tests

					// Initialize the times
					for (size_t i = 0; i < nb_points; i++) {
						via_times[i] = i * total_duration / (nb_points - 1);	// Times calculated so that the duration of all segments are equal
					}

					// Initialize the velocities
					for (size_t i = 0; i < nb_points; i++) {
						for (size_t j = 0; j < 6; j++)
							via_velocities[i][j] = 0;	// Set starting velocities to zero for continuity

						if (!((i == 0) || (i == nb_points-1))) {
							for (size_t j = 0; j < 6; j++)
								via_velocities[i][j] = (via_positions[i+1][j] - via_positions[i-1][j]) / (via_times[i+1] - via_times[i-1]);	// Set velocities to mean value for a trajectory from the previous point to the next one
						}

						for (size_t j = 0; j < 6; j++)
							if (fabs(via_velocities[i][j]) < err)
								via_velocities[i][j] = 0;	// Set ending velocities to zero for continuity
					}

					// Initialize the accelerations
					for (size_t i = 0; i < nb_points; i++) {
						for (size_t j = 0; j < 6; j++)
							via_accelerations[i][j] = 0;	// Set starting accelerations to zero for continuity

						if (!((i == 0) || (i == nb_points-1))) {
							for (size_t j = 0; j < 6; j++)
								via_accelerations[i][j] = (via_velocities[i+1][j] - via_velocities[i-1][j]) / (via_times[i+1] - via_times[i-1]);	// Set accelerations to mean value for a trajectory from the previous point to the next one
						}

						for (size_t j = 0; j < 6; j++)
							if (fabs(via_velocities[i][j]) < err)
								via_accelerations[i][j] = 0;	// Set ending accelerations to zero for continuity
					}

					// Initialize the coefficients respecting the conditions at limits
					double delta_time;
					for (size_t i = 0; i < nb_points-1; i++) {
						for (size_t k = 0; k < 6; k++) {
							via_coeff[i][k][0] = via_positions[i][k];
							via_coeff[i][k][1] = via_velocities[i][k];
							via_coeff[i][k][2] = via_accelerations[i][k] / 2;

							delta_time = via_times[i+1] - via_times[i];
							double b1 = via_positions[i+1][k] - via_coeff[i][k][0] - via_coeff[i][k][1]*delta_time - via_coeff[i][k][2]*std::pow(delta_time, 2.0);
							double b2 = via_velocities[i+1][k] - via_coeff[i][k][1] - 2*via_coeff[i][k][2]*delta_time;
							double b3 = via_accelerations[i+1][k] - 2*via_coeff[i][k][2];
							via_coeff[i][k][3] = b1*5/2/std::pow(delta_time, 3.0) - b2/4/std::pow(delta_time, 2.0) - b3/8/delta_time;
							via_coeff[i][k][4] = -b2/2/std::pow(delta_time, 3.0) + b3/4/std::pow(delta_time, 2.0);
							via_coeff[i][k][5] = -b1*3/2/std::pow(delta_time, 5.0) + b2*3/4/std::pow(delta_time, 4.0) - b3/8/std::pow(delta_time, 3.0);
						}
					}

				}

				// Initialize the via point and the matrix
				std::array<double, 6> points;							// Position and orientation of the current point
				std::array<double, 16> positions = initial_position;	// Matrix form of the current point

				// Initialize the current point
				int i;
				for (size_t k = 0; k < nb_points-1; k++) {
					if ((via_times[k] <= time) && (time <= via_times[k+1]))
						i = k;
				}

				// Calculate the positions and orientations
				double dt = time - via_times[i];	// Relative time
				for (size_t k = 0; k < 6; k++) {
					points[k] = via_coeff[i][k][0] + via_coeff[i][k][1]*dt + via_coeff[i][k][2]*std::pow(dt, 2.0) + via_coeff[i][k][3]*std::pow(dt, 3.0) + via_coeff[i][k][4]*std::pow(dt, 4.0) + via_coeff[i][k][5]*std::pow(dt, 5.0);
				}

				// Calculate the transformation matrix from the point configuration, using the roll-pitch-yaw convention for the orientation
				double ca = cos(points[5]);
				double cb = cos(points[4]);
				double cc = cos(points[3]);
				double sa = sin(points[5]);
				double sb = sin(points[4]);
				double sc = sin(points[3]);
				positions[0] = ca*cb; positions[4] = ca*sb*sc - sa*cc; positions[8] = ca*sb*cc + sa*sc; positions[12] = points[0];
				positions[1] = sa*cb; positions[5] = sa*sb*sc + ca*cc; positions[9] = sa*sb*cc - ca*sc; positions[13] = points[1];
				positions[2] = -sb; positions[6] = cb*sc; positions[10] = cb*cc; positions[14] = points[2];
				positions[3] = 0; positions[7] = 0; positions[11] = 0; positions[15] = 1;

				// Finish the trajectory
				if (time >= total_duration) {
					std::cout << std::endl << "Finished motion, shutting down example" << std::endl;
					return franka::MotionFinished(positions);
				}

				// Send the command to the robot
				return positions;
			});

		} while (z > z_threshold);	// Repeat until the camera is close enough

		// Try to grasp the object
		if (!gripper.grasp(0.0, 0.1, 60)) {
			std::cout << "Failed to grasp object." << std::endl;
		} else {
			// Check if object grasped
			franka::GripperState gripper_state = gripper.readOnce();
			if (!gripper_state.is_grasped) {
				std::cout << "Object lost." << std::endl;
			} else {
				std::cout << "Grasped object, will release it now." << std::endl;
				gripper.stop();
			}
		}

	} catch (const franka::Exception& e) {
		std::cout << e.what() << std::endl;
		return -1;
	}

	return 0;
}
