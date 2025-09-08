// Eye-tracker.cpp: define o ponto de entrada para o aplicativo.
//

#include "Eye-tracker.h"

void tracking() {
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize("../shape_predictor_68_face_landmarks.dat");
	
	cv::VideoCapture cap;
	cap.open("teste.mp4");
	cv::Mat frame;
	cv::VideoWriter writer("resultado.mp4",
		cv::VideoWriter::fourcc('H', '2', '6', '4'),
		30,
		cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT))
	);
	

	while (true) {
		cap >> frame;
		if (frame.empty()) break;

		dlib::cv_image<dlib::bgr_pixel> cimg(frame);
		std::vector<dlib::rectangle> faces = detector(cimg);
		std::vector<dlib::full_object_detection> shapes;
		for (unsigned long i = 0; i < faces.size(); i++) {
			shapes.push_back(pose_model(cimg, faces[i]));
		}

		std::vector<cv::Point2f> left_eye_points;
		std::vector<cv::Point2f> right_eye_points;

		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(36).x(), shapes.at(0).part(36).y()));
		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(37).x(), shapes.at(0).part(37).y()));
		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(38).x(), shapes.at(0).part(38).y()));
		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(39).x(), shapes.at(0).part(39).y()));
		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(40).x(), shapes.at(0).part(40).y()));
		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(41).x(), shapes.at(0).part(41).y()));

		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(42).x(), shapes.at(0).part(42).y()));
		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(43).x(), shapes.at(0).part(43).y()));
		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(44).x(), shapes.at(0).part(44).y()));
		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(45).x(), shapes.at(0).part(45).y()));
		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(46).x(), shapes.at(0).part(46).y()));
		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(47).x(), shapes.at(0).part(47).y()));

		cv::RotatedRect left_eye_rect = cv::minAreaRect(left_eye_points);
		cv::RotatedRect right_eye_rect = cv::minAreaRect(right_eye_points);

		cv::Mat left_rotated;
		cv::Mat right_rotated;
		cv::Point2f left_center;
		cv::Point2f right_center;
		cv::Mat left_bounding_crop;
		cv::Mat right_bounding_crop;
		cv::Mat left_gray_crop;
		cv::Mat right_gray_crop;

		cv::Mat left_rot_mat = cv::getRotationMatrix2D(left_eye_rect.center, left_eye_rect.angle, 1.0);
		cv::Mat right_rot_mat = cv::getRotationMatrix2D(right_eye_rect.center, right_eye_rect.angle, 1.0);
		cv::warpAffine(frame, left_rotated, left_rot_mat, frame.size());
		cv::warpAffine(frame, right_rotated, right_rot_mat, frame.size());
		cv::Size left_size = cv::Point2f(shapes.at(0).part(40).x()) - cv::Point2f(shapes.at(0).part(37).x());
		cv::Size right_size = cv::Point2f(shapes.at(0).part(46).x()) - cv::Point2f(shapes.at(0).part(43).x());
		cv::getRectSubPix(left_rotated, left_size, left_center, left_bounding_crop);
		cv::getRectSubPix(right_rotated, right_size, right_center, right_bounding_crop);

		cv::cvtColor(left_bounding_crop, left_gray_crop, cv::COLOR_BGR2GRAY);
		cv::cvtColor(right_bounding_crop, right_gray_crop, cv::COLOR_BGR2GRAY);

		cv::GaussianBlur(left_gray_crop, left_gray_crop, cv::Size(5, 5), 0);
		cv::GaussianBlur(right_gray_crop, right_gray_crop, cv::Size(5, 5), 0);

		for (int y = 0; y < left_gray_crop.rows; y++) {
			for (int x = 0; x < left_gray_crop.cols; x++) {
				for (int c = 0; c < left_gray_crop.channels(); c++) {
					left_gray_crop.at<cv::Vec3b>(y, x)[c] =
						/*cv::saturate_cast<uchar>(
							(alpha_slider / 100.0) * left_gray_crop.at<cv::Vec3b>(y, x)[c] + betha_slider
						);*/
						cv::saturate_cast<uchar>(
							left_gray_crop.at<cv::Vec3b>(y, x)[c]
						);
				}

			}
		}

		for (int y = 0; y < right_gray_crop.rows; y++) {
			for (int x = 0; x < right_gray_crop.cols; x++) {
				for (int c = 0; c < right_gray_crop.channels(); c++) {
					right_gray_crop.at<cv::Vec3b>(y, x)[c] =
						/*cv::saturate_cast<uchar>(
							(alpha_slider / 100.0) * right_gray_crop.at<cv::Vec3b>(y, x)[c] + betha_slider
						);*/
						cv::saturate_cast<uchar>(
							right_gray_crop.at<cv::Vec3b>(y, x)[c]
						);
				}
			}
		}

		cv::Mat left_bin_crop;
		cv::Mat right_bin_crop;

		/*cv::threshold(left_gray_crop, left_bin_crop, left_bin_slider, 255, cv::THRESH_BINARY_INV);
		cv::threshold(right_gray_crop, right_bin_crop, right_bin_slider, 255, cv::THRESH_BINARY_INV);*/

		cv::threshold(left_gray_crop, left_bin_crop, 127, 255, cv::THRESH_BINARY_INV);
		cv::threshold(right_gray_crop, right_bin_crop, 127, 255, cv::THRESH_BINARY_INV);

		cv::Mat bin_sum;
		cv::bitwise_and(left_bin_crop, right_bin_crop, bin_sum);

		std::vector<std::vector<cv::Point>> sum_contours;
		std::vector<cv::Vec4i> sum_hierarchy;
		std::vector<std::vector<cv::Point>> sum_contours_poly;
		std::vector<cv::Point2f> sum_centers;
		std::vector<float> sum_radius;

		cv::findContours(bin_sum, sum_contours, sum_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		sum_contours_poly.resize(sum_contours.size());
		sum_centers.resize(sum_contours.size());
		sum_radius.resize(sum_contours.size());

		for (size_t i = 0; i < sum_contours.size(); i++) {
			double circledist = 1.0;
			cv::approxPolyDP(sum_contours[i], sum_contours_poly[i], circledist, true);
			cv::minEnclosingCircle(sum_contours_poly[i], sum_centers[i], sum_radius[i]);
			cv::circle(bin_sum, sum_centers[i], (int)sum_radius[i], cv::Scalar(255, 0, 0), -1);
		}

		cv::Mat erode_mat = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(16, 6));
		cv::erode(bin_sum, bin_sum, erode_mat, cv::Point(-1, -1));
		cv::dilate(bin_sum, bin_sum, erode_mat, cv::Point(-1, -1));
		writer.write(erode_mat);
	}
	writer.release();
}

int main()
{
	tracking();
	return 0;
}
