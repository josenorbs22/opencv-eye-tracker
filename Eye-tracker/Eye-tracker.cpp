﻿// Eye-tracker.cpp: define o ponto de entrada para o aplicativo.
//

#include "Eye-tracker.h"

void tracking_with_face() {
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize("./eye_predictor.dat") >> pose_model;
	
	cv::VideoCapture cap;
	cap.open("teste.mp4");
	cv::Mat frame;
	cv::VideoWriter writer("resultado.mp4",
		cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
		cap.get(cv::CAP_PROP_FPS),
		cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
		false
	);
	cv::VideoWriter writer_rot("resultado_rot.mp4",
		cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
		cap.get(cv::CAP_PROP_FPS),
		cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT))
	);
	cv::VideoWriter writer_crop("resultado_crop.mp4",
		cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
		cap.get(cv::CAP_PROP_FPS),
		cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT))
	);
	cv::VideoWriter writer_gray("resultado_gray.mp4",
		cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
		cap.get(cv::CAP_PROP_FPS),
		cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
		false
	);
	cv::VideoWriter writer_blur("resultado_blur.mp4",
		cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
		cap.get(cv::CAP_PROP_FPS),
		cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
		false
	);
	cv::VideoWriter writer_threshold("resultado_threshold.mp4",
		cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
		cap.get(cv::CAP_PROP_FPS),
		cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
		false
	);
	

	while (cap.read(frame)) {
		dlib::cv_image<dlib::bgr_pixel> cimg(frame);
		std::vector<dlib::rectangle> faces = detector(cimg);
		std::vector<dlib::full_object_detection> shapes;
		for (unsigned long i = 0; i < faces.size(); i++) {
			shapes.push_back(pose_model(cimg, faces[i]));
		}
		/*for (size_t i = 0; i < shapes.at(0).num_parts(); i++) {
			std::cout << "y: " << shapes.at(0).part(i).y() << " - x: " << shapes.at(0).part(i).x() << std::endl;
		}*/
		std::vector<cv::Point2f> left_eye_points;
		std::vector<cv::Point2f> right_eye_points;

		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(0).x(), shapes.at(0).part(0).y()));
		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(1).x(), shapes.at(0).part(1).y()));
		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(2).x(), shapes.at(0).part(2).y()));
		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(3).x(), shapes.at(0).part(3).y()));
		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(4).x(), shapes.at(0).part(4).y()));
		left_eye_points.push_back(cv::Point2f(shapes.at(0).part(5).x(), shapes.at(0).part(5).y()));

		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(6).x(), shapes.at(0).part(6).y()));
		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(7).x(), shapes.at(0).part(7).y()));
		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(8).x(), shapes.at(0).part(8).y()));
		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(9).x(), shapes.at(0).part(9).y()));
		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(10).x(), shapes.at(0).part(10).y()));
		right_eye_points.push_back(cv::Point2f(shapes.at(0).part(11).x(), shapes.at(0).part(11).y()));

		cv::RotatedRect left_eye_rect = cv::minAreaRect(left_eye_points);
		cv::RotatedRect right_eye_rect = cv::minAreaRect(right_eye_points);

		cv::Mat left_rotated;
		cv::Mat right_rotated;
		cv::Point2f left_center = left_eye_rect.center;
		cv::Point2f right_center = right_eye_rect.center;
		cv::Mat left_bounding_crop;
		cv::Mat right_bounding_crop;
		cv::Mat left_gray_crop;
		cv::Mat right_gray_crop;

		/*cv::Mat left_rot_mat = cv::getRotationMatrix2D(left_eye_rect.center, left_eye_rect.angle, 1.0);
		cv::Mat right_rot_mat = cv::getRotationMatrix2D(right_eye_rect.center, right_eye_rect.angle, 1.0);*/
		cv::Mat left_rot_mat = cv::getRotationMatrix2D(left_eye_rect.center, 0, 1.0);
		cv::Mat right_rot_mat = cv::getRotationMatrix2D(right_eye_rect.center, 0, 1.0);
		cv::warpAffine(frame, left_rotated, left_rot_mat, frame.size());
		cv::warpAffine(frame, right_rotated, right_rot_mat, frame.size());

		//Criar vídeo só com os olhos rotacionados
		cv::Mat rot = left_rotated;
		writer_rot.write(rot);
		//-------------------------

		cv::Size left_size(
			std::abs(shapes.at(0).part(3).x() - shapes.at(0).part(0).x()),
			std::abs(shapes.at(0).part(5).y() - shapes.at(0).part(2).y())
		);
		cv::Size right_size(
			std::abs(shapes.at(0).part(8).x() - shapes.at(0).part(6).x()),
			std::abs(shapes.at(0).part(10).y() - shapes.at(0).part(8).y())
		);
		cv::Size dimensions(left_size.width > right_size.width ? left_size.width : right_size.width,
			left_size.height > right_size.height ? left_size.height : right_size.height);
		cv::getRectSubPix(left_rotated, dimensions, left_center, left_bounding_crop);
		cv::getRectSubPix(right_rotated, dimensions, right_center, right_bounding_crop);

		//Criar vídeo só com os olhos cortados
		cv::Mat crop = left_bounding_crop;
		cv::resize(crop, crop, frame.size());
		writer_crop.write(crop);
		//-------------------------

		cv::cvtColor(left_bounding_crop, left_gray_crop, cv::COLOR_BGR2GRAY);
		cv::cvtColor(right_bounding_crop, right_gray_crop, cv::COLOR_BGR2GRAY);

		//Criar vídeo cinza
		cv::Mat gray = left_gray_crop;
		cv::resize(gray, gray, frame.size());
		writer_gray.write(gray);
		//-------------------------

		cv::GaussianBlur(left_gray_crop, left_gray_crop, cv::Size(5, 5), 0);
		cv::GaussianBlur(right_gray_crop, right_gray_crop, cv::Size(5, 5), 0);

		//Criar vídeo desfocado
		cv::Mat blur = left_gray_crop;
		cv::resize(blur, blur, frame.size());
		writer_blur.write(blur);
		//-------------------------

		for (int y = 0; y < left_gray_crop.rows; y++) {
			for (int x = 0; x < left_gray_crop.cols; x++) {
				for (int c = 0; c < left_gray_crop.channels(); c++) {
					left_gray_crop.at<uchar>(y, x) =
						/*cv::saturate_cast<uchar>(
							(alpha_slider / 100.0) * left_gray_crop.at<cv::Vec3b>(y, x)[c] + betha_slider
						);*/
						cv::saturate_cast<uchar>(
							left_gray_crop.at<uchar>(y, x)
						);
				}

			}
		}

		for (int y = 0; y < right_gray_crop.rows; y++) {
			for (int x = 0; x < right_gray_crop.cols; x++) {
				for (int c = 0; c < right_gray_crop.channels(); c++) {
					right_gray_crop.at<uchar>(y, x) =
						/*cv::saturate_cast<uchar>(
							(alpha_slider / 100.0) * right_gray_crop.at<cv::Vec3b>(y, x)[c] + betha_slider
						);*/
						cv::saturate_cast<uchar>(
							right_gray_crop.at<uchar>(y, x)
						);
				}
			}
		}

		cv::Mat left_bin_crop;
		cv::Mat right_bin_crop;

		/*cv::threshold(left_gray_crop, left_bin_crop, left_bin_slider, 255, cv::THRESH_BINARY_INV);
		cv::threshold(right_gray_crop, right_bin_crop, right_bin_slider, 255, cv::THRESH_BINARY_INV);*/

		cv::threshold(left_gray_crop, left_bin_crop, 50, 255, cv::THRESH_BINARY_INV);
		cv::threshold(right_gray_crop, right_bin_crop, 50, 255, cv::THRESH_BINARY_INV);

		//Criar vídeo apenas com pontos brancos e pretos
		cv::Mat threshold = left_bin_crop;
		cv::resize(threshold, threshold, frame.size());
		writer_threshold.write(threshold);
		//-------------------------

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

		
		//cv::cvtColor(bin_sum, bin_sum, cv::COLOR_GRAY2BGR);
		//bin_sum.copySize(frame);
		cv::resize(bin_sum, bin_sum, frame.size());

		writer.write(bin_sum);
	}
	writer.release();
	writer_rot.release();
	writer_crop.release();
	writer_gray.release();
	writer_blur.release();
	writer_threshold.release();
	cap.release();
}

void tracking_eyes_only() {
	cv::CascadeClassifier eyeCascade;
	if (!eyeCascade.load("haarcascade_eye_tree_eyeglasses.xml")) {
		std::cerr << "Erro ao carregar Haar Cascade de olhos!" << std::endl;
		return;
	}
	//cv::VideoCapture cap(0);
	cv::VideoCapture cap;
	cap.open("teste.mp4");
	cv::Mat frame, gray;
	cv::VideoWriter writer("resultado.mp4",
		cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
		cap.get(cv::CAP_PROP_FPS),
		cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
		false
	);
	cv::VideoWriter writer_crop("resultado_crop.mp4",
		cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
		cap.get(cv::CAP_PROP_FPS),
		cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT))
	);
	cv::VideoWriter writer_bin("resultado_bin.mp4",
		cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
		cap.get(cv::CAP_PROP_FPS),
		cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
		false
	);
	while (cap.read(frame)) {
		int width = 0, height = 0, i = 0;

		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray, gray);
		std::vector<cv::Rect> eyes;
		eyeCascade.detectMultiScale(gray, eyes, 1.2, 6, 0, cv::Size(150, 150));
		std::vector<cv::Mat> bin_eyes(eyes.size());

		cv::Mat bin_sum;

		for (auto& eye : eyes) {
			width = (eye.width > width ? eye.width : width);
			height = (eye.height > height ? eye.height : height);
			bin_sum = cv::Mat::zeros(height, width, CV_8UC1);
		}

		if (bin_eyes.size() > 0) bin_eyes.at(0) = cv::Mat::zeros(height, width, CV_8UC1);
		if(bin_eyes.size() > 1) bin_eyes.at(1) = cv::Mat::zeros(height, width, CV_8UC1);

		for (auto& eye : eyes) {
			
			cv::Mat rotated;
			cv::Mat bounding_crop;
			cv::Mat gray_crop;
			cv::Mat bin_crop;
			std::vector<cv::Point2f> eye_points;

			eye_points.push_back(cv::Point2f(eye.x, eye.y));
			eye_points.push_back(cv::Point2f(eye.x + eye.width, eye.y));	
			eye_points.push_back(cv::Point2f(eye.x, eye.y + eye.height));
			eye_points.push_back(cv::Point2f(eye.x + eye.width, eye.y + eye.height));
			cv::RotatedRect eye_rect = cv::minAreaRect(eye_points);
			cv::Mat rot_mat = cv::getRotationMatrix2D(eye_rect.center, 0, 1.0);
			cv::warpAffine(frame, rotated, rot_mat, frame.size());
			
			cv::getRectSubPix(rotated, cv::Size(eye.width, eye.height), eye_rect.center, bounding_crop);

			//Criar vídeo só com os olhos cortados
			cv::Mat crop = bounding_crop;
			cv::resize(crop, crop, frame.size());
			//writer_crop.write(crop);
			//-------------------------

			cv::resize(bounding_crop, bounding_crop, cv::Size(width, height));

			cv::cvtColor(bounding_crop, gray_crop, cv::COLOR_BGR2GRAY);

			cv::GaussianBlur(gray_crop, gray_crop, cv::Size(5, 5), 0);

			for (int y = 0; y < gray_crop.rows; y++) {
				for (int x = 0; x < gray_crop.cols; x++) {
					for (int c = 0; c < gray_crop.channels(); c++) {
						gray_crop.at<uchar>(y, x) =
							/*cv::saturate_cast<uchar>(
								(alpha_slider / 100.0) * left_gray_crop.at<cv::Vec3b>(y, x)[c] + betha_slider
							);*/
							cv::saturate_cast<uchar>(
								gray_crop.at<uchar>(y, x)
							);
					}

				}
			}

			cv::threshold(gray_crop, bin_crop, 50, 255, cv::THRESH_BINARY_INV);
			bin_eyes.at(i) = bin_crop;

			if (bin_eyes.size() > 1) cv::bitwise_and(bin_eyes.at(0), bin_eyes.at(1), bin_sum);
			i++;
		}
		//Criar vídeo binário somado
		cv::Mat bin = bin_sum;
		if(width > 0 && height > 0) cv::resize(bin, bin, frame.size());
		writer_bin.write(bin);
		//-------------------------

		std::vector<std::vector<cv::Point>> sum_contours;
		std::vector<cv::Vec4i> sum_hierarchy;
		std::vector<std::vector<cv::Point>> sum_contours_poly;
		std::vector<cv::Point2f> sum_centers;
		std::vector<float> sum_radius;

		if(!eyes.empty())
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

		if (!eyes.empty()) {
			cv::erode(bin_sum, bin_sum, erode_mat, cv::Point(-1, -1));
			cv::dilate(bin_sum, bin_sum, erode_mat, cv::Point(-1, -1));
			cv::resize(bin_sum, bin_sum, frame.size());
			//cv::imshow("Detecção de Olhos", bin_sum);
			//if (cv::waitKey(10) == 27) break;
			writer.write(bin_sum);
		}
		//else {
			//cv::imshow("Detecção de Olhos", frame);
		//}

		
	}
	writer.release();
	writer_crop.release();
	writer_bin.release();
	cap.release();
}

int main()
{
	//tracking_with_face();
	tracking_eyes_only();
	cv::destroyAllWindows();
	return 0;
}
