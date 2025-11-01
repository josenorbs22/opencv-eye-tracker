// Eye-tracker.h: arquivo de inclusão para arquivos de inclusão padrão do sistema,
// ou arquivos de inclusão específicos a um projeto.

#pragma once

#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/shape_predictor.h>
//#include <dlib/opencv/cv_image.h>

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/timestamp.h"
//#include "matplotlibcpp.h"
#include "gnuplot-iostream.h"




constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "multi_face_landmarks";


absl::Status tracking_mediapipe(std::string file_name);


//void tracking_with_face();
//void tracking_haar();

// TODO: Referencie os cabeçalhos adicionais de que seu programa precisa aqui.
