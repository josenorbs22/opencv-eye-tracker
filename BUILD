# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

licenses(["notice"])

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/face_mesh:face_mesh_tflite
# RUN:
# bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_tflite \
#   --calculator_graph_config_file=<.../face_mesh_desktop.pbtxt> \
#   --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>


cc_binary(
    name = "eye_tracking",
    srcs = ["Eye-tracker.cpp",
            "Eye-tracker.h",
            #"matplotlibcpp.h"
            "gnuplot-iostream.h"
            ],
    data = ["assets/face_mesh_desktop.pbtxt",
            #"assets/face_mesh_desktop_live.pbtxt",
            #"assets/face_mesh_desktop_live_gpu.pbtxt",
            "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
            "//mediapipe/modules/face_detection:face_detection_short_range.tflite"
            ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework:timestamp",
        "//mediapipe/graphs/face_mesh:desktop_calculators",
        #"//mediapipe/graphs/face_mesh:desktop_live_calculators",
        #"//mediapipe/graphs/face_mesh:desktop_live_gpu_calculators",
        #"@pybind11//:pybind11",
    ],
    copts = [
        #"-Ithird_party/matplotlibcpp",

        #"-I/usr/include/python3.12",

        "-I/usr/include",

        "-Ithird_party/gnuplot-iostream",
    ],
    linkopts = ["-lboost_iostreams", "-lboost_serialization"],
)
