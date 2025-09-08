# CMakeList.txt: arquivo de projeto de CMake de nível superior, fazer configuração global
# e inclua subprojetos aqui.
#
cmake_minimum_required (VERSION 3.8)


# Habilite a Recarga Dinâmica para compiladores MSVC, se houver suporte.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project (eye-tracker)

find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v19.24
)
FetchContent_MakeAvailable(dlib)

add_executable(eye-tracker "./Eye-tracker/Eye-tracker.cpp" "./Eye-tracker/Eye-tracker.h")

target_include_directories(eye-tracker PUBLIC ${OpenCV_INCLUDE_DIRS} "D:/Documentos/TCC/Eye-tracker/out/build/x64-debug/_deps/dlib-src")

target_link_libraries(eye-tracker PUBLIC ${OpenCV_LIBS} dlib)