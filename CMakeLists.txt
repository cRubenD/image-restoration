cmake_minimum_required(VERSION 3.14)
project(Lab6)

set(CMAKE_CXX_STANDARD 17)

# --- Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Lab6
        main.cpp
        src/IOUtils.cpp
        src/IOUtils.h
        src/Detector.cpp
        src/Detector.h
        src/Restorer.cpp
        src/Restorer.h
        src/Helper.cpp
        src/Helper.h
)

add_executable(TestApp
        src/Test.cpp
        src/Test.h
)

target_link_libraries(Lab6 ${OpenCV_LIBS})
target_link_libraries(TestApp PRIVATE ${OpenCV_LIBS})
