#ifndef DETECTOR_H
#define DETECTOR_H

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <QDebug>

#include "INIReader.h"
#include "object.h"
#include "observer.h"
#include "cuda_header.h"

#include <chrono>
#include <thread>

class Detector
{
private:
    int width_window_for_Ring;
    int Width_window_for_Detect;
    int DELTA_INTENS;
    int SKO_POROG;
    int NMSwindow;
    Observer obs;

    std::vector<cv::Point> getGPUfilteredPoints(const cv::Mat&);
    bool get_ini_params(const std::string&);
    void NMS(std::vector<cv::Point>&, const cv::Mat&);

public:
    Detector(const std::string&);

    void processImage(const cv::Mat&, cv::Mat&);
};

#endif // DETECTOR_H
