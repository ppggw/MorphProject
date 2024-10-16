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
    int PUD;
    int NMSwindow;
    int num_of_threads_x;
    int num_of_threads_y;
    int DELTA_INTENS;
    int SKO_POROG;
    int width_of_el_RowColRingPoints;
    Observer obs;

    void NMS(std::vector<cv::Point>&, const cv::Mat&);
    std::vector<cv::Point> getGPUfilteredPoints(const cv::Mat&);
    cv::Rect getMask(int, int, cv::Point, int);
    void getCPUfilteredPoints(const cv::Mat&, std::vector<cv::Point>&);
    void FilterRowColRing(const cv::Mat&, const cv::Point&, char& result);
    void getRowColRingPoints(const cv::Mat&, std::vector<cv::Point>&);
    bool get_ini_params(const std::string&);
public:
    Detector(const std::string&);

    void processImage(const cv::Mat&, cv::Mat&);
};

#endif // DETECTOR_H
