#ifndef OBSERVER_H
#define OBSERVER_H

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include "mutex"

#include "object.h"
#include "INIReader.h"

class Observer
{
private:
    int WidthOfReg;
    int NumOfPoints;
    int NFramesForSeek;
    std::vector<Object> objects;

    bool get_ini_params(const std::string&);
    void ProcessPoint(const std::vector<cv::Point>&, cv::Point);
public:
    Observer(){};
    Observer(const std::string&);

    void ProcessPoints(std::vector<cv::Point>&);
    void drawObjects(cv::Mat&);
};

#endif // OBSERVER_H
