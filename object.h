#ifndef OBJECT_H
#define OBJECT_H

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>

class Object
{
private:
    std::vector<cv::Point> points;
    int NextPosition = 0;
    int NumPointsRez = 0;
    int NFramesForSeek = 0;
    int CounterActualPoints = 0;
    bool IsAddWasCalled = false;
    bool IsTracked = false;

    int getNumOfPoint();
public:
    Object(int N, int NFramesForSeek_) : points(N), NFramesForSeek(NFramesForSeek_){};

    bool DrawTraekt = false;

    cv::Point getLastPoint();
    bool CheckForCompl(cv::Point&);
    void Add(cv::Point&);
    bool CheckAddCalled();
    void reset();
    int getNumPointsRezArr();
    void drawTraekt(cv::Mat&);
    void CalcTraektPoint();
    bool StateOfTracked();
};

#endif // OBJECT_H
