#include <QCoreApplication>
#include <QDir>

#include "detector.h"
#include <cuda_runtime.h>
#include <stdio.h>


int main(int argc, char** argv)
{
    std::string path_to_ini = "E:/Qt/Projects/MorphProject/param_mine.ini";
    Detector det(path_to_ini);

//    QString path = "E:/Qt/Projects/MorpProjectAA/for_copy/";
    QString path = "E:/studvesna/";
    QDir dir(path);
    QStringList images = dir.entryList(QStringList() << "*.PNG", QDir::Files);

    for(auto it = images.begin(); it != images.end(); it++){
        std::string path_to_image = (path + *it).toStdString();

        cv::Mat image = cv::imread(path_to_image, cv::IMREAD_COLOR);
        cv::Mat grayFrame;
        cv::cvtColor(image, grayFrame, cv::COLOR_BGR2GRAY);

        det.processImage(grayFrame, image);

        cv::imshow("vid", image);
        cv::waitKey(1);
    }

    return 0;

}
