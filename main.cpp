#include <QCoreApplication>
#include <QDir>

#include "detector.h"
#include <cuda_runtime.h>
#include <stdio.h>


int main(int argc, char** argv)
{
    std::string path_to_ini = "E:/Qt/Projects/MorphProject/param_mine.ini";
    Detector det(path_to_ini);

    QString path = "E:/studvesna/";
    QDir dir(path);
    QStringList images = dir.entryList(QStringList() << "*.png", QDir::Files);

    for(auto it = images.begin(); it != images.end(); it++){
        std::string path_to_image = (path + *it).toStdString();

        cv::Mat image = cv::imread(path_to_image, cv::IMREAD_COLOR);
        cv::Mat grayFrame;
        cv::cvtColor(image, grayFrame, cv::COLOR_BGR2GRAY);

        auto start = std::chrono::high_resolution_clock::now();

        det.processImage(grayFrame, image);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        qDebug() << "Время всего = " <<duration.count() << "\n";

        cv::imshow("vid", image);
        cv::waitKey(1);
    }

    return 0;

}
