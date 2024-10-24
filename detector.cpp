#include "detector.h"


Detector::Detector(const std::string& path_to_ini)
{
    get_ini_params(path_to_ini);
    obs = Observer(path_to_ini);
}


bool Detector::get_ini_params(const string& config)
{
    std::cout << "BEGIN get_ini_params" << std::endl;
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    INIReader reader(config);
    if(reader.ParseError()<0)
    {
        std::cout << "Can't load '" << config << "' !!!\n";
        return 0;
    }

    //threshold_values
    SKO_POROG = reader.GetReal("threshold_values", "SKO_POROG", -1);
    if(SKO_POROG == -1){std::cerr << "threshold_values_error!\n"; return 0;}

    DELTA_INTENS = reader.GetInteger("threshold_values", "DELTA_INTENS", -1);
    if(DELTA_INTENS == -1){std::cerr << "threshold_values_error!\n"; return 0;}

    //windows
    PUD = reader.GetInteger("windows", "PUD", -1);
    if(PUD == -1){std::cerr << "windows_error!\n"; return 0;}

    width_window_for_Ring = reader.GetInteger("windows", "width_window_for_Ring", -1);
    if(width_window_for_Ring == -1){std::cerr << "windows_error!\n"; return 0;}

    return 1;
} // -- END


std::vector<cv::Point> Detector::getGPUfilteredPoints(const cv::Mat& GrayImage){
    ContForPoints* cont = GPUCalc((unsigned char*)GrayImage.data, GrayImage.rows,
                                  GrayImage.cols, PUD, DELTA_INTENS, width_window_for_Ring, SKO_POROG);
    std::vector<cv::Point> find_points;
    find_points.reserve(*cont->counter);
    for(int i=0; i != *cont->counter; i++){
        find_points.push_back({cont->vectorX[i], cont->vectorY[i]});
    }
    free(cont);
    return find_points;
}


void Detector::processImage(const cv::Mat& GrayImage, cv::Mat& ColorImage){
    auto startGPU = std::chrono::high_resolution_clock::now();
    std::vector<cv::Point> find_points = getGPUfilteredPoints(GrayImage);
    auto endGPU = std::chrono::high_resolution_clock::now();
    auto durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(endGPU - startGPU);
    qDebug() << "Время распознавания GPU и NMS в ядре = " <<durationGPU.count();


    obs.ProcessPoints(find_points);
    obs.drawObjects(ColorImage);

    for(auto& p : find_points){
        cv::circle(ColorImage, p, 3, cv::Scalar(0,0,255));
    }
}
