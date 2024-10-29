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

    NMSwindow = reader.GetInteger("windows", "NMSwindow", -1);
    if(NMSwindow == -1){std::cerr << "windows_error!\n"; return 0;}

    width_window_for_Ring = reader.GetInteger("windows", "width_window_for_Ring", -1);
    if(width_window_for_Ring == -1){std::cerr << "windows_error!\n"; return 0;}

    return 1;
} // -- END


void Detector::NMS(std::vector<cv::Point>& find_points, const cv::Mat& image){
    using cor_it = std::vector<cv::Point>::iterator;
    std::vector<cv::Point> for_delete;

    struct Local_Point{
        cv::Point p;
        bool down; // если true - точку не используем
    };

    std::vector<Local_Point> copy;
    for(cor_it i = find_points.begin(); i != find_points.end(); ++i){
        copy.push_back({*i, false});
    }

    for(std::vector<Local_Point>::iterator i = copy.begin(); i != copy.end(); ++i){
        for(std::vector<Local_Point>::iterator j = copy.begin(); j != copy.end(); ++j){
        if(j->down == false && i->down == false && i->p != j->p){
            if( (j->p.x > i->p.x-NMSwindow && j->p.x < i->p.x+NMSwindow) && (j->p.y > i->p.y-NMSwindow && j->p.y < i->p.y+NMSwindow) ){
                if(image.at<uchar>(i->p) >= image.at<uchar>(j->p)){
                        for_delete.push_back(j->p);
                        j->down = true;
                    }
                }
            }
        }
    }

    for(cv::Point& point : for_delete){
        for(cor_it i = find_points.begin(); i != find_points.end(); i++){
            if(*i == point){
                i = find_points.erase(i);
                break;
            }
        }
    }
}


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

    auto startNMS = std::chrono::high_resolution_clock::now();
    NMS(find_points, GrayImage);
    auto endNMS = std::chrono::high_resolution_clock::now();
    auto durationNMS = std::chrono::duration_cast<std::chrono::milliseconds>(endNMS - startNMS);
//    qDebug() << "Время работы NMS на CPU = " <<durationNMS.count();

    obs.ProcessPoints(find_points);
    obs.drawObjects(ColorImage);

    for(auto& p : find_points){
        cv::circle(ColorImage, p, 3, cv::Scalar(0,0,255));
    }
}
