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

    //thread
    num_of_threads_x = reader.GetInteger("thread_", "num_of_threads_x", -1);
    if(num_of_threads_x == -1){std::cerr << "thread_error!\n"; return 0;}

    num_of_threads_y = reader.GetInteger("thread_", "num_of_threads_y", -1);
    if(num_of_threads_y == -1){std::cerr << "thread_error!\n"; return 0;}

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

    width_of_el_RowColRingPoints = reader.GetInteger("windows", "width_of_el_RowColRingPoints", -1);
    if(width_of_el_RowColRingPoints == -1){std::cerr << "windows_error!\n"; return 0;}

    width_window_for_Ring = reader.GetInteger("windows", "width_window_for_Ring", -1);
    if(width_window_for_Ring == -1){std::cerr << "windows_error!\n"; return 0;}

    return 1;
} // -- END


std::vector<cv::Point> Detector::getGPUfilteredPoints(const cv::Mat& GrayImage){
    ContForPoints* cont = GPUCalc((unsigned char*)GrayImage.data, GrayImage.rows, GrayImage.cols, PUD, DELTA_INTENS);
    std::vector<cv::Point> find_points;
    find_points.reserve(*cont->counter);
    for(int i=0; i != *cont->counter; i++){
        find_points.push_back({cont->vectorX[i], cont->vectorY[i]});
    }
    free(cont);
    return find_points;
}


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


void Detector::FilterRowColRing(const cv::Mat& GrayImage, const cv::Point& coor_of_object, char& result){
    if(coor_of_object.x < width_window_for_Ring || GrayImage.cols - coor_of_object.x < width_window_for_Ring ||
       coor_of_object.y < width_window_for_Ring || GrayImage.rows - coor_of_object.y < width_window_for_Ring){
        result = 1;
        return;
    }

    float M_Ring = 0;
    for(int l = -width_window_for_Ring/2+1; l != width_window_for_Ring/2; l++){
        M_Ring += GrayImage.at<uchar>(coor_of_object.y - width_window_for_Ring/2, coor_of_object.x + l);
        M_Ring += GrayImage.at<uchar>(coor_of_object.y + width_window_for_Ring/2, coor_of_object.x + l);
    }

    for(int l = -width_window_for_Ring/2+1; l != width_window_for_Ring/2; l++){
        M_Ring += GrayImage.at<uchar>(coor_of_object.y + l, coor_of_object.x - width_window_for_Ring/2);
        M_Ring += GrayImage.at<uchar>(coor_of_object.y + l, coor_of_object.x - width_window_for_Ring/2);
    }

    M_Ring = M_Ring/( (2*width_window_for_Ring + 2) + (2*width_window_for_Ring - 2));

    float SumForSKORing = 0;
    for(int l = -width_window_for_Ring/2; l != width_window_for_Ring/2 + 1; l++){
        SumForSKORing += std::pow( (GrayImage.at<uchar>(coor_of_object.y - width_window_for_Ring/2, coor_of_object.x + l) - M_Ring), 2);
        SumForSKORing += std::pow( (GrayImage.at<uchar>(coor_of_object.y + width_window_for_Ring/2, coor_of_object.x + l) - M_Ring), 2);
    }

    for(int l = -width_window_for_Ring/2 + 1; l != width_window_for_Ring/2; l++){
        SumForSKORing += std::pow((GrayImage.at<uchar>(coor_of_object.y + l, coor_of_object.x - width_window_for_Ring/2) - M_Ring), 2);
        SumForSKORing += std::pow((GrayImage.at<uchar>(coor_of_object.y + l, coor_of_object.x + width_window_for_Ring/2) - M_Ring), 2);
    }

    SumForSKORing /= (2*width_window_for_Ring + 2) + (2*width_window_for_Ring - 2);
    SumForSKORing = std::sqrt(SumForSKORing);

    if(GrayImage.at<uchar>(coor_of_object.y, coor_of_object.x) <=  M_Ring + SKO_POROG * SumForSKORing){
        result = 1;
        return;
    }

    result = 0;
}


cv::Rect Detector::getMask(int cols, int rows, cv::Point p, int width){
    cv::Rect mask;
    if(p.x < width/2){
        mask.x = 0;
    }
    if(p.x > cols - width/2){
        mask.x = cols-width;
    }
    if(p.x >= width && p.x <= cols - width/2){
        mask.x = p.x - width/2;
    }

    if(p.y < width/2){
        mask.y = 0;
    }
    if(p.y > rows - width/2){
        mask.y = rows - width;
    }
    if(p.y >= width && p.y <= rows - width/2){
        mask.y = p.y - width/2;
    }
    mask.width = width;
    mask.height = width;

    return mask;
}


void Detector::getCPUfilteredPoints(const cv::Mat& GrayImage, std::vector<cv::Point>& points){
    const int num_of_threads = points.size(); // оптимальная работа моих алгоритмов с максимальным числом потоков
    if(points.size() != 0){
        std::vector<std::thread> threads;
        std::vector<char> results_for_each_thread(num_of_threads); // с bool проблемы

        for(int i=0; i != num_of_threads-1; i++){
            cv::Rect mask = getMask(GrayImage.cols, GrayImage.rows, points[i], width_of_el_RowColRingPoints);
            cv::Mat masked_gray_image = GrayImage(mask);
            cv::Point norm_coor{points[i].x - mask.x, points[i].y - mask.y};
            threads.push_back(std::thread(&Detector::FilterRowColRing, this, masked_gray_image, norm_coor, std::ref(results_for_each_thread[i])));
        }
            cv::Rect mask = getMask(GrayImage.cols, GrayImage.rows, points[num_of_threads-1], width_of_el_RowColRingPoints);
            cv::Mat masked_gray_image = GrayImage(mask);
            cv::Point norm_coor{points[num_of_threads-1].x - mask.x, points[num_of_threads-1].y - mask.y};
            FilterRowColRing(masked_gray_image, norm_coor, results_for_each_thread[num_of_threads-1]);

        for(auto& entry : threads){
            entry.join();
        }

        //слейка всех точек в один вектор
        int counter = 0;
        for(auto it = points.begin(); it != points.end();){
            if(results_for_each_thread[counter] != 0){
                it = points.erase(it);
            }
            else{ it++; }
            counter++;
        }
    }
}


void Detector::processImage(const cv::Mat& GrayImage, cv::Mat& ColorImage){
//    ContForPoints* cont = GPUCalc((unsigned char*)GrayImage.data, GrayImage.rows, GrayImage.cols, PUD, DELTA_INTENS);
//    free(cont);

    auto startGPU = std::chrono::high_resolution_clock::now();
    std::vector<cv::Point> find_points = getGPUfilteredPoints(GrayImage);
    auto endGPU = std::chrono::high_resolution_clock::now();
    auto durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(endGPU - startGPU);
    qDebug() << "Время распознавания GPU и NMS в ядре = " <<durationGPU.count();


    //де-факто не нужно, тк НМС 32х32 уже делается в ядре
    auto startNMS = std::chrono::high_resolution_clock::now();
    NMS(find_points, GrayImage);
    auto endNMS= std::chrono::high_resolution_clock::now();
    auto durationNMS = std::chrono::duration_cast<std::chrono::milliseconds>(endNMS - startNMS);
    qDebug() << "Время NMS на CPU = " <<durationNMS.count();

    auto startCPUFilter = std::chrono::high_resolution_clock::now();
    getCPUfilteredPoints(GrayImage, find_points);
    auto endCPUFilter = std::chrono::high_resolution_clock::now();
    auto durationCPUFilter = std::chrono::duration_cast<std::chrono::milliseconds>(endCPUFilter - startCPUFilter);
    qDebug() << "Время фильтрации по кольцу = " <<durationCPUFilter.count();

    obs.ProcessPoints(find_points);
    obs.drawObjects(ColorImage);

    for(auto& p : find_points){
        cv::circle(ColorImage, p, 3, cv::Scalar(0,0,255));
    }
}
