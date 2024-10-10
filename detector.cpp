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


cv::Mat Detector::getGPUfilteredRingImage(const cv::Mat& GrayImage){
    DataAfterGPU = GPUCalc((unsigned char*)GrayImage.data, GrayImage.rows, GrayImage.cols, PUD, DELTA_INTENS);

    cv::Mat GPUFiltredImage = cv::Mat(
                                GrayImage.rows,
                                GrayImage.cols,
                                CV_8UC1,
                                DataAfterGPU
                        );
//    cv::imshow("f", GPUFiltredImage);
//    cv::waitKey(0);

    return GPUFiltredImage;
}


void Detector::NMS(std::vector<cv::Point>& find_points, const cv::Mat& image, const int offset_X, const int offset_Y){
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
                if(image.at<uchar>({i->p.x - offset_X, i->p.y - offset_Y}) >=
                        image.at<uchar>(j->p.x - offset_X, j->p.y - offset_Y)){
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


void Detector::FilterRing(const cv::Mat& GrayImage, const int offset_X, const int offset_Y, std::vector<cv::Point>& find_points){
    find_points.reserve(100);
    for(int i=0; i!= GrayImage.rows; i++){
        for(int j=0; j!= GrayImage.cols; j++){
            if(GrayImage.at<uchar>(i, j) !=0){
               find_points.push_back({j + offset_X, i + offset_Y});
            }
        }
    }

//    if(find_points.size() != 0){
//        NMS(find_points, GrayImage, offset_X, offset_Y);
//    }
}


void Detector::FilterRowColRing(const cv::Mat& GrayImage, const cv::Point& coor_of_object, char& result){
    if(coor_of_object.x < width_window_for_Ring || GrayImage.cols - coor_of_object.x < width_window_for_Ring ||
       coor_of_object.y < width_window_for_Ring || GrayImage.rows - coor_of_object.y < width_window_for_Ring){
        result = 1;
        return;
    }

//    int SumPixInCol = 0;
//    for(int i=0; i != GrayImage.cols; i++){
//        if( i != coor_of_object.x){ SumPixInCol += GrayImage.at<uchar>(coor_of_object.y, i); }
//    }
//    float M = (float)SumPixInCol/ (GrayImage.cols-1);
//    float SumForSKOCol = 0;
//    for(int i=0; i != GrayImage.cols; i++){
//        if( i != coor_of_object.x){
//            SumForSKOCol += std::pow((GrayImage.at<uchar>(coor_of_object.y, i) - M), 2);
//        }
//    }
//    float SKO_Col = std::sqrt(SumForSKOCol/ (GrayImage.cols-1) );
//    if( (GrayImage.at<uchar>(coor_of_object.y, coor_of_object.x) < M + (SKO_POROG-1)*SKO_Col)){
//        result = 1;
//        return;
//    }

//    int SumPixInRow = 0;
//    for(int i=0; i != GrayImage.rows; i++){
//        if( i != coor_of_object.y){ SumPixInRow += GrayImage.at<uchar>(i, coor_of_object.x); }
//    }
//    float M_Row = (float)SumPixInRow/(GrayImage.cols-1);
//    float SumForSKORow = 0;
//    for(int i=0; i != GrayImage.rows; i++){
//        if( i != coor_of_object.y){
//            SumForSKORow += std::pow((GrayImage.at<uchar>(i, coor_of_object.x) - M_Row), 2);
//        }
//    }
//    float SKO_Row = std::sqrt(SumForSKORow/ (GrayImage.cols - 1) );
//    if( (GrayImage.at<uchar>(coor_of_object.y, coor_of_object.x) < M_Row + (SKO_POROG-1)*SKO_Row)){
//        result = 1;
//        return;
//    }

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


std::vector<cv::Point> Detector::getCPUfilteredPoints(const cv::Mat& GPUfilteredImage){
    static const int num_of_threads = num_of_threads_x * num_of_threads_y;
    static const int width_of_el_x = GPUfilteredImage.cols/num_of_threads_x;
    static const int width_of_el_y = GPUfilteredImage.rows/num_of_threads_y;

    std::vector<std::thread> threads(num_of_threads-1);
    std::vector <std::vector<cv::Point> > results_for_each_thread(num_of_threads);

    int counter = 0;
    for(int i=0; i != num_of_threads_x; i++){
        for(int j=0; j!=num_of_threads_y; j++){
            if(counter == num_of_threads-1){ break; }
            cv::Rect mask(i*width_of_el_x, j*width_of_el_y, width_of_el_x, width_of_el_y);
            cv::Mat MaskedGPUfilteredImage = GPUfilteredImage(mask);
            threads[counter] = std::thread(&Detector::FilterRing, this, MaskedGPUfilteredImage, width_of_el_x*i, width_of_el_y*j,
                                           std::ref(results_for_each_thread[counter]));
            counter++;
        }
    }
    cv::Mat LastMaskedGPUfilteredImage = GPUfilteredImage(cv::Rect{width_of_el_x*(num_of_threads_x-1), width_of_el_y*(num_of_threads_y-1),
                                            width_of_el_x, width_of_el_y});
    FilterRing(LastMaskedGPUfilteredImage, width_of_el_x*(num_of_threads_x-1), width_of_el_y*(num_of_threads_y-1),
                  results_for_each_thread[num_of_threads-1]);

    for(auto& entry : threads){
        entry.join();
    }

    //слейка всех точек в один вектор
    std::vector<cv::Point> find_points;
    for(int i=0; i != results_for_each_thread.size(); i++){
        for(auto it=results_for_each_thread[i].begin(); it!=results_for_each_thread[i].end(); it++){
            find_points.push_back(*it);
        }
    }

    return find_points;
}

std::vector<cv::Point> Detector::getRingPoints(const cv::Mat& GrayImage){
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat GPUfilteredImage = getGPUfilteredRingImage(GrayImage);
    std::vector<cv::Point> find_points = getCPUfilteredPoints(GPUfilteredImage);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    qDebug() << "Время распознавания по кольцу + NMS = " <<duration.count();

    return find_points;
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


void Detector::getRowColRingPoints(const cv::Mat& GrayImage, std::vector<cv::Point>& points){
    auto start = std::chrono::high_resolution_clock::now();

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

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    qDebug() << "Время распознавания по кольцу, строке и столбцу = " <<duration.count();
}


void Detector::clearDataAfterGPU(){
    free(DataAfterGPU);
}


void Detector::processImage(const cv::Mat& GrayImage, cv::Mat& ColorImage){
    // в FilterRowColRing у RowCol Sko= sko_porog-2
    std::vector<cv::Point> find_points = getRingPoints(GrayImage);
    getRowColRingPoints(GrayImage, find_points);

//    obs.ProcessPoints(find_points);
//    obs.drawObjects(ColorImage);

    for(auto& p : find_points){
        cv::circle(ColorImage, p, 3, cv::Scalar(0,0,255));
    }
    clearDataAfterGPU();
}
