#include "observer.h"
#include <QDebug>

namespace{
    std::mutex m;
}


Observer::Observer(const std::string& path_to_ini)
{
    get_ini_params(path_to_ini);
}


bool Observer::get_ini_params(const string& config)
{
    std::cout << "BEGIN get_ini_params" << std::endl;
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    INIReader reader(config);
    if(reader.ParseError()<0)
    {
        std::cout << "Can't load '" << config << "' !!!\n";
        return 0;
    }

    NumOfPoints = reader.GetInteger("detector", "NElelmentInList", -1);
    if(NumOfPoints == -1){std::cerr << "detector_error!\n"; return 0;}

    NFramesForSeek = reader.GetInteger("detector", "NFramesForSeek", -1);
    if(NFramesForSeek == -1){std::cerr << "detector_error!\n"; return 0;}

    WidthOfReg = reader.GetInteger("detector", "WidthForDetector", -1);
    if(WidthOfReg == -1){std::cerr << "detector_error!\n"; return 0;}

    return 1;
} // -- END

void Observer::ProcessPoint(const std::vector<cv::Point>& previous_points, cv::Point p){
    //ищем того, кто трекается или получаем, что никто не трекается
    auto it = std::find_if(objects.begin(), objects.end(), [](Object & item)->bool{
        return item.StateOfTracked() == true;
    });

    if(it != objects.end()){
        bool IsMatch = it->CheckForCompl(p);
        if(IsMatch){
            std::lock_guard<std::mutex> lock(m);
            it->Add(p);
            return;
        }
    }
    else{
        if(!objects.empty()){
            for(Object& ob : objects){
                bool IsMatch = ob.CheckForCompl(p);
                if(IsMatch){
                    std::lock_guard<std::mutex> lock(m);
                    ob.Add(p);
                    return;
                }
            }
        }
    }
    for(auto& previous_p : previous_points){
        if(p.x >= previous_p.x - WidthOfReg/2 && p.x <= previous_p.x + WidthOfReg/2 &&
            p.y >= previous_p.y - WidthOfReg/2 && p.y <= previous_p.y + WidthOfReg/2){
            Object o(NumOfPoints, NFramesForSeek);
            o.Add(p);

            std::lock_guard<std::mutex> lock(m);
            objects.push_back(o);
            return;
        }
    }
}


void Observer::drawObjects(cv::Mat& image){
    for(Object& ob : objects){
        if(ob.getNumPointsRezArr() == 5){
            cv::circle(image, ob.getLastPoint(), 5, cv::Scalar(0,255,0));
        }
        if(ob.DrawTraekt){
            cv::circle(image, ob.getLastPoint(), 5, cv::Scalar(255,0,0));
        }
    }
    cv::putText(image, "N = " + std::to_string(objects.size()), {15, 15}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0));
}


void Observer::ProcessPoints(std::vector<cv::Point>& find_points){
    static std::vector<cv::Point> previous_points{};
    const int num_of_threads = find_points.size(); // оптимальная работа моих алгоритмов с максимальным числом потоков

    auto start = std::chrono::high_resolution_clock::now();

    if(!previous_points.empty() && !find_points.empty()){
        std::vector<std::thread> threads(num_of_threads-1);

        for(int i=0; i != num_of_threads-1; i++){
            threads[i] = std::thread(&Observer::ProcessPoint, this, std::ref(previous_points), find_points[i]);
        }
        ProcessPoint(previous_points, find_points[num_of_threads-1]);

        for(auto& entry : threads){
            entry.join();
        }
    }

    for(auto ob = objects.begin(); ob != objects.end();){
        bool isCalled = ob->CheckAddCalled();
        if(isCalled){
            ob->reset();
            ob++;
        }
        else{
            if(ob->getNumPointsRezArr() == 0){
                ob = objects.erase(ob);
            }
            else{
                ob->CalcTraektPoint();
                ob++;
            }
        }
    }


    previous_points = find_points;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    qDebug() << "Время работы трекера  = " <<duration.count();
}
