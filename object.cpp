#include "object.h"
#include <QDebug>


bool Object::StateOfTracked(){
    return IsTracked;
}


void Object::CalcTraektPoint(){
    //NumPointsRez меняется до этого
    if(IsTracked){
        int Vx, Vy;
        cv::Point pred1 = getLastPoint();
        if(NextPosition == 0){ NextPosition = points.size();}
        else{ NextPosition--; } // костыль

        cv::Point pred2 = getLastPoint();
        NextPosition++;

        Vx = pred1.x - pred2.x;
        Vy = pred1.y - pred2.y;

        cv::Point TraektPoint{pred1.x + Vx, pred1.y + Vy};

        //почему-то растет
        if(NextPosition == points.size()){
            NextPosition = 0;
        }
        points[NextPosition] = TraektPoint;
        NextPosition++;

        DrawTraekt = true;
    }
}


void Object::drawTraekt(cv::Mat& image){
    for(auto&p : points){
        cv::circle(image, p, 5, cv::Scalar(0,255,0));
    }
}


int Object::getNumPointsRezArr(){
    return NumPointsRez;
}


void Object::reset(){
    IsAddWasCalled = false;
}


bool Object::CheckAddCalled(){
    if(!IsAddWasCalled){
        NumPointsRez--;
        if(CounterActualPoints>0){
            CounterActualPoints--;
        }
        return false;
    }
    else{ return true; }
}


int Object::getNumOfPoint(){
    int CurrentPos;
    if(NextPosition == points.size() || NextPosition == 0){
        CurrentPos = points.size()-1;
    }
    else{
        CurrentPos = NextPosition-1;
    }
    return CurrentPos;
}


cv::Point Object::getLastPoint(){
    return points[getNumOfPoint()];
}


bool Object::CheckForCompl(cv::Point& p){
    const int WidthOfReg = 30;
    int CurrentPos = getNumOfPoint();

    if(p.x >= points[CurrentPos].x - WidthOfReg/2 && p.x <= points[CurrentPos].x + WidthOfReg/2 &&
        p.y >= points[CurrentPos].y - WidthOfReg/2 && p.y <= points[CurrentPos].y + WidthOfReg/2){
        return true;
    }
    else{ return false; }
}


void Object::Add(cv::Point& p){
    if(NextPosition == points.size()){
        NextPosition = 0;
    }
    points[NextPosition] = p;
    NextPosition++;

    if(CounterActualPoints < 3){
        CounterActualPoints++;
    }
    if(CounterActualPoints >= 2 && IsTracked){
        NumPointsRez = 5;
    }

    if(NumPointsRez < 2 && !IsTracked){
        NumPointsRez++;
    }
    else{
        IsTracked = true;
    }

    IsAddWasCalled = true;
    DrawTraekt = false;
}
