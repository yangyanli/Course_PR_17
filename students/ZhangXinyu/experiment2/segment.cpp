#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;


int main()
{
    
    Mat src = imread("/Users/Lagrant/Downloads/mnist/img/input.jpg",1);
    Mat out ;
    cvtColor(src,out,COLOR_BGR2GRAY);
    
    threshold(out,out, 48, 255, CV_THRESH_BINARY_INV);
    Mat prp = out.clone() ;
    
    vector<vector<Point>>contours;
    findContours(out,contours, RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    vector<std::vector<cv::Point>>::const_iterator itc =  contours.begin();
    while (itc != contours.end()) {
        if (itc->size() < 10 || itc->size() > 1000)
            itc =  contours.erase(itc);
        else
            ++itc;
    }
    char file_namew[255];
    sprintf(file_namew,"/Users/Lagrant/Downloads/mnist/img/");
    char rect_number[255];
    char file[255];
    Mat pic;
    for (int i = 0;i<contours.size();i++)
    {
        sprintf(file,"%s%d",file_namew,i);
        sprintf(rect_number,"%s%d.png",file_namew,i);
        Rect r  = boundingRect(Mat(contours[i]));
        Point tl  =  r.tl();
        if(tl.x>5 && tl.y>5)
            r  =  r+Point(-5,-5);
        Point br  =  r.br();
        int dy  =  prp.rows - br.y;
        int dx  =  prp.cols - br.x;
        if(dy>10 && dx>10)
            r  =  r+Size(10,10);
        pic = prp(r);
         resize(pic,pic,Size(28,28));
        imwrite(rect_number,pic);
    }
    return 0;
}
