#include<iostream>
#include<fstream>
#include<opencv.hpp>
#include"cnn.hpp"

using namespace std;
using namespace cv;

int main()
{
	init();
	bool flag = readModelFile("cnnmodel.dat");
	if (!flag)
	{
		cout << "read cnnmodel.dat error!" << endl;
		return -1;
	}

	int width = 32, height = 32;
	int keepRunning = 1;
	string imgPath = "";
	while (keepRunning)
	{
		cout << "Input file name with extension part (input -1 to exit):" << endl;
		cin >> imgPath;

		if (imgPath == "-1")
		{
			keepRunning = 0;
			break;
		}
		Mat src = imread(imgPath);
		if (src.data == NULL)
		{
			cout << "Wrong file name! Please input correct file name." << endl;
			continue;
		}
		Mat imgGray;
		imgGray.create(src.rows, src.cols, CV_8UC1);
		cvtColor(src, imgGray, CV_BGR2GRAY);
		threshold(imgGray, imgGray, 100, 255, CV_THRESH_BINARY_INV);
		namedWindow("thresholdImg");
		imshow("thresholdImg", imgGray);
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(imgGray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		
		/*Mat img1;
		Mat img2;
		Mat img3;
		Mat img4;
		Mat img5;*/

		for (int i = 0; i<contours.size(); i++)
		{
			Rect rect = boundingRect(Mat(contours[i]));

			int r = rect.width, c = rect.height;
			if (r < 5 || c < 5) continue; //If the sub picture is too small, skip it.
			int deltax = 0, deltay = 0;
			if (r < c)
			{
				deltax = c - r;
			}
			else if (r > c)
			{
				deltay = r - c;
			}
			int boundary = rect.width + deltax; //set blank boundary
			boundary /= 4;
			Mat tmp(rect.width + deltax + boundary * 2, rect.height + deltay + boundary * 2, CV_8UC1, Scalar::all(0)); //set background
			Rect tmpRect(deltax / 2 + boundary, deltay / 2 + boundary, rect.width, rect.height);
			imgGray(rect).copyTo(tmp(tmpRect));

			resize(tmp, tmp, Size(width, height));
			int ret = predict(tmp.data, width, height);

			Point tp;
			tp.x = rect.x;
			tp.y = rect.y;
			putText(src, to_string(ret), tp, FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255), 1, 8, false);

			rectangle(src, rect.tl(), rect.br(), Scalar(255, 0, 0));
			/*Mat roi = src(rect);
			switch (i)
			{
			case 1:
				roi.convertTo(img1, roi.type());
			case 4:
				roi.convertTo(img2, roi.type());
			case 5:
				roi.convertTo(img3, roi.type());
			case 7:
				roi.convertTo(img4, roi.type());
			case 8:
				roi.convertTo(img5, roi.type());
			}*/

			/*Range rx, ry;
			rx.start = rect.x;
			ry.start = rect.y;
			rx.end = rect.x + rect.width;
			ry.end = rect.y + rect.height;
			Mat cont = Mat(imgGray, rx, ry);*/
			/*Mat tmp = cont(Rect(rect.x, rect.y, rect.width, rect.height));*/
			
			/*Mat tmp;
			tmp.create(cont.rows, cont.cols, CV_8UC1);
			cvtColor(cont, tmp, CV_BGR2GRAY);
			subtract(tmp, cont, tmp);*/
		}
		/*imshow("img1", img1);
		imshow("img2", img2);
		imshow("img3", img3);
		imshow("img4", img4);
		imshow("img5", img5);*/
		imshow("contoursImg", src);
		waitKey(0);
		//src.release();
		//imgGray.release();
		contours.clear();
		hierarchy.clear();
		/*Mat tmp(src.rows, src.cols, CV_8UC1, Scalar::all(255));
		subtract(tmp, src, tmp);
		resize(tmp, tmp, Size(width, height));
		int ret = predict(tmp.data, width, height);
		cout << "The predict digit is: " << ret << endl;*/
	}
	release();

	return 0;
}