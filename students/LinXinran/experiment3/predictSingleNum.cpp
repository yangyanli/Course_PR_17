#include<iostream>
#include<fstream>
#include<opencv.hpp>
#include"knn.hpp"

using namespace std;
using namespace cv;

int main()
{
	init();
	int keepRunning = 1;
	int k, ret;
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
		Mat tmp(imgGray.rows, imgGray.cols, CV_8UC1, Scalar::all(255));
		subtract(tmp, imgGray, tmp);
		resize(tmp, tmp, Size(n_rows, n_cols));
		cout << "Input k: ";
		cin >> k;
		ret = KNNclassify(tmp, k);
		cout << "The predict digit is: " << ret << endl;
	}
	release();
	return 0;
}