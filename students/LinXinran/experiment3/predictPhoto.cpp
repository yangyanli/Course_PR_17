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
	int k;
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

		cout << "Input k: ";
		cin >> k;
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

			resize(tmp, tmp, Size(n_rows, n_cols));
			int ret = KNNclassify(tmp, k);

			Point tp;
			tp.x = rect.x;
			tp.y = rect.y;
			putText(src, to_string(ret), tp, FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255), 1, 8, false);

			rectangle(src, rect.tl(), rect.br(), Scalar(255, 0, 0));
		}
		imshow("contoursImg", src);
		waitKey(0);
		contours.clear();
		hierarchy.clear();
	}
	release();

	return 0;
}