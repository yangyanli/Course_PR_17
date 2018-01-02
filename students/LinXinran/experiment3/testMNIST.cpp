#include<iostream>
#include<fstream>
#include<opencv.hpp>
#include"knn.hpp"

using namespace std;
using namespace cv;

#define kmax 25

double accuracyCnt[kmax + 1]; //for different k

void saveModelFile(int x)
{
	string filename = "./model/test";
	string filenum = "";
	if (x < 1000) filenum += "0";
	if (x < 100) filenum += "0";
	if (x < 10) filenum += "0";
	if (x < 10000) filenum += to_string(x);
	filename = filename + filenum + ".dat";
	
	ofstream file(filename);
	assert(file.is_open());
	for (int i = 1; i <= kmax; i++)
		file << accuracyCnt[i] << endl;
	file.close();
}

void readModelFile(int x)
{
	string filename = "./model/test";
	string filenum = "";
	if (x < 1000) filenum += "0";
	if (x < 100) filenum += "0";
	if (x < 10) filenum += "0";
	if (x < 10000) filenum += to_string(x);
	filename = filename + filenum + ".dat";
	
	ifstream file(filename);
	assert(file.is_open());
	for (int i = 1; i <= kmax; i++)
		file >> accuracyCnt[i];
	file.close();
}

void KNNclassify0(int x)
{
	while (!diffqueue.empty()) diffqueue.pop();
	//assert(k <= trainImagesNum);
	testset tmp;
	for (int i = 0; i < kmax; i++)
	{
		double diff = 0, dist2 = 0;
		for (int yi = 2; yi < n_cols - 2; yi++)
			for (int xi = 2; xi < n_rows - 2; xi++)
			{
				diff = abs((int)(*dataInputTrain[i].ptr(xi, yi)) - (int)(*dataInputTest[x].ptr(xi, yi)));
				dist2 += (diff * diff);
			}
		tmp.dist = sqrt(dist2);
		tmp.ind = i;
		diffqueue.push(tmp);
	}
	for (int i = kmax; i < trainImagesNum; i++)
	{
		double diff = 0, dist2 = 0;
		for (int yi = 2; yi < n_cols - 2; yi++)
			for (int xi = 2; xi < n_rows - 2; xi++)
			{
				diff = abs((int)(*dataInputTrain[i].ptr(xi, yi)) - (int)(*dataInputTest[x].ptr(xi, yi)));
				dist2 += (diff * diff);
			}
		tmp.dist = sqrt(dist2);
		tmp.ind = i;
		if (tmp.dist < diffqueue.top().dist)
		{
			diffqueue.pop();
			diffqueue.push(tmp);
		}
	}

	int numcnt[kmax + 1][10];
	memset(numcnt, 0, sizeof(numcnt));
	int knum = kmax;
	while (!diffqueue.empty())
	{
		tmp = diffqueue.top();
		for (int i = kmax; i >= knum; i--)
			numcnt[i][dataOutputTrain[tmp.ind]]++;
		diffqueue.pop();
		knum--;
	}
	for (int i = 1; i <= kmax; i++)
	{
		int testnum = 0;
		for (int j = 1; j < 10; j++)
			if (numcnt[i][j] > numcnt[i][testnum]) testnum = j;
		if (testnum == dataOutputTest[x]) accuracyCnt[i]++;
	}
}

void test0(int x)
{
	for (int i = x; i < testImagesNum; i++)
	{
		KNNclassify0(i);
		if ((i + 1) % 10 == 0) cout << "Tested " << (i + 1) * 100 / testImagesNum << "% (" << i + 1 << "/" << testImagesNum << ")." << endl;
		if ((i + 1) % 10 == 0) saveModelFile(i + 1);
	}
}

void getAcRate()
{
	int bestk = 1;
	for (int i = 2; i < kmax; i++)
		if (accuracyCnt[i] > accuracyCnt[bestk]) bestk = i;
	double accuracyRate = (double)accuracyCnt[bestk] / testImagesNum;
	cout << "Accuracy rate: " << accuracyRate << " (k = " << bestk << ")." << endl;
}

int main()
{
	init();
	/*cout << "Input k: ";
	int k;
	double accuracyRate;
	cin >> k;
	test(accuracyRate, k);
	cout << "Accuracy rate: " << accuracyRate << endl;*/
	memset(accuracyCnt, 0, sizeof(accuracyCnt));
	cout << "Load testdata file? (Y/N)" << endl;
	char c;
	cin >> c;
	if (c=='y'||c=='Y')
	{
		int numx;
		cout << "Input testdata file number:" << endl;
		cin >> numx;
		readModelFile(numx);
		test0(numx);
		getAcRate();
	}
	else
	{
		test0(0);
		getAcRate();
	}
	release();
	return 0;
}