#ifndef KNN_HPP
#define KNN_HPP

#include<iostream>
#include<fstream>
#include<string>
#include<cmath>
#include<vector>
#include<queue>
#include<random>
#include<ctime>
#include<cv.hpp>

using namespace cv;
using namespace std;

#define testImagesNum 10000
#define trainImagesNum 60000
#define n_rows 32
#define n_cols 32

//double* dataInputTrain;
//double* dataOutputTrain;
//double* dataInputTest;
//double* dataOutputTest;
//double* dataSingleImage;
//double* dataSingleLabel;
vector<Mat> dataInputTrain;
vector<int> dataOutputTrain;
vector<Mat> dataInputTest;
vector<int> dataOutputTest;
vector<Mat> dataSingleImage;
vector<int> dataSingleLabel;

struct testset
{
	double dist;
	int ind;
};

/*class cmp
{
bool isReverse;
public:
cmp(const bool& revparam = false)
{
isReverse = revparam;
}
bool operator()(testset& lhs, testset& rhs)
{
if (isReverse) return (lhs.dist > rhs.dist);
else return (lhs.dist < rhs.dist);
}
};

priority_queue<testset, vector<testset>, cmp> diffqueue(cmp(true));*/

bool operator<(testset lind, testset rind)
{
	return (lind.dist < rind.dist);
}

bool operator>(testset lind, testset rind)
{
	return (lind.dist > rind.dist);
}

priority_queue<testset> diffqueue;

/*void initVar(double* val, double c, int len)
{
	for (int i = 0; i < len; i++)
		val[i] = c;
}

int getIndex(int x, int y, int channel, int width, int height, int depth)
{
	assert(x >= 0 && x < width);
	assert(y >= 0 && y < height);
	assert(channel >= 0 && channel < depth);
	return (height * channel + y) * width + x;
}

double uniformRand(double minR, double maxR)
{
	random_device rd;
	static mt19937 gen(rd());
	//static mt19937 gen(1);
	uniform_real_distribution<double> dst(minR, maxR);
	return dst(gen);
}

bool uniformRand(double* src, int len, double minR, double maxR)
{
	for (int i = 0; i < len; i++)
		src[i] = uniformRand(minR, maxR);
	return true;
}*/

static int reverseInt(int i)
{
	uchar ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + (int)ch4;
}

static void readMnistImage(string filename, vector<Mat>& vect)
{
	ifstream file(filename, std::ios::binary);
	assert(file.is_open());
	int magicNumber = 0;
	int numberOfImages = 0;
	int nRows = 0;
	int nCols = 0;
	file.read((char*)&magicNumber, sizeof(magicNumber));
	magicNumber = reverseInt(magicNumber);
	file.read((char*)&numberOfImages, sizeof(numberOfImages));
	numberOfImages = reverseInt(numberOfImages);
	file.read((char*)&nRows, sizeof(nRows));
	nRows = reverseInt(nRows);
	file.read((char*)&nCols, sizeof(nCols));
	nCols = reverseInt(nCols);
	assert(nRows == 28 && nCols == 28);

	//int n_rows = 32, n_cols = 32; //28*28 -> 32*32

	for (int i = 0; i < numberOfImages; i++)
	{
		//Mat tmp = Mat::zeros(nRows, nCols, CV_8UC1);
		Mat tmp = Mat::zeros(n_rows, n_cols, CV_8UC1);
		for (int xi = 0; xi < nRows; xi++)
			for (int yi = 0; yi < nCols; yi++)
			{
				uchar temp = 0;
				file.read((char*)&temp, sizeof(temp));
				uchar* p = tmp.ptr(xi + 2, yi + 2);
				*p = temp;
			}
		vect.push_back(tmp);
	}
	file.close();
}

static void readMnistImage(string filename, double* dataDst)
{
	ifstream file(filename, std::ios::binary);
	assert(file.is_open());
	int magicNumber = 0;
	int numberOfImages = 0;
	int nRows = 0;
	int nCols = 0;
	file.read((char*)&magicNumber, sizeof(magicNumber));
	magicNumber = reverseInt(magicNumber);
	file.read((char*)&numberOfImages, sizeof(numberOfImages));
	numberOfImages = reverseInt(numberOfImages);
	file.read((char*)&nRows, sizeof(nRows));
	nRows = reverseInt(nRows);
	file.read((char*)&nCols, sizeof(nCols));
	nCols = reverseInt(nCols);
	assert(nRows == 28 && nCols == 28);

	int singleImageSize = 32 * 32; //28*28 -> 32*32
	const double scaleMin = -1;
	const double scaleMax = 1;
	for (int i = 0; i < numberOfImages; i++)
	{
		int addr = singleImageSize * i;
		for (int xi = 0; xi < nRows; xi++)
			for (int yi = 0; yi < nCols; yi++)
			{
				uchar temp = 0;
				file.read((char*)&temp, sizeof(temp));
				dataDst[addr + 32 * (xi + 2) + yi + 2] = (temp / 255.0) * (scaleMax - scaleMin) + scaleMin;
			}
	}
	file.close();
}

static void readMnistLabel(string filename, vector<int>& vect)
{
	ifstream file(filename, std::ios::binary);
	assert(file.is_open());
	int magicNumber = 0;
	int numberOfImages = 0;
	file.read((char*)&magicNumber, sizeof(magicNumber));
	magicNumber = reverseInt(magicNumber);
	file.read((char*)&numberOfImages, sizeof(numberOfImages));
	numberOfImages = reverseInt(numberOfImages);

	const double scaleMax = 0.8;
	for (int i = 0; i < numberOfImages; i++)
	{
		uchar temp = 0;
		file.read((char*)&temp, sizeof(temp));
		vect.push_back((int)temp);
	}
	file.close();
}

static void readMnistLabel(string filename, double* dataDst)
{
	ifstream file(filename, std::ios::binary);
	assert(file.is_open());
	int magicNumber = 0;
	int numberOfImages = 0;
	file.read((char*)&magicNumber, sizeof(magicNumber));
	magicNumber = reverseInt(magicNumber);
	file.read((char*)&numberOfImages, sizeof(numberOfImages));
	numberOfImages = reverseInt(numberOfImages);

	const double scaleMax = 0.8;
	for (int i = 0; i < numberOfImages; i++)
	{
		uchar temp = 0;
		file.read((char*)&temp, sizeof(temp));
		dataDst[i * 10 + temp] = scaleMax; //output map number: 10
	}
	file.close();
}

static string getImageName(int number, int arr[])
{
	string str = "";
	if (arr[number] < 10000) str += "0";
	if (arr[number] < 1000) str += "0";
	if (arr[number] < 100) str += "0";
	if (arr[number] < 10) str += "0";
	arr[number]++;
	str += to_string(arr[number]);
	str = "pic" + to_string(number) + "_" + str;
	return str;
}

bool getSrcData() //get MNIST data
{
	//assert(dataInputTest && dataOutputTest && dataInputTrain && dataOutputTrain);
	string testImages = "./data/t10k-images.idx3-ubyte";
	readMnistImage(testImages, dataInputTest);
	string testLabels = "./data/t10k-labels.idx1-ubyte";
	readMnistLabel(testLabels, dataOutputTest);
	string trainImages = "./data/train-images.idx3-ubyte";
	readMnistImage(trainImages, dataInputTrain);
	string trainLabels = "./data/train-labels.idx1-ubyte";
	readMnistLabel(trainLabels, dataOutputTrain);
	return true;
}

void init()
{
	/*int len1 = imgSize[0] * imgSize[0] * trainImagesNum;
	dataInputTrain = new double[len1];
	initVar(dataInputTrain, -1.0, len1);
	int len2 = mapNum[6] * trainImagesNum;
	dataOutputTrain = new double[len2];
	initVar(dataOutputTrain, -0.8, len2);
	int len3 = imgSize[0] * imgSize[0] * testImagesNum;
	dataInputTest = new double[len3];
	initVar(dataInputTest, -1.0, len3);
	int len4 = mapNum[6] * testImagesNum;
	dataOutputTest = new double[len4];
	initVar(dataOutputTest, -0.8, len4);
	dataSingleImage = NULL;
	dataSingleLabel = NULL;*/
	dataInputTrain.clear();
	dataOutputTrain.clear();
	dataInputTest.clear();
	dataOutputTest.clear();
	dataSingleImage.clear();
	dataSingleLabel.clear();
	getSrcData();
}

void release()
{
	/*if (dataInputTrain)
	{
		delete[] dataInputTrain;
		dataInputTrain = NULL;
	}
	if (dataOutputTrain)
	{
		delete[] dataOutputTrain;
		dataOutputTrain = NULL;
	}
	if (dataInputTest)
	{
		delete[] dataInputTest;
		dataInputTest = NULL;
	}
	if (dataOutputTest)
	{
		delete[] dataOutputTest;
		dataOutputTest = NULL;
	}*/
	dataInputTrain.clear();
	dataOutputTrain.clear();
	dataInputTest.clear();
	dataOutputTest.clear();
	dataSingleImage.clear();
	dataSingleLabel.clear();
}

int MnistToImage()
{
	//test images and labels

	//read MNIST image into OpenCV vector<Mat>
	string testImages = "./data/t10k-images.idx3-ubyte";
	vector<Mat> testImagesVect;
	readMnistImage(testImages, testImagesVect);

	//read MNIST label into vector<int>
	string testLabels = "./data/t10k-labels.idx1-ubyte";
	vector<int> testLabelsVect;
	readMnistLabel(testLabels, testLabelsVect);

	/*if (testImagesVect.size() != testLabelsVect.size())
	{
	cout << "parse MNIST test file error!" << endl;
	return -1;
	}*/
	assert(testImagesVect.size() == testLabelsVect.size());

	//save test images
	int cntDigits[10];
	memset(cntDigits, 0, sizeof(cntDigits));
	string saveTestImagesPath = "./testimages/";
	for (int i = 0; i < testImagesVect.size(); i++)
	{
		int number = testLabelsVect[i];
		string imageName = getImageName(number, cntDigits);
		imageName = saveTestImagesPath + imageName + ".jpg";
		imwrite(imageName, testImagesVect[i]);
	}

	//train images and labels

	//read MNIST image into OpenCV vector<Mat>
	string trainImages = "./data/train-images.idx3-ubyte";
	vector<Mat> trainImagesVect;
	readMnistImage(trainImages, trainImagesVect);

	//read MNIST label into vector<int>
	string trainLabels = "./data/train-labels.idx1-ubyte";
	vector<int> trainLabelsVect;
	readMnistLabel(trainLabels, trainLabelsVect);

	/*if (trainImagesVect.size() != trainLabelsVect.size())
	{
	cout << "parse MNIST test file error!" << endl;
	return -1;
	}*/
	assert(trainImagesVect.size() == trainLabelsVect.size());

	//save train images
	memset(cntDigits, 0, sizeof(cntDigits));
	string saveTrainImagesPath = "./trainimages/";
	for (int i = 0; i < trainImagesVect.size(); i++)
	{
		int number = trainLabelsVect[i];
		string imageName = getImageName(number, cntDigits);
		imageName = saveTrainImagesPath + imageName + ".jpg";
		imwrite(imageName, trainImagesVect[i]);
	}

	//save big images
	string bigImages = "./bigimages/";
	int width = 28 * 20;
	int height = 28 * 10;
	Mat dst(height, width, CV_8UC1);

	for (int i = 0; i < 10; i++)
		for (int j = 1; j <= 20; j++)
		{
			int x = (j - 1) * 28;
			int y = i * 28;
			Mat part = dst(Rect(x, y, 28, 28));

			string str = "";
			if (j < 10000) str += "0";
			if (j < 1000) str += "0";
			if (j < 100) str += "0";
			if (j < 10) str += "0";
			str += to_string(j);
			str = to_string(i) + "_" + str + ".jpg";
			string inputImg = saveTrainImagesPath + str;

			Mat src = imread(inputImg, 0);
			if (src.empty())
			{
				fprintf(stderr, "read image error: %s\n", inputImg.c_str());
				return -1;
			}

			src.copyTo(part);
		}

	string outputImg = bigImages + "result.png";
	imwrite(outputImg, dst);
	return 0;
}

int KNNclassify(Mat vect, int k)
{
	while (!diffqueue.empty()) diffqueue.pop();
	assert(k <= trainImagesNum);
	testset tmp;
	for (int i = 0; i < k; i++)
	{
		double diff = 0, dist2 = 0;
		for (int yi = 0; yi < n_cols; yi++)
			for (int xi = 0; xi < n_rows; xi++)
			{
				diff = abs((int)(*dataInputTrain[i].ptr(xi, yi)) - (int)(*vect.ptr(xi, yi)));
				dist2 += (diff * diff);
			}
		tmp.dist = sqrt(dist2);
		tmp.ind = i;
		diffqueue.push(tmp);
	}
	for (int i = k; i < trainImagesNum; i++)
	{
		double diff = 0, dist2 = 0;
		for (int yi = 0; yi < n_cols; yi++)
			for (int xi = 0; xi < n_rows; xi++)
			{
				diff = abs((int)(*dataInputTrain[i].ptr(xi, yi)) - (int)(*vect.ptr(xi, yi)));
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

	int numcnt[10];
	memset(numcnt, 0, sizeof(numcnt));
	while (!diffqueue.empty())
	{
		tmp = diffqueue.top();
		numcnt[dataOutputTrain[tmp.ind]]++;
		diffqueue.pop();
	}
	int testnum = 0;
	for (int i = 1; i < 10; i++)
		if (numcnt[i] > numcnt[testnum]) testnum = i;
	return testnum;
}

/*bool train()
{
}*/

void test(double& accuracyRate, int k)
{
	int accuracyCnt = 0;
	for (int i = 0; i < testImagesNum/100; i++)
	{
		int testnum = KNNclassify(dataInputTest[i], k);
		if (testnum == dataOutputTest[i]) accuracyCnt++;
		if ((i + 1) * 100 % (testImagesNum/100) == 0) cout << "Tested " << (i + 1) * 100 / testImagesNum << "% (" << i + 1 << "/" << testImagesNum << ")." << endl;
	}
	accuracyRate = (double)accuracyCnt / testImagesNum;
}

int predictTest()
{
	init();
	cout << "Input k: ";
	int k;
	double accuracyRate;
	cin >> k;
	test(accuracyRate, k);
	cout << "Accuracy rate: " << accuracyRate << endl;
	release();
	return 0;
}

#endif //KNN_HPP
