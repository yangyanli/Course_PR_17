#ifndef CNN_HPP
#define CNN_HPP

#include<iostream>
#include<fstream>
#include<string>
#include<cmath>
#include<vector>
#include<random>
#include<ctime>
#include<cv.hpp>

using namespace cv;
using namespace std;

#define eps 1e-8
#define INF 0x3f3f3f3f

double* dataInputTrain;
double* dataOutputTrain;
double* dataInputTest;
double* dataOutputTest;
double* dataSingleImage;
double* dataSingleLabel;

const int testImagesNum = 10000;
const int trainImagesNum = 60000;
const int imgSize[7] = { 32, 28, 14, 10, 5, 1, 1 };
const int mapNum[7] = { 1, 6, 6, 16, 16, 120, 10 };
const int epochNum = 100;
const double learningRate = 0.01; //learning rate, set by yourself
const double accuracyRate = 0.98; //minimum accuracy rate, set by yourself

const int weightNum[7] = { 0, 150, 6, 2400, 16, 48000, 1200 };
const int biasNum[7] = { 0, 6, 6, 16, 16, 120, 10 };
double weightC1[150]; //C1 weight number: 5*5*6*1=150
double biasC1[6]; //C1 threshold number: 6
double weightS2[6]; //S2 weight number: 6
double biasS2[6]; //C1 threshold number: 6
double weightC3[2400]; //C1 weight number: 5*5*16*6=2400
double biasC3[16]; //C1 threshold number: 16
double weightS4[16]; //C1 weight number: 1*16=16
double biasS4[16]; //C1 threshold number: 16
double weightC5[48000]; //C1 weight number: 5*5*16*120=48000
double biasC5[120]; //C1 threshold number: 120
double weightOutput[1200]; //output weight number: 120*10=1200
double biasOutput[10]; //output threshold number: 10

double deltaWeightC1[150]; //C1 weight number: 5*5*6*1=150
double deltaBiasC1[6]; //C1 threshold number: 6
double deltaWeightS2[6]; //S2 weight number: 6
double deltaBiasS2[6]; //C1 threshold number: 6
double deltaWeightC3[2400]; //C1 weight number: 5*5*16*6=2400
double deltaBiasC3[16]; //C1 threshold number: 16
double deltaWeightS4[16]; //C1 weight number: 1*16=16
double deltaBiasS4[16]; //C1 threshold number: 16
double deltaWeightC5[48000]; //C1 weight number: 5*5*16*120=48000
double deltaBiasC5[120]; //C1 threshold number: 120
double deltaWeightOutput[1200]; //output weight number: 120*10=1200
double deltaBiasOutput[10]; //output threshold number: 10

//single image data
const int neuronNum[7] = { 1024, 4704, 1176, 1600, 400, 120, 10 };
double neuronInput[1024]; //input neuron number: 32*32=1024
double neuronC1[4704]; //C1 neuron number: 28*28*6=4704
double neuronS2[1176]; //S2 neuron number: 14*14*6=1176
double neuronC3[1600]; //C3 neuron number: 10*10*16=1600
double neuronS4[400]; //S4 neuron number: 5*5*16=400
double neuronC5[120]; //C5 neuron number: 1*1*120=120
double neuronOutput[10]; //output neuron number: 1*10=10

double deltaNeuronInput[1024]; //input neuron number: 32*32=1024
double deltaNeuronC1[4704]; //C1 neuron number: 28*28*6=4704
double deltaNeuronS2[1176]; //S2 neuron number: 14*14*6=1176
double deltaNeuronC3[1600]; //C3 neuron number: 10*10*16=1600
double deltaNeuronS4[400]; //S4 neuron number: 5*5*16=400
double deltaNeuronC5[120]; //C5 neuron number: 1*1*120=120
double deltaNeuronOutput[10]; //output neuron number: 1*10=10

vector<vector<pair<int, int> > > out2wiS2; //outId -> (weightId, inId)
vector<int> out2biasS2;
vector<vector<pair<int, int> > > out2wiS4; //outId -> (weightId, inId)
vector<int> out2biasS4;
vector<vector<pair<int, int> > > in2woC3; //inId -> (weightId,outId)
vector<vector<pair<int, int> > > weight2ioC3; //weightId -> (inId,outId)
vector<vector<int> > bias2outC3;
vector<vector<pair<int, int> > > in2woC1; //inId -> (weightId,outId)
vector<vector<pair<int, int> > > weight2ioC1; //weightId -> (inId,outId)
vector<vector<int> > bias2outC1;

double eWeightC1[150];
double eBiasC1[6];
double eWeightS2[6];
double eBiasS2[6];
double eWeightC3[2400];
double eBiasC3[16];
double eWeightS4[16];
double eBiasS4[16];
double eWeightC5[48000];
double eBiasC5[120];
double eWeightOutput[1200];
double eBiasOutput[10];

static const bool tbl[6][16] = {
	{ true, false, false, false, true, true, true, false, false, true, true, true, true, false, true, true },
	{ true, true, false, false, false, true, true, true, false, false, true, true, true, true, false, true },
	{ true, true, true, false, false, false, true, true, true, false, false, true, false, true, true, true },
	{ false, true, true, true, false, false, true, true, true, true, false, false, true, false, true, true },
	{ false, false, true, true, true, false, false, true, true, true, true, false, true, true, false, true },
	{ false, false, false, true, true, true, false, false, true, true, true, true, false, true, true, true }
};

void initVar(double* val, double c, int len)
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

double dot(double* s1, double* s2, int len) //dot product
{
	double sum = 0;
	for (int i = 0; i < len; i++)
		sum += s1[i] * s2[i];
	return sum;
}

double tanh_d(double x) //derivative of tanh(x)
{
	return (1.0 - x * x);
}

void gradient(const double* y, const double* t, double* dst, int len)
{
	for (int i = 0; i < len; i++)
		dst[i] = y[i] - t[i];
}

bool mulAdd(const double* src, double c, int len, double* dst)
{
	for (int i = 0; i < len; i++)
		dst[i] += (src[i] * c);
	return true;
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
}

bool initWeightThreshold() //initialize, random weight between [-1.0, 1.0]
{
	srand((uint)time(NULL));
	const double scale = 6.0;
	double minR, maxR;
	minR = -sqrt(scale / (25 + 150));
	maxR = sqrt(scale / (25 + 150));
	uniformRand(weightC1, weightNum[1], minR, maxR);
	initVar(biasC1, 0.0, biasNum[1]);
	minR = -sqrt(scale / (4 + 1));
	maxR = sqrt(scale / (4 + 1));
	uniformRand(weightS2, weightNum[2], minR, maxR);
	initVar(biasS2, 0.0, biasNum[2]);
	minR = -sqrt(scale / (150 + 400));
	maxR = sqrt(scale / (150 + 400));
	uniformRand(weightC3, weightNum[3], minR, maxR);
	initVar(biasC3, 0.0, biasNum[3]);
	minR = -sqrt(scale / (4 + 1));
	maxR = sqrt(scale / (4 + 1));
	uniformRand(weightS4, weightNum[4], minR, maxR);
	initVar(biasS4, 0.0, biasNum[4]);
	minR = -sqrt(scale / (400 + 3000));
	maxR = sqrt(scale / (400 + 3000));
	uniformRand(weightC5, weightNum[5], minR, maxR);
	initVar(biasC5, 0.0, biasNum[5]);
	minR = -sqrt(scale / (120 + 10));
	maxR = sqrt(scale / (120 + 10));
	uniformRand(weightOutput, weightNum[6], minR, maxR);
	initVar(biasOutput, 0.0, biasNum[6]);
	return true;
}

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

	for (int i = 0; i < numberOfImages; i++)
	{
		Mat tmp = Mat::zeros(nRows, nCols, CV_8UC1);
		for (int xi = 0; xi < nRows; xi++)
			for (int yi = 0; yi < nCols; yi++)
			{
				uchar temp = 0;
				file.read((char*)&temp, sizeof(temp));
				uchar* p = tmp.ptr(xi, yi);
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

/*void test_readMnistImage()
{
	vector<Mat> images;
	readMnistImage("t10k-images.idx3-ubyte", images);
	for (int i = 0; i < images.size(); i++)
	{
		for (int j = 0; j < images[i].cols; j++)
		for (int k = 0; k < images[i].rows; k++)
		{
			uchar* p = images[i].ptr(k, j);
			cout << (int)(*p) << " ";
		}
		cout << endl;
	}
}*/

bool getSrcData() //get MNIST data
{
	assert(dataInputTest && dataOutputTest && dataInputTrain && dataOutputTrain);
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
	int len1 = imgSize[0] * imgSize[0] * trainImagesNum;
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
	dataSingleLabel = NULL;
	initVar(eWeightC1, 0.0, weightNum[1]);
	initVar(eBiasC1, 0.0, biasNum[1]);
	initVar(eWeightS2, 0.0, weightNum[2]);
	initVar(eBiasS2, 0.0, biasNum[2]);
	initVar(eWeightC3, 0.0, weightNum[3]);
	initVar(eBiasC3, 0.0, biasNum[3]);
	initVar(eWeightS4, 0.0, weightNum[4]);
	initVar(eBiasS4, 0.0, biasNum[4]);
	initVar(eWeightC5, 0.0, weightNum[5]);
	initVar(eBiasC5, 0.0, biasNum[5]);
	initVar(eWeightOutput, 0.0, weightNum[6]);
	initVar(eBiasOutput, 0.0, biasNum[6]);
	initWeightThreshold();
	getSrcData();
}

void release()
{
	if (dataInputTrain)
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
	}
}

bool forwardC1()
{
	initVar(neuronC1, 0.0, neuronNum[1]);
	for (int i = 0; i < mapNum[1]; i++)
	{
		for (int j = 0; j < mapNum[0]; j++)
		{
			int addr1 = getIndex(0, 0, mapNum[0] * i + j, 5, 5, mapNum[1] * mapNum[0]);
			int addr2 = getIndex(0, 0, j, imgSize[0], imgSize[0], mapNum[0]);
			int addr3 = getIndex(0, 0, i, imgSize[1], imgSize[1], mapNum[1]);

			const double* pw = &weightC1[0] + addr1;
			const double* pi = dataSingleImage + addr2;
			double* pa = &neuronC1[0] + addr3;

			for (int y = 0; y < imgSize[1]; y++)
				for (int x = 0; x < imgSize[1]; x++)
				{
					const double* ppw = pw;
					const double* ppi = pi + y * imgSize[0] + x;
					double sum = 0;
					for (int wy = 0; wy < 5; wy++)
						for (int wx = 0; wx < 5; wx++)
						{
							//sum += *ppw * ppi[wy * imgSize[0] + wx];
							//*ppw++;
							sum += *ppw++ * ppi[wy * imgSize[0] + wx];
						}
					pa[y * imgSize[1] + x] += sum;
				}
		}
		int addr3 = getIndex(0, 0, i, imgSize[1], imgSize[1], mapNum[1]);
		double* pa = &neuronC1[0] + addr3;
		double b = biasC1[i];
		for (int y = 0; y < imgSize[1]; y++)
			for (int x = 0; x < imgSize[1]; x++)
				pa[y * imgSize[1] + x] += b;
	}

	for (int i = 0; i < neuronNum[1]; i++)
		neuronC1[i] = tanh(neuronC1[i]);

	return true;
}

bool forwardS2()
{
	initVar(neuronS2, 0.0, neuronNum[2]);
	double scaleFactor = 1.0 / (2 * 2);
	assert(out2wiS2.size() == neuronNum[2]);
	assert(out2biasS2.size() == neuronNum[2]);
	for (int i = 0; i < neuronNum[2]; i++)
	{
		const vector<pair<int, int> >& connections = out2wiS2[i];
		neuronS2[i] = 0;
		for (int ind = 0; ind < connections.size(); ind++)
			neuronS2[i] += weightS2[connections[ind].first] * neuronC1[connections[ind].second];
		neuronS2[i] = neuronS2[i] * scaleFactor + biasS2[out2biasS2[i]];
	}

	for (int i = 0; i < neuronNum[2]; i++)
		neuronS2[i] = tanh(neuronS2[i]);

	return true;
}

bool forwardC3()
{
	initVar(neuronC3, 0.0, neuronNum[3]);
	for (int i = 0; i < mapNum[3]; i++)
	{
		for (int j = 0; j < mapNum[2]; j++)
		{
			if (!tbl[j][i]) continue;

			int addr1 = getIndex(0, 0, mapNum[2] * i + j, 5, 5, mapNum[3] * mapNum[2]);
			int addr2 = getIndex(0, 0, j, imgSize[2], imgSize[2], mapNum[2]);
			int addr3 = getIndex(0, 0, i, imgSize[3], imgSize[3], mapNum[3]);

			const double* pw = &weightC3[0] + addr1;
			const double* pi = &neuronS2[0] + addr2;
			double* pa = &neuronC3[0] + addr3;

			for (int y = 0; y < imgSize[3]; y++)
				for (int x = 0; x < imgSize[3]; x++)
				{
					const double* ppw = pw;
					const double* ppi = pi + y * imgSize[2] + x;
					double sum = 0;
					for (int wy = 0; wy < 5; wy++)
						for (int wx = 0; wx < 5; wx++)
						{
							//sum += *ppw * ppi[wy * imgSize[2] + wx];
							//*ppw++;
							sum += *ppw++ * ppi[wy * imgSize[2] + wx];
						}
					pa[y * imgSize[3] + x] += sum;
				}
		}
		int addr3 = getIndex(0, 0, i, imgSize[3], imgSize[3], mapNum[3]);
		double* pa = &neuronC3[0] + addr3;
		double b = biasC3[i];
		for (int y = 0; y < imgSize[3]; y++)
			for (int x = 0; x < imgSize[3]; x++)
				pa[y * imgSize[3] + x] += b;
	}

	for (int i = 0; i < neuronNum[3]; i++)
		neuronC3[i] = tanh(neuronC3[i]);

	return true;
}

bool forwardS4()
{
	initVar(neuronS4, 0.0, neuronNum[4]);
	double scaleFactor = 1.0 / (2 * 2);
	assert(out2wiS4.size() == neuronNum[4]);
	assert(out2biasS4.size() == neuronNum[4]);
	for (int i = 0; i < neuronNum[4]; i++)
	{
		const vector<pair<int, int> >& connections = out2wiS4[i];
		neuronS4[i] = 0;
		for (int ind = 0; ind < connections.size(); ind++)
			neuronS4[i] += weightS4[connections[ind].first] * neuronC3[connections[ind].second];
		neuronS4[i] = neuronS4[i] * scaleFactor + biasS4[out2biasS4[i]];
	}

	for (int i = 0; i < neuronNum[4]; i++)
		neuronS4[i] = tanh(neuronS4[i]);

	return true;
}

bool forwardC5()
{
	initVar(neuronC5, 0.0, neuronNum[5]);
	for (int i = 0; i < mapNum[5]; i++)
	{
		for (int j = 0; j < mapNum[4]; j++)
		{
			int addr1 = getIndex(0, 0, mapNum[4] * i + j, 5, 5, mapNum[5] * mapNum[4]);
			int addr2 = getIndex(0, 0, j, imgSize[4], imgSize[4], mapNum[4]);
			int addr3 = getIndex(0, 0, i, imgSize[5], imgSize[5], mapNum[5]);

			const double* pw = &weightC5[0] + addr1;
			const double* pi = &neuronS4[0] + addr2;
			double* pa = &neuronC5[0] + addr3;

			for (int y = 0; y < imgSize[5]; y++)
				for (int x = 0; x < imgSize[5]; x++)
				{
					const double* ppw = pw;
					const double* ppi = pi + y * imgSize[4] + x;
					double sum = 0;
					for (int wy = 0; wy < 5; wy++)
						for (int wx = 0; wx < 5; wx++)
						{
							//sum += *ppw * ppi[wy * imgSize[4] + wx];
							//*ppw++;
							sum += *ppw++ * ppi[wy * imgSize[4] + wx];
						}
					pa[y * imgSize[5] + x] += sum;
				}
		}
		int addr3 = getIndex(0, 0, i, imgSize[5], imgSize[5], mapNum[5]);
		double* pa = &neuronC5[0] + addr3;
		double b = biasC5[i];
		for (int y = 0; y < imgSize[5]; y++)
			for (int x = 0; x < imgSize[5]; x++)
				pa[y * imgSize[5] + x] += b;
	}

	for (int i = 0; i < neuronNum[5]; i++)
		neuronC5[i] = tanh(neuronC5[i]);

	return true;
}

bool forwardOutput()
{
	initVar(neuronOutput, 0.0, neuronNum[6]);
	for (int i = 0; i < neuronNum[6]; i++)
	{
		neuronOutput[i] = 0;
		for (int j = 0; j < neuronNum[5]; j++)
			neuronOutput[i] += weightOutput[j * neuronNum[6] + i] * neuronC5[j];
		neuronOutput[i] += biasOutput[i];
	}

	for (int i = 0; i < neuronNum[6]; i++)
		neuronOutput[i] = tanh(neuronOutput[i]);

	return true;
}

bool backwardOutput()
{
	initVar(deltaNeuronOutput, 0.0, neuronNum[6]);
	double* dE_dy = new double[neuronNum[6]];
	initVar(dE_dy, 0.0, neuronNum[6]);
	gradient(neuronOutput, dataSingleLabel, dE_dy, neuronNum[6]); //loss function gradient, mean squared error

	//delta = dE/da = (dE/dy) * (dy/da)
	for (int i = 0; i < neuronNum[6]; i++)
	{
		double* dy_da = new double[neuronNum[6]];
		initVar(dy_da, 0.0, neuronNum[6]);
		dy_da[i] = tanh_d(neuronOutput[i]);
		deltaNeuronOutput[i] = dot(dE_dy, dy_da, neuronNum[6]);
	}

	return true;
}

bool backwardC5()
{
	initVar(deltaNeuronC5, 0.0, neuronNum[5]);
	initVar(deltaWeightOutput, 0.0, weightNum[6]);
	initVar(deltaBiasOutput, 0.0, biasNum[6]);
	for (int i = 0; i < neuronNum[5]; i++)
	{
		//propagate delta to previous layer
		deltaNeuronC5[i] = dot(&deltaNeuronOutput[0], &weightOutput[i * neuronNum[6]], neuronNum[6]);
		deltaNeuronC5[i] *= tanh_d(neuronC5[i]);
	}

	//accumulate weight-step using delta
	for (int i = 0; i < neuronNum[5]; i++)
		mulAdd(&deltaNeuronOutput[0], neuronC5[i], neuronNum[5], &deltaWeightOutput[0] + i * neuronNum[6]);

	for (int i = 0; i < biasNum[6]; i++)
		deltaBiasOutput[i] += deltaNeuronOutput[i];

	return true;
}

bool backwardS4()
{
	initVar(deltaNeuronS4, 0.0, neuronNum[4]);
	initVar(deltaWeightC5, 0.0, weightNum[5]);
	initVar(deltaBiasC5, 0.0, biasNum[5]);

	//propagate delta to previous layer
	for (int inc = 0; inc < mapNum[4]; inc++)
		for (int outc = 0; outc < mapNum[5]; outc++)
		{
			int addr1 = getIndex(0, 0, mapNum[4] * outc + inc, 5, 5, mapNum[4] * mapNum[5]);
			int addr2 = getIndex(0, 0, outc, imgSize[5], imgSize[5], mapNum[5]);
			int addr3 = getIndex(0, 0, inc, imgSize[4], imgSize[4], mapNum[4]);

			const double* pw = &weightC5[0] + addr1;
			const double* pdeltaSrc = &deltaNeuronC5[0] + addr2;
			double* pdeltaDst = &deltaNeuronS4[0] + addr3;

			for (int y = 0; y < imgSize[5]; y++)
				for (int x = 0; x < imgSize[5]; x++)
				{
					const double* ppw = pw;
					const double ppdeltaSrc = pdeltaSrc[y * imgSize[5] + x];
					double* ppdeltaDst = pdeltaDst + y * imgSize[4] + x;

					for (int wy = 0; wy < 5; wy++)
						for (int wx = 0; wx < 5; wx++)
						{
							//ppdeltaDst[wy * imgSize[4] + wx] += *ppw * ppdeltaSrc;
							//*ppw++;
							ppdeltaDst[wy * imgSize[4] + wx] += *ppw++ * ppdeltaSrc;
						}
				}
		}

	for (int i = 0; i < neuronNum[4]; i++)
		deltaNeuronS4[i] *= tanh_d(neuronS4[i]);

	//accumulate dw
	for (int inc = 0; inc < mapNum[4]; inc++)
		for (int outc = 0; outc < mapNum[5]; outc++)
			for (int wy = 0; wy < 5; wy++)
				for (int wx = 0; wx < 5; wx++)
				{
					int addr1 = getIndex(wx, wy, inc, imgSize[4], imgSize[4], mapNum[4]);
					int addr2 = getIndex(0, 0, outc, imgSize[5], imgSize[5], mapNum[5]);
					int addr3 = getIndex(wx, wy, mapNum[4] * outc + inc, 5, 5, mapNum[4] * mapNum[5]);

					double dst = 0;
					double* pre = &neuronS4[0] + addr1;
					double* delta = &deltaNeuronC5[0] + addr2;

					for (int y = 0; y < imgSize[5]; y++)
						dst += dot(pre + y * imgSize[4], delta + y*imgSize[5], imgSize[5]);
					deltaWeightC5[addr3] += dst;
				}

	//accumulate db
	for (int outc = 0; outc < mapNum[5]; outc++)
	{
		int addr2 = getIndex(0, 0, outc, imgSize[5], imgSize[5], mapNum[5]);
		const double* delta = &deltaNeuronC5[0] + addr2;

		for (int y = 0; y < imgSize[5]; y++)
			for (int x = 0; x < imgSize[5]; x++)
				deltaBiasC5[outc] += delta[y * imgSize[5] + x];
	}

	return true;
}

bool backwardC3()
{
	initVar(deltaNeuronC3, 0.0, neuronNum[3]);
	initVar(deltaWeightS4, 0.0, weightNum[4]);
	initVar(deltaBiasS4, 0.0, biasNum[4]);
	double scaleFactor = 1.0 / (2 * 2);

	assert(in2woC3.size() == neuronNum[3]);
	assert(weight2ioC3.size() == weightNum[4]);
	assert(bias2outC3.size() == biasNum[4]);

	for (int i = 0; i < neuronNum[3]; i++)
	{
		const vector<pair<int, int> >& connections = in2woC3[i];
		double delta = 0;
		for (int j = 0; j < connections.size(); j++)
			delta += weightS4[connections[j].first] * deltaNeuronS4[connections[j].second];
		deltaNeuronC3[i] = delta * scaleFactor * tanh_d(neuronC3[i]);
	}

	for (int i = 0; i < weightNum[4]; i++)
	{
		const vector<pair<int, int> >& connections = weight2ioC3[i];
		double diff = 0;
		for (int j = 0; j < connections.size(); j++)
			diff += neuronC3[connections[j].first] * deltaNeuronS4[connections[j].second];
		deltaWeightS4[i] += diff * scaleFactor;
	}

	for (int i = 0; i < biasNum[4]; i++)
	{
		const vector<int>& outs = bias2outC3[i];
		double diff = 0;
		for (int j = 0; j < outs.size(); j++)
			diff += deltaNeuronS4[outs[j]];
		deltaBiasS4[i] += diff;
	}

	return true;
}

bool backwardS2()
{
	initVar(deltaNeuronS2, 0.0, neuronNum[2]);
	initVar(deltaWeightC3, 0.0, weightNum[3]);
	initVar(deltaBiasC3, 0.0, biasNum[3]);

	//propagate delta to previous layer
	for (int inc = 0; inc < mapNum[2]; inc++)
		for (int outc = 0; outc < mapNum[3]; outc++)
		{
			int addr1 = getIndex(0, 0, mapNum[2] * outc + inc, 5, 5, mapNum[2] * mapNum[3]);
			int addr2 = getIndex(0, 0, outc, imgSize[3], imgSize[3], mapNum[3]);
			int addr3 = getIndex(0, 0, inc, imgSize[2], imgSize[2], mapNum[2]);

			const double* pw = &weightC3[0] + addr1;
			const double* pdeltaSrc = &deltaNeuronC3[0] + addr2;
			double* pdeltaDst = &deltaNeuronS2[0] + addr3;

			for (int y = 0; y < imgSize[3]; y++)
				for (int x = 0; x < imgSize[3]; x++)
				{
					const double* ppw = pw;
					const double ppdeltaSrc = pdeltaSrc[y * imgSize[3] + x];
					double* ppdeltaDst = pdeltaDst + y * imgSize[2] + x;

					for (int wy = 0; wy < 5; wy++)
						for (int wx = 0; wx < 5; wx++)
						{
							//ppdeltaDst[wy * imgSize[2] + wx] += *ppw * ppdeltaSrc;
							//*ppw++;
							ppdeltaDst[wy * imgSize[2] + wx] += *ppw++ * ppdeltaSrc;
						}
				}
		}

	for (int i = 0; i < neuronNum[2]; i++)
		deltaNeuronS2[i] *= tanh_d(neuronS2[i]);

	//accumulate dw
	for (int inc = 0; inc < mapNum[2]; inc++)
		for (int outc = 0; outc < mapNum[3]; outc++)
		{
			if (!tbl[inc][outc]) continue;
			
			for (int wy = 0; wy < 5; wy++)
				for (int wx = 0; wx < 5; wx++)
				{
					int addr1 = getIndex(wx, wy, inc, imgSize[2], imgSize[2], mapNum[2]);
					int addr2 = getIndex(0, 0, outc, imgSize[3], imgSize[3], mapNum[3]);
					int addr3 = getIndex(wx, wy, mapNum[2] * outc + inc, 5, 5, mapNum[2] * mapNum[3]);

					double dst = 0;
					double* pre = &neuronS2[0] + addr1;
					double* delta = &deltaNeuronC3[0] + addr2;

					for (int y = 0; y < imgSize[3]; y++)
						dst += dot(pre + y * imgSize[2], delta + y*imgSize[3], imgSize[3]);
					deltaWeightC3[addr3] += dst;
				}
		}

	//accumulate db
	for (int outc = 0; outc < mapNum[3]; outc++)
	{
		int addr2 = getIndex(0, 0, outc, imgSize[3], imgSize[3], mapNum[3]);
		const double* delta = &deltaNeuronC3[0] + addr2;

		for (int y = 0; y < imgSize[3]; y++)
			for (int x = 0; x < imgSize[3]; x++)
				deltaBiasC3[outc] += delta[y * imgSize[3] + x];
	}

	return true;
}

bool backwardC1()
{
	initVar(deltaNeuronC1, 0.0, neuronNum[1]);
	initVar(deltaWeightS2, 0.0, weightNum[2]);
	initVar(deltaBiasS2, 0.0, biasNum[2]);
	double scaleFactor = 1.0 / (2 * 2);

	assert(in2woC1.size() == neuronNum[1]);
	assert(weight2ioC1.size() == weightNum[2]);
	assert(bias2outC1.size() == biasNum[2]);

	for (int i = 0; i < neuronNum[1]; i++)
	{
		const vector<pair<int, int> >& connections = in2woC1[i];
		double delta = 0;
		for (int j = 0; j < connections.size(); j++)
			delta += weightS2[connections[j].first] * deltaNeuronS2[connections[j].second];
		deltaNeuronC1[i] = delta * scaleFactor * tanh_d(neuronC1[i]);
	}

	for (int i = 0; i < weightNum[2]; i++)
	{
		const vector<pair<int, int> >& connections = weight2ioC1[i];
		double diff = 0;
		for (int j = 0; j < connections.size(); j++)
			diff += neuronC1[connections[j].first] * deltaNeuronS2[connections[j].second];
		deltaWeightS2[i] += diff * scaleFactor;
	}

	for (int i = 0; i < biasNum[2]; i++)
	{
		const vector<int>& outs = bias2outC1[i];
		double diff = 0;
		for (int j = 0; j < outs.size(); j++)
			diff += deltaNeuronS2[outs[j]];
		deltaBiasS2[i] += diff;
	}

	return true;
}

bool backwardInput()
{
	initVar(deltaNeuronInput, 0.0, neuronNum[0]);
	initVar(deltaWeightC1, 0.0, weightNum[1]);
	initVar(deltaBiasC1, 0.0, biasNum[1]);

	//propagate delta to previous layer
	for (int inc = 0; inc < mapNum[0]; inc++)
		for (int outc = 0; outc < mapNum[1]; outc++)
		{
			int addr1 = getIndex(0, 0, mapNum[0] * outc + inc, 5, 5, mapNum[0] * mapNum[1]);
			int addr2 = getIndex(0, 0, outc, imgSize[1], imgSize[1], mapNum[1]);
			int addr3 = getIndex(0, 0, inc, imgSize[0], imgSize[0], mapNum[0]);

			const double* pw = &weightC1[0] + addr1;
			const double* pdeltaSrc = &deltaNeuronC1[0] + addr2;
			double* pdeltaDst = &deltaNeuronInput[0] + addr3;

			for (int y = 0; y < imgSize[1]; y++)
				for (int x = 0; x < imgSize[1]; x++)
				{
					const double* ppw = pw;
					const double ppdeltaSrc = pdeltaSrc[y * imgSize[1] + x];
					double* ppdeltaDst = pdeltaDst + y * imgSize[0] + x;

					for (int wy = 0; wy < 5; wy++)
						for (int wx = 0; wx < 5; wx++)
						{
							//ppdeltaDst[wy * imgSize[0] + wx] += *ppw * ppdeltaSrc;
							//*ppw++;
							ppdeltaDst[wy * imgSize[0] + wx] += *ppw++ * ppdeltaSrc;
						}
				}
		}

	/*for (int i = 0; i < neuronNum[0]; i++)
	deltaNeuronInput[i] *= tanh_d(neuronInput[i]);*/
	/*double identityDerivative(double x){return 1;}
	for (int i = 0; i < neuronNum[0]; i++)
	deltaNeuronInput[i] *= identityDerivative(dataSingleImage[i]);*/

	//accumulate dw
	for (int inc = 0; inc < mapNum[0]; inc++)
		for (int outc = 0; outc < mapNum[1]; outc++)
			for (int wy = 0; wy < 5; wy++)
				for (int wx = 0; wx < 5; wx++)
				{
					int addr1 = getIndex(wx, wy, inc, imgSize[0], imgSize[0], mapNum[0]);
					int addr2 = getIndex(0, 0, outc, imgSize[1], imgSize[1], mapNum[1]);
					int addr3 = getIndex(wx, wy, mapNum[0] * outc + inc, 5, 5, mapNum[0] * mapNum[1]);

					double dst = 0;
					double* pre = dataSingleImage + addr1; //&neuronInput[0]
					double* delta = &deltaNeuronC1[0] + addr2;

					for (int y = 0; y < imgSize[3]; y++)
						dst += dot(pre + y * imgSize[0], delta + y * imgSize[1], imgSize[1]);
					deltaWeightC1[addr3] += dst;
				}

	//accumulate db
	for (int outc = 0; outc < mapNum[1]; outc++)
	{
		int addr2 = getIndex(0, 0, outc, imgSize[1], imgSize[1], mapNum[1]);
		const double* delta = &deltaNeuronC1[0] + addr2;

		for (int y = 0; y < imgSize[1]; y++)
			for (int x = 0; x < imgSize[1]; x++)
				deltaBiasC1[outc] += delta[y * imgSize[1] + x];
	}

	return true;
}

void updateWeightsBias(const double* delta, double* eWeight, double* weight, int len)
{
	for (int i = 0; i < len; i++)
	{
		eWeight[i] += delta[i] * delta[i];
		weight[i] -= learningRate * delta[i] / (sqrt(eWeight[i]) + eps);
	}
}

bool updateWeights()
{
	updateWeightsBias(deltaWeightC1, eWeightC1, weightC1, weightNum[1]);
	updateWeightsBias(deltaBiasC1, eBiasC1, biasC1, biasNum[1]);
	updateWeightsBias(deltaWeightS2, eWeightS2, weightS2, weightNum[2]);
	updateWeightsBias(deltaBiasS2, eBiasS2, biasS2, biasNum[2]);
	updateWeightsBias(deltaWeightC3, eWeightC3, weightC3, weightNum[3]);
	updateWeightsBias(deltaBiasC3, eBiasC3, biasC3, biasNum[3]);
	updateWeightsBias(deltaWeightS4, eWeightS4, weightS4, weightNum[4]);
	updateWeightsBias(deltaBiasS4, eBiasS4, biasS4, biasNum[4]);
	updateWeightsBias(deltaWeightC5, eWeightC5, weightC5, weightNum[5]);
	updateWeightsBias(deltaBiasC5, eBiasC5, biasC5, biasNum[5]);
	updateWeightsBias(deltaWeightOutput, eWeightOutput, weightOutput, weightNum[6]);
	updateWeightsBias(deltaBiasOutput, eBiasOutput, biasOutput, biasNum[6]);
	return true;
}

void calc_out2wi(int inWidth, int inHeight, int outWidth, int outHeight, int outDepth, vector<vector<pair<int, int> > >& out2wi)
{
	for (int i = 0; i < outDepth; i++)
	{
		int block = inWidth * inHeight * i;
		for (int y = 0; y < outHeight; y++)
			for (int x = 0; x < outWidth; x++)
			{
				int r = y * 2;
				int c = x * 2;
				vector<pair<int, int> > wi;
				pair<int, int> p;
				for (int m = 0; m < 2; m++)
					for (int n = 0; n < 2; n++)
					{
						p.first = i;
						p.second = (r + m) * inWidth + c + n + block;
						wi.push_back(p);
					}
				out2wi.push_back(wi);
			}
	}
}

void calc_out2bias(int width, int height, int depth, vector<int>& out2bias)
{
	for (int i = 0; i < depth; i++)
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
				out2bias.push_back(i);
}

void calc_in2wo(int inWidth, int inHeight, int outWidth, int outHeight, int inDepth, int outDepth, vector<vector<pair<int, int> > >& in2wo)
{
	int len = inWidth * inHeight * inDepth;
	in2wo.resize(len);
	for (int i = 0; i < inDepth; i++)
		for (int y = 0; y < inHeight; y += 2)
			for (int x = 0; x < inWidth; x += 2)
			{
				int dyMax = min(2, inHeight - y);
				int dxMax = min(2, inWidth - x);
				int dstx = x / 2;
				int dsty = y / 2;
				for (int dy = 0; dy < dyMax; dy++)
					for (int dx = 0; dx < dxMax; dx++)
					{
						int indexIn = getIndex(x + dx, y + dy, i, inWidth, inHeight, inDepth);
						int indexOut = getIndex(dstx, dsty, i, outWidth, outHeight, outDepth);
						vector<pair<int, int> > wo;
						pair<int, int> p;
						p.first = i;
						p.second = indexOut;
						wo.push_back(p);
						in2wo[indexIn] = wo;
					}
			}
}

void calc_weight2io(int inWidth, int inHeight, int outWidth, int outHeight, int inDepth, int outDepth, vector<vector<pair<int, int> > >& weight2io)
{
	int len = inDepth;
	weight2io.resize(len);
	for (int i = 0; i < inDepth; i++)
		for (int y = 0; y < inHeight; y += 2)
			for (int x = 0; x < inWidth; x += 2)
			{
				int dyMax = min(2, inHeight - y);
				int dxMax = min(2, inWidth - x);
				int dstx = x / 2;
				int dsty = y / 2;
				for (int dy = 0; dy < dyMax; dy++)
					for (int dx = 0; dx < dxMax; dx++)
					{
						int indexIn = getIndex(x + dx, y + dy, i, inWidth, inHeight, inDepth);
						int indexOut = getIndex(dstx, dsty, i, outWidth, outHeight, outDepth);
						pair<int, int> p;
						p.first = indexIn;
						p.second = indexOut;
						weight2io[i].push_back(p);
					}
			}
}

void calc_bias2out(int inWidth, int inHeight, int outWidth, int outHeight, int inDepth, int outDepth, vector<vector<int > >& bias2out)
{
	int len = inDepth;
	bias2out.resize(len);
	for (int i = 0; i < inDepth; i++)
		for (int y = 0; y < outHeight; y++)
			for (int x = 0; x < outWidth; x++)
			{
				int outIndex = getIndex(x, y, i, outWidth, outHeight, outDepth);
				bias2out[i].push_back(outIndex);
			}
}

/*inline bool writeInt(ofstream& file, int i)
{
const char* ch = new char[4]{ i & 255, (i >> 8) & 255, (i >> 16) & 255, (i >> 24) & 255 };
file.write(ch, sizeof(int));
}*/

double test()
{
	int accuracyCnt = 0;
	for (int i = 0; i < testImagesNum; i++)
	{
		dataSingleImage = dataInputTest + i * neuronNum[0];
		dataSingleLabel = dataOutputTest + i * neuronNum[6];

		forwardC1();
		forwardS2();
		forwardC3();
		forwardS4();
		forwardC5();
		forwardOutput();

		int tInd = -1;
		int yInd = -2;
		double maxt = -INF;
		double maxy = -INF;
		for (int j = 0; j < neuronNum[6]; j++)
		{
			if (neuronOutput[j] > maxy)
			{
				maxy = neuronOutput[j];
				yInd = j;
			}
			if (dataSingleLabel[j] > maxt)
			{
				maxt = dataSingleLabel[j];
				tInd = j;
			}
		}
		if (yInd == tInd) accuracyCnt++;
	}
	return ((double)accuracyCnt / testImagesNum);
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

bool saveModelFile(string filename)
{
	ofstream file(filename, std::ios::binary);
	assert(file.is_open());
	/*writeInt(file, imgSize[0]);
	writeInt(file, imgSize[0]);
	writeInt(file, imgSize[1]);
	writeInt(file, imgSize[1]);
	writeInt(file, imgSize[2]);
	writeInt(file, imgSize[2]);
	writeInt(file, imgSize[3]);
	writeInt(file, imgSize[3]);
	writeInt(file, imgSize[4]);
	writeInt(file, imgSize[4]);
	writeInt(file, imgSize[5]);
	writeInt(file, imgSize[5]);
	writeInt(file, imgSize[6]);
	writeInt(file, imgSize[6]);
	writeInt(file, 5);
	writeInt(file, 5);
	writeInt(file, 2);
	writeInt(file, 2);
	writeInt(file, mapNum[0]);
	writeInt(file, mapNum[1]);
	writeInt(file, mapNum[2]);
	writeInt(file, mapNum[3]);
	writeInt(file, mapNum[4]);
	writeInt(file, mapNum[5]);
	writeInt(file, mapNum[6]);
	writeInt(file, weightNum[1]);
	writeInt(file, biasNum[1]);
	writeInt(file, weightNum[2]);
	writeInt(file, biasNum[2]);
	writeInt(file, weightNum[3]);
	writeInt(file, biasNum[3]);
	writeInt(file, weightNum[4]);
	writeInt(file, biasNum[4]);
	writeInt(file, weightNum[5]);
	writeInt(file, biasNum[5]);
	writeInt(file, weightNum[6]);
	writeInt(file, biasNum[6]);
	writeInt(file, neuronNum[0]);
	writeInt(file, neuronNum[1]);
	writeInt(file, neuronNum[2]);
	writeInt(file, neuronNum[3]);
	writeInt(file, neuronNum[4]);
	writeInt(file, neuronNum[5]);
	writeInt(file, neuronNum[6]);*/
	file.write((char*)(&imgSize[0]), sizeof(imgSize[0]));
	file.write((char*)(&imgSize[0]), sizeof(imgSize[0]));
	file.write((char*)(&imgSize[1]), sizeof(imgSize[1]));
	file.write((char*)(&imgSize[1]), sizeof(imgSize[1]));
	file.write((char*)(&imgSize[2]), sizeof(imgSize[2]));
	file.write((char*)(&imgSize[2]), sizeof(imgSize[2]));
	file.write((char*)(&imgSize[3]), sizeof(imgSize[3]));
	file.write((char*)(&imgSize[3]), sizeof(imgSize[3]));
	file.write((char*)(&imgSize[4]), sizeof(imgSize[4]));
	file.write((char*)(&imgSize[4]), sizeof(imgSize[4]));
	file.write((char*)(&imgSize[5]), sizeof(imgSize[5]));
	file.write((char*)(&imgSize[5]), sizeof(imgSize[5]));
	file.write((char*)(&imgSize[6]), sizeof(imgSize[6]));
	file.write((char*)(&imgSize[6]), sizeof(imgSize[6]));
	int kernel[2] = { 5, 2 };
	file.write((char*)(&kernel[0]), sizeof(kernel[0]));
	file.write((char*)(&kernel[0]), sizeof(kernel[0]));
	file.write((char*)(&kernel[1]), sizeof(kernel[1]));
	file.write((char*)(&kernel[1]), sizeof(kernel[1]));
	file.write((char*)(&mapNum[0]), sizeof(mapNum[0]));
	file.write((char*)(&mapNum[1]), sizeof(mapNum[1]));
	file.write((char*)(&mapNum[2]), sizeof(mapNum[2]));
	file.write((char*)(&mapNum[3]), sizeof(mapNum[3]));
	file.write((char*)(&mapNum[4]), sizeof(mapNum[4]));
	file.write((char*)(&mapNum[5]), sizeof(mapNum[5]));
	file.write((char*)(&mapNum[6]), sizeof(mapNum[6]));
	file.write((char*)(&weightNum[1]), sizeof(weightNum[1]));
	file.write((char*)(&biasNum[1]), sizeof(biasNum[1]));
	file.write((char*)(&weightNum[2]), sizeof(weightNum[2]));
	file.write((char*)(&biasNum[2]), sizeof(biasNum[2]));
	file.write((char*)(&weightNum[3]), sizeof(weightNum[3]));
	file.write((char*)(&biasNum[3]), sizeof(biasNum[3]));
	file.write((char*)(&weightNum[4]), sizeof(weightNum[4]));
	file.write((char*)(&biasNum[4]), sizeof(biasNum[4]));
	file.write((char*)(&weightNum[5]), sizeof(weightNum[5]));
	file.write((char*)(&biasNum[5]), sizeof(biasNum[5]));
	file.write((char*)(&weightNum[6]), sizeof(weightNum[6]));
	file.write((char*)(&biasNum[6]), sizeof(biasNum[6]));
	file.write((char*)(&neuronNum[0]), sizeof(neuronNum[0]));
	file.write((char*)(&neuronNum[1]), sizeof(neuronNum[1]));
	file.write((char*)(&neuronNum[2]), sizeof(neuronNum[2]));
	file.write((char*)(&neuronNum[3]), sizeof(neuronNum[3]));
	file.write((char*)(&neuronNum[4]), sizeof(neuronNum[4]));
	file.write((char*)(&neuronNum[5]), sizeof(neuronNum[5]));
	file.write((char*)(&neuronNum[6]), sizeof(neuronNum[6]));
	file.write((char*)weightC1, sizeof(weightC1));
	file.write((char*)biasC1, sizeof(biasC1));
	file.write((char*)weightS2, sizeof(weightS2));
	file.write((char*)biasS2, sizeof(biasS2));
	file.write((char*)weightC3, sizeof(weightC3));
	file.write((char*)biasC3, sizeof(biasC3));
	file.write((char*)weightS4, sizeof(weightS4));
	file.write((char*)biasS4, sizeof(biasS4));
	file.write((char*)weightC5, sizeof(weightC5));
	file.write((char*)biasC5, sizeof(biasC5));
	file.write((char*)weightOutput, sizeof(weightOutput));
	file.write((char*)biasOutput, sizeof(biasOutput));
	file.close();
	return true;
}

bool train()
{
	out2wiS2.clear();
	out2biasS2.clear();
	out2wiS4.clear();
	out2biasS4.clear();
	in2woC3.clear();
	weight2ioC3.clear();
	bias2outC3.clear();
	in2woC1.clear();
	weight2ioC1.clear();
	bias2outC1.clear();

	calc_out2wi(imgSize[1], imgSize[1], imgSize[2], imgSize[2], mapNum[2], out2wiS2);
	calc_out2bias(imgSize[2], imgSize[2], mapNum[2], out2biasS2);
	calc_out2wi(imgSize[3], imgSize[3], imgSize[4], imgSize[4], mapNum[4], out2wiS4);
	calc_out2bias(imgSize[4], imgSize[4], mapNum[4], out2biasS4);
	calc_in2wo(imgSize[3], imgSize[3], imgSize[4], imgSize[4], mapNum[3], mapNum[4], in2woC3);
	calc_weight2io(imgSize[3], imgSize[3], imgSize[4], imgSize[4], mapNum[3], mapNum[4], weight2ioC3);
	calc_bias2out(imgSize[3], imgSize[3], imgSize[4], imgSize[4], mapNum[3], mapNum[4], bias2outC3);
	calc_in2wo(imgSize[1], imgSize[1], imgSize[2], imgSize[2], mapNum[1], mapNum[2], in2woC1);
	calc_weight2io(imgSize[1], imgSize[1], imgSize[2], imgSize[2], mapNum[1], mapNum[2], weight2ioC1);
	calc_bias2out(imgSize[1], imgSize[1], imgSize[2], imgSize[2], mapNum[1], mapNum[2], bias2outC1);

	int iter = 0;
	for (int iter = 0; iter < epochNum; iter++)
	{
		cout << "epoch: " << iter + 1 << endl;
		for (int i = 0; i < trainImagesNum; i++)
		{
			if ((i + 1) % 1000 == 0) cout << "epoch: " << iter + 1 << ", trained images number: " << i + 1 << "/" << trainImagesNum << endl;

			dataSingleImage = dataInputTrain + i * neuronNum[0];
			dataSingleLabel = dataOutputTrain + i * neuronNum[6];

			forwardC1();
			forwardS2();
			forwardC3();
			forwardS4();
			forwardC5();
			forwardOutput();
			backwardOutput();
			backwardC5();
			backwardS4();
			backwardC3();
			backwardS2();
			backwardC1();
			backwardInput();
			updateWeights();
		}
		double testAccuracyRate = test();
		cout << "Accuracy rate: " << testAccuracyRate << endl;

		//save temp file, with test accuracy as suffix of file name
		int testAccuracyNum = (int)(testAccuracyRate * testImagesNum);
		string testmodelName = "./model/cnnmodel";
		if (testAccuracyNum < 1000) testmodelName += "0";
		if (testAccuracyNum < 100) testmodelName += "0";
		if (testAccuracyNum < 10) testmodelName += "0";
		testmodelName = testmodelName + to_string(testAccuracyNum) + ".dat";
		saveModelFile(testmodelName);

		if (testAccuracyRate > accuracyRate)
		{
			saveModelFile("cnnmodel.dat");
			cout << "generate CNN model." << endl;
			break;
		}
		if (iter == epochNum)
		{
			saveModelFile("cnnmodel.dat");
			cout << "generate CNN model." << endl;
		}
	}

	return true;
}

int testTrain()
{
	init();
	train();
	release();
	return 0;
}

bool readModelFile(string filename)
{
	ifstream file(filename, std::ios::binary);
	assert(file.is_open());
	int imgsize[7];
	file.read((char*)(&imgsize[0]), sizeof(imgSize[0]));
	file.read((char*)(&imgsize[0]), sizeof(imgSize[0]));
	file.read((char*)(&imgsize[1]), sizeof(imgSize[1]));
	file.read((char*)(&imgsize[1]), sizeof(imgSize[1]));
	file.read((char*)(&imgsize[2]), sizeof(imgSize[2]));
	file.read((char*)(&imgsize[2]), sizeof(imgSize[2]));
	file.read((char*)(&imgsize[3]), sizeof(imgSize[3]));
	file.read((char*)(&imgsize[3]), sizeof(imgSize[3]));
	file.read((char*)(&imgsize[4]), sizeof(imgSize[4]));
	file.read((char*)(&imgsize[4]), sizeof(imgSize[4]));
	file.read((char*)(&imgsize[5]), sizeof(imgSize[5]));
	file.read((char*)(&imgsize[5]), sizeof(imgSize[5]));
	file.read((char*)(&imgsize[6]), sizeof(imgSize[6]));
	file.read((char*)(&imgsize[6]), sizeof(imgSize[6]));
	int kernel[2];
	file.read((char*)(&kernel[0]), sizeof(int));
	file.read((char*)(&kernel[0]), sizeof(int));
	file.read((char*)(&kernel[1]), sizeof(int));
	file.read((char*)(&kernel[1]), sizeof(int));
	int mapnum[7];
	file.read((char*)(&mapnum[0]), sizeof(mapNum[0]));
	file.read((char*)(&mapnum[1]), sizeof(mapNum[1]));
	file.read((char*)(&mapnum[2]), sizeof(mapNum[2]));
	file.read((char*)(&mapnum[3]), sizeof(mapNum[3]));
	file.read((char*)(&mapnum[4]), sizeof(mapNum[4]));
	file.read((char*)(&mapnum[5]), sizeof(mapNum[5]));
	file.read((char*)(&mapnum[6]), sizeof(mapNum[6]));
	int weightnum[7], biasnum[7];
	file.read((char*)(&weightnum[1]), sizeof(weightnum[1]));
	file.read((char*)(&biasnum[1]), sizeof(biasnum[1]));
	file.read((char*)(&weightnum[2]), sizeof(weightnum[2]));
	file.read((char*)(&biasnum[2]), sizeof(biasnum[2]));
	file.read((char*)(&weightnum[3]), sizeof(weightnum[3]));
	file.read((char*)(&biasnum[3]), sizeof(biasnum[3]));
	file.read((char*)(&weightnum[4]), sizeof(weightnum[4]));
	file.read((char*)(&biasnum[4]), sizeof(biasnum[4]));
	file.read((char*)(&weightnum[5]), sizeof(weightnum[5]));
	file.read((char*)(&biasnum[5]), sizeof(biasnum[5]));
	file.read((char*)(&weightnum[6]), sizeof(weightnum[6]));
	file.read((char*)(&biasnum[6]), sizeof(biasnum[6]));
	int neuronnum[7];
	file.read((char*)(&neuronnum[0]), sizeof(neuronnum[0]));
	file.read((char*)(&neuronnum[1]), sizeof(neuronnum[1]));
	file.read((char*)(&neuronnum[2]), sizeof(neuronnum[2]));
	file.read((char*)(&neuronnum[3]), sizeof(neuronnum[3]));
	file.read((char*)(&neuronnum[4]), sizeof(neuronnum[4]));
	file.read((char*)(&neuronnum[5]), sizeof(neuronnum[5]));
	file.read((char*)(&neuronnum[6]), sizeof(neuronnum[6]));
	file.read((char*)weightC1, sizeof(weightC1));
	file.read((char*)biasC1, sizeof(biasC1));
	file.read((char*)weightS2, sizeof(weightS2));
	file.read((char*)biasS2, sizeof(biasS2));
	file.read((char*)weightC3, sizeof(weightC3));
	file.read((char*)biasC3, sizeof(biasC3));
	file.read((char*)weightS4, sizeof(weightS4));
	file.read((char*)biasS4, sizeof(biasS4));
	file.read((char*)weightC5, sizeof(weightC5));
	file.read((char*)biasC5, sizeof(biasC5));
	file.read((char*)weightOutput, sizeof(weightOutput));
	file.read((char*)biasOutput, sizeof(biasOutput));
	file.close();

	out2wiS2.clear();
	out2biasS2.clear();
	out2wiS4.clear();
	out2biasS4.clear();
	calc_out2wi(imgSize[1], imgSize[1], imgSize[2], imgSize[2], mapNum[2], out2wiS2);
	calc_out2bias(imgSize[2], imgSize[2], mapNum[2], out2biasS2);
	calc_out2wi(imgSize[3], imgSize[3], imgSize[4], imgSize[4], mapNum[4], out2wiS4);
	calc_out2bias(imgSize[4], imgSize[4], mapNum[4], out2biasS4);

	return true;
}

int predict(const uchar* data, int width, int height)
{
	assert(data && width == imgSize[0] && height == imgSize[0]);
	const double scaleMin = -1;
	const double scaleMax = 1;
	double* tmp = new double[width * height];
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			tmp[y * width + x] = (data[y * width + x] / 255) * (scaleMax - scaleMin) + scaleMin;
	dataSingleImage = &tmp[0];

	forwardC1();
	forwardS2();
	forwardC3();
	forwardS4();
	forwardC5();
	forwardOutput();

	int pos = -1;
	double maxValue = -INF;
	for (int i = 0; i < neuronNum[6]; i++)
		if (neuronOutput[i] > maxValue)
		{
			maxValue = neuronOutput[i];
			pos = i;
		}

	return pos;
}

int testPredict()
{
	init();
	bool flag = readModelFile("cnnmodel.dat");
	if (!flag)
	{
		cout << "read cnnmodel.dat error!" << endl;
		return -1;
	}

	int width = 32, height = 32;
	/*vector<int> target{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };*/
	int keepRunning = 1;
	string imgPath = "";
	while (keepRunning)
	{
		cout << "Input file name with extension part (input -1 to exit):" << endl;
		cin >> imgPath;
		/*for (auto i : target)
		{
		imgPath = to_string(i) + ".jpg";

		Mat src = imread(imgPath, 0);
		if (src.data == NULL)
		{
		fprintf(stderr, "read image error: %s\n", imgPath.c_str());
		return -1;
		}
		Mat tmp(src.rows, src.cols, CV_8UC1, Scalar::all(255));
		subtract(tmp, src, tmp);
		resize(tmp, tmp, Size(width, height));
		int ret = predict(tmp.data, width, height);
		cout << "The actural digit is: " << ret << ", correct digit is: " << i << endl;
		}*/

		if (imgPath == "-1")
		{
			keepRunning = 0;
			break;
		}
		Mat src = imread(imgPath, 0);
		if (src.data == NULL)
		{
			/*fprintf(stderr, "read image error: %s\n", imgPath.c_str());
			keepRunning = 0;
			return -1;*/
			cout << "Wrong file name! Please input correct file name." << endl;
			continue;
		}
		Mat tmp(src.rows, src.cols, CV_8UC1, Scalar::all(255));
		subtract(tmp, src, tmp);
		resize(tmp, tmp, Size(width, height));
		int ret = predict(tmp.data, width, height);
		/*cout << "The actural digit is: " << ret << ", correct digit is: " << i << endl;*/
		cout << "The predict digit is: " << ret << endl;
	}
	release();

	return 0;
}

#endif //CNN_HPP
