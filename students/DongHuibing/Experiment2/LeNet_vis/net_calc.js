//amounts
var nConvFilters1 = 6;
var nConvFilters2 = 16;

var filterSize = 5;
var halfFilterSize = 2;

var nConvNodes1 = 28 * 28 * nConvFilters1;//32 - 5 + 1 == 28
var nPoolNodes1 = 14 * 14 * nConvFilters1;
var nConvNodes2 = 10 * 10 * nConvFilters2;//14 - 5 + 1 == 10
var nPoolNodes2 = 5 * 5 * nConvFilters2;

var nFcNodes1 = 120;
var nFcNodes2 = 100;


//weights
var input, inputTranspose;
var whtConv1, whtConv2;//whtConv1[i][j][k]: i_th filter, weight[j][k]
var whtFc1, whtFc2;
var whtOut;
var allOut;

var normWhtConv1, normWhtConv2, normWhtFc1, normWhtFc2, normWhtOut;

var isKept;//isKept[i][j]==1 if conv2_i and con1_j should be calculated

var hasStarted = false;


loadWeights();
forwardProp();

function mat2Arr(mat)
{
	var ele = mat.elements;
	var res = ele.map(function(arr) {
		return arr.slice();
	})
	return res;
}

function normalize2DArr(arr)
{
	var minVal = 1000
	var maxVal = -1000;

	var rows = arr.length;
	var cols = arr[0].length;

	for(var i = 0; i < rows; ++i)
	{
		for(var j = 0; j < cols; ++j)
		{
			minVal = Math.min(minVal, arr[i][j]);
			maxVal = Math.max(maxVal, arr[i][j]);
		}
	}
	var diff = maxVal - minVal;

	for(var i = 0; i < rows; ++i)
	{
		for(var j = 0; j < cols; ++j)
		{
			arr[i][j] = (arr[i][j] - minVal) / diff;
		}
	}

	return arr;
}

function normalize1DArr(arr)
{
	var minVal = 1000
	var maxVal = -1000;

	for(var i = 0; i < arr.length; ++i)
	{
		minVal = Math.min(minVal, arr[i]);
		maxVal = Math.max(maxVal, arr[i]);
	}
	var diff = maxVal - minVal;

	for(var i = 0; i < arr.length; ++i)
		arr[i] = (arr[i] - minVal) / diff;

	return arr;
}

function normalize3DArr(arr)
{
	var minVal = 1000;
	var maxVal = -1000;
	for(var i = 0; i < arr.length; ++i)
		for(var x = 0; x < arr[0].length; ++x)
			for(var y = 0; y < arr[0][0].length; ++y)
			{
				minVal = Math.min(minVal, arr[i][x][y]);
				maxVal = Math.max(maxVal, arr[i][x][y]);
			}
	var diff = maxVal - minVal;

	for(var i = 0; i < arr.length; ++i)
		for(var x = 0; x < arr[0].length; ++x)
			for(var y = 0; y < arr[0][0].length; ++y)
				arr[i][x][y] = (arr[i][x][y] - minVal) / diff;

	return arr;
}

function copy2DArr(ori)
{
	var res = new Array(ori.length);
	for(i = 0; i < res.length; ++i)
	{
		res[i] = new Array(ori[i].length);
		for(var j = 0; j < res[i].length; ++j) res[i][j] = ori[i][j];
	}
	return res;
}

function loadWeights()
{
	whtConv1 = new Array(nConvFilters1);
	normWhtConv1 = new Array(nConvFilters1);
	for(var i = 0; i < whtConv1.length; ++i)
	{
		whtConv1[i] = mat2Arr(conv_nodes[0][i]);
		normWhtConv1[i] = copy2DArr(whtConv1[i]);
		normWhtConv1[i] = normalize2DArr(normWhtConv1[i]);
	}

	isKept = mat2Arr(keepers.transpose());

	whtConv2 = new Array(nConvFilters2);
	normWhtConv2 = new Array(nConvFilters2);
	for(var i = 0; i < nConvFilters2; ++i)
	{
		whtConv2[i] = new Array(nConvFilters1);
		normWhtConv2[i] = new Array(nConvFilters1);
		var cnt = 0;
		for(var j = 0; j < nConvFilters1; ++j)
		{
			if(isKept[i][j])
			{
				whtConv2[i][j] = mat2Arr(conv_nodes[1][i][cnt++]);
				normWhtConv2[i][j] = copy2DArr(whtConv2[i][j]);
				normWhtConv2[i][j] = normalize2DArr(normWhtConv2[i][j]);
			}
		}
	}

	whtFc1 = mat2Arr(hidden_weights_1);
	normWhtFc1 = copy2DArr(whtFc1);
	normWhtFc1 = normalize2DArr(normWhtFc1);

	whtFc2 = mat2Arr(hidden_weights_2);
	normWhtFc2 = copy2DArr(whtFc2);
	normWhtFc2 = normalize2DArr(normWhtFc2);

	whtOut = mat2Arr(final_weights);
	normWhtOut = copy2DArr(whtOut);
	normWhtOut = normalize2DArr(normWhtOut);
}

function maxPooling(arr, inSize, stride)
{
	var outSize = inSize / stride;
	var res = new Array(outSize);
	for(var i = 0, i1 = 0; i1 < outSize; i += stride, ++i1)
	{
		res[i1] = valArr(outSize, -1000);
		for(var j = 0, j1 = 0; j1 < outSize; j += stride, ++j1)
			{
				for(var k = 0; k < stride; ++k)
					res[i1][j1] = Math.max(res[i1][j1], arr[i+k][j+k]);
			}
	}
	return res;
}

function convolution(input, inSize, out, outSize, wht, filterSize, init)
{
	if(init == 1)
	{
		out = new Array(outSize);
		for(var i = 0; i < outSize; ++i) out[i] = valArr(outSize, 0);
	}
	for(var i = halfFilterSize; i+halfFilterSize < inSize; ++i)
	{
		for(var j = halfFilterSize; j+halfFilterSize < inSize; ++j)
			for(var x = -halfFilterSize; x <= halfFilterSize; ++x)
				for(var y = -halfFilterSize; y <= halfFilterSize; ++y)
				{
					out[i-halfFilterSize][j-halfFilterSize] += input[i+x][j+y] * wht[halfFilterSize+x][halfFilterSize+y];
				}
	}
	return out;
}

function activate2D(arr, bias)
{
	for(var i = 0; i < arr.length; ++i)
		for(var j = 0; j < arr[0].length; ++j)
			arr[i][j] = 1.7159*math.tanh(0.666667 * (arr[i][j] + bias));
	return arr;
}

function activate1D(arr, bias)
{
	for(var i = 0; i < arr.length; ++i)
		arr[i][j] = 1.7159*math.tanh(0.666667 * (arr[i] + bias));
	return arr;
}

function activate(val, bias)
{
	return 1.7159*math.tanh(0.666667 * (val + bias));
}

function valArr(len, val)
{
	var arr = new Array(len);
	for(var i = 0; i < len; ++i) arr[i] = val;
	return arr;
}

function out2D(arr)
{
	for(var i = 0; i < arr.length; ++i)
	{
		var s = "";
		for(var j = 0; j < arr[0].length; ++j)
		{
			// s = s + (( parseFloat(arr[i][j]).toFixed(1)) > 0 ? '*' : ' ') + ' ';
			s = s + (( parseFloat(arr[i][j]).toFixed(1)) ) + ' ';
		}
		console.log(s);
	}
}

function out1D(arr)
{
	var s = "";
	for(var i = 0; i < arr.length; ++i)
			s = s + (( parseFloat(arr[i]).toFixed(1)) > 0 ? '*' : ' ') + (i % 10 == 0 ? '\n' : ' ');
	console.log(s);
}

function val3DArr(num, len, val)
{
	var arr = new Array(num);
	for(var i = 0; i < num; ++i)
	{
		arr[i] = new Array(len);
		for(var j = 0; j < len; ++j) arr[i][j] = valArr(len, val);
	}
	return arr;
}

function forwardProp()
{
	// console.log("keeper");
	// out2D(isKept);
	var canvas2 = document.getElementById('canvas2');
	var ctx = canvas2.getContext('2d');
	var imgData = (ctx.getImageData(0, 0, 28, 28)).data;//in RGBA order
	var input28Linear = new Array(784);
	for(var i = 3, j = 0; i < imgData.length; i += 4)
		input28Linear[j++] = ((imgData[i]/255)*1.275)-0.1;
	input = new Array(32);

	for(var i = 0; i < 32; ++i) input[i] = valArr(32, -0.1);
	for(var i = 0; i < 28; ++i)
		for(var j = 0; j < 28; ++j)
			input[i+2][j+2] = input28Linear[i*28+j];
	// console.log("my input");
	// out2D(input);
	inputTranspose = Matrix.create(input).transpose().elements;
	if(!hasStarted)
	{
		hasStarted = true;
		allOut = new Array(7);
		allOut[0] = new val3DArr(nConvFilters1, 28, 0);
		allOut[1] = new val3DArr(nConvFilters1, 14, 0);
		allOut[2] = new val3DArr(nConvFilters2, 10, 0);
		allOut[3] = new val3DArr(nConvFilters2, 5, 0);
		allOut[4] = valArr(120, 0);
		allOut[5] = valArr(100, 0);
		allOut[6] = valArr(10, 0);
		return;
	}

	allOut = new Array(7);
	var inSize = 32;
	var outSize = 28;
	var out;
	var wht;

	//allOut[0]: conv1
	allOut[0] = new Array(nConvFilters1);
	for(var f = 0; f < nConvFilters1; ++f)
	{
		allOut[0][f] = convolution(input, inSize, allOut[0][f], outSize, whtConv1[f], 5, 1);
		allOut[0][f] = activate2D(allOut[0][f], conv_biases_1.e(f+1));
	}

	// console.log("my conv1");
	// out2D(allOut[0][0]);

	//allOut[1]: pool1
	allOut[1] = new Array(nConvFilters1);
	for(var f = 0; f < nConvFilters1; ++f)
		allOut[1][f] = maxPooling(allOut[0][f], 28, 2);

	// console.log("my pool1");
	// out2D(allOut[1][0]);
	//allOut[2]: conv2
	allOut[2] = new Array(nConvFilters2);
	inSize = 14;
	outSize = 10;
	for(var f1 = 0; f1 < nConvFilters2; ++f1)
	{
		allOut[2][f1] = new Array(outSize);
		for(var i = 0; i < outSize; ++i)
			allOut[2][f1][i] = valArr(outSize, 0);
	}
	for(var f1 = 0; f1 < nConvFilters2; ++f1)
	{
		for(var f2 = 0; f2 < nConvFilters1; ++f2)
		{
			if(isKept[f1][f2])
				allOut[2][f1] = convolution(allOut[1][f2], 14, allOut[2][f1], 10, whtConv2[f1][f2], 5, 0);
		}
		allOut[2][f1] = activate2D(allOut[2][f1], conv_biases_2.e(f1+1));
	}
	// console.log("my conv2");
	// out2D(allOut[2][0]);

	//allOut[3]: pool2
	allOut[3] = new Array(nConvFilters2);
	for(var f = 0; f < nConvFilters2; ++f) allOut[3][f] = maxPooling(allOut[2][f], 10, 2);
	// console.log("my pool2");
	// out2D(allOut[3][0]);


	//allOut[4]: fc1
	allOut[4] = valArr(nFcNodes1, 0);
	for(var i = 0; i < nFcNodes1; ++i)
	{
		var cnt = 0;
		for(var j = 0; j < nConvFilters2; ++j)
			for(var x = 0; x < filterSize; ++x)
				for(var y = 0; y < filterSize; ++y)
					allOut[4][i] += allOut[3][j][y][x] * whtFc1[i][cnt++];
		allOut[4][i] = activate(allOut[4][i], hidden_biases_1.e(i+1));
	}
	// console.log("my fc1");
	// out1D(allOut[4]);

	//allOut[5]: fc2
	allOut[5] = valArr(nFcNodes2, 0);
	for(var i = 0; i < nFcNodes2; ++i)
	{
		for(var  j = 0; j < nFcNodes1; ++j)
			allOut[5][i] += allOut[4][j] * whtFc2[i][j];
		allOut[5][i] = activate(allOut[5][i], hidden_biases_2.e(i+1));
	}
	// console.log("my fc2");
	// out1D(allOut[5]);


	//allOut[6]: final
	allOut[6] = valArr(10, 0);
	for(var i = 0; i < 10; ++i)
	{
		for(var j = 0; j < nFcNodes2; ++j)
			allOut[6][i] += allOut[5][j] * whtOut[i][j];
		allOut[6][i] = activate(allOut[6][i], final_biases.e(i + 1));
	}
	var s = "";
	for(var i = 0; i < 10; ++i) s = s + allOut[6][i].toFixed(1) + ' ';
	console.log("my final");
	console.log(s);


	allOut[0] = normalize3DArr(allOut[0], nConvFilters1, 28);
	allOut[1] = normalize3DArr(allOut[1], nConvFilters1, 14);
	allOut[2] = normalize3DArr(allOut[2], nConvFilters1, 10);
	allOut[3] = normalize3DArr(allOut[3], nConvFilters1, 5);
	allOut[4] = normalize1DArr(allOut[4]);
	allOut[5] = normalize1DArr(allOut[5]);
	allOut[6] = normalize1DArr(allOut[6]);
	// console.log("10 numbers p:");
	// for(var i = 0; i < 10; ++i) console.log(i + '\t' + allOut[6][i].toFixed(2));

}
