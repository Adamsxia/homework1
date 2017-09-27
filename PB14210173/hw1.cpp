// homework1.cpp : 定义控制台应用程序的入口点。
//

#include "afx.h"
#include <opencv2/opencv.hpp>
#include <time.h>
#include   <iostream>   
using   namespace   std;
using namespace cv;

#define IMG_SHOW
#define SUB_IMAGE_MATCH_OK 1
#define SUB_IMAGE_MATCH_FAIL -1

float SqrtByRSQRTSS(float a);
float InvSqrt(float x);
//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (bgrImg.data == NULL)
	{
		printf("Image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;
	for (int row_i = height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - 1; col_j >= 0; col_j--)
		{
			int b = bgrImg.data[3 * (row_i * width + col_j) + 0];
			int g = bgrImg.data[3 * (row_i * width + col_j) + 1];
			int r = bgrImg.data[3 * (row_i * width + col_j) + 2];

			int grayVal = (b * 30 + 150 * g + 76 * r) >> 8;
			grayImg.data[row_i * width + col_j] = grayVal;
		}
	}

#ifdef IMG_SHOW
	namedWindow("grayImg", WINDOW_NORMAL);
	imshow("grayImag", grayImg);
	waitKey(0);
#endif

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据灰度图像计算梯度图像
//grayImg：灰度图，单通道
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (grayImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	//x方向的梯度
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);

	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			int grad_x =
				grayImg.data[(row_i - 1) * width + col_j + 1]
				+ 2 * grayImg.data[(row_i)* width + col_j + 1]
				+ grayImg.data[(row_i + 1)* width + col_j + 1]
				- grayImg.data[(row_i - 1) * width + col_j - 1]
				- 2 * grayImg.data[(row_i)* width + col_j - 1]
				- grayImg.data[(row_i + 1)* width + col_j - 1];

			((float*)gradImg_x.data)[row_i * width + col_j] = grad_x;

			int grad_y =
				grayImg.data[(row_i + 1) * width + col_j - 1]
				+ 2 * grayImg.data[(row_i + 1) * width + col_j]
				+ grayImg.data[(row_i + 1) * width + col_j + 1]
				- grayImg.data[(row_i - 1) * width + col_j - 1]
				- 2 * grayImg.data[(row_i + 1) * width + col_j]
				- grayImg.data[(row_i + 1) * width + col_j + 1];

			((float*)gradImg_y.data)[row_i * width + col_j] = grad_y;
		}
	}

#ifdef IMG_SHOW
	Mat gradImg_x_8U(height, width, CV_8UC1);
	Mat gradImg_y_8U(height, width, CV_8UC1);
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			int val_x = ((float*)gradImg_x.data)[row_i * width + col_j];
			int val_y = ((float*)gradImg_y.data)[row_i * width + col_j];
			gradImg_x_8U.data[row_i * width + col_j] = abs(val_x);
			gradImg_y_8U.data[row_i * width + col_j] = abs(val_y);
		}
	}

	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", gradImg_x_8U);
	waitKey(100);

	namedWindow("gradImg_y_8U", 0);
	imshow("gradImg_y_8U", gradImg_y_8U);
	waitKey(100);
#endif

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据水平和垂直梯度，计算角度和幅值图
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//angleImg：角度图，浮点类型图像，CV32FC1
//magImg：幅值图，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (gradImg_x.data == NULL || gradImg_y.data == NULL)
	{
		printf("data is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;

	//计算角度图
	angleImg.setTo(0);

	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j++)
		{
			float grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
			float grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];
			float angle = atan2(grad_y, grad_x);
			if (angle > 0.0)
			{
				angle = 180 * angle / CV_PI;
			}
			else
			{
				angle = 360 + 180 * angle / CV_PI;
			}
			((float*)angleImg.data)[row_i * width + col_j] = angle;

			float mag = SqrtByRSQRTSS(grad_y * grad_y + grad_x * grad_x);
			((float*)magImg.data)[row_i * width + col_j] = mag;
		}
	}

#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j++)
		{
			//angleImg_8U.data[row_i * width + col_j] = 0.5 * angleImg.data[row_i * width + col_j];
			float angle = ((float*)angleImg.data)[row_i * width + col_j];
			angle *= 180 / CV_PI;
			angle += 180;
			//为了能在8U上显示，缩小到0-180之间
			angle /= 2;
			angleImg_8U.data[row_i * width + col_j] = angle;
		}
	}

	namedWindow("angleImg_8U", 0);
	imshow("angleImg_8U", angleImg_8U);
	waitKey(100);

	namedWindow("magImg", 0);
	imshow("magImg", magImg);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像进行二值化
//grayImg：灰度图，单通道
//binaryImg：二值图，单通道
//th：二值化阈值，高于此值，255，低于此值0
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (grayImg.data == NULL)
	{
		printf("image is NULL\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int heigth = grayImg.rows;

	for (int row_i = heigth - 1; row_i >= 0; row_i--)
	{
		int temp0 = row_i * width;
		for (int col_j = width - 1; col_j >= 0; col_j--)
		{
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			int dstVal = 0;
			if (pixVal > th)
			{
				dstVal = 255;
			}
			else
			{
				dstVal = 0;
			}
			binaryImg.data[temp1] = dstVal;
		}
	}
#ifdef IMG_SHOW
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像计算直方图
//grayImg：灰度图，单通道
//hist：直方图
//hist_len：直方图的亮度等级，直方图数组的长度
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (grayImg.data == NULL || hist == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;

	for (int i = hist_len - 1; i >= 0; i--)
	{
		hist[i] = 0;
	}

	for (int row_i = height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - 1; col_j >= 0; col_j--)
		{
			int pixVal = grayImg.data[row_i * width + col_j];
			hist[pixVal]++;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用亮度进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == NULL || subImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	int sum = INT_MAX;
	if (width <= sub_width || height <= sub_height)
	{
		printf("subgraph is bigger than original picture!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	//Mat searchImg(height, width, CV_32FC1);
	//searchImg.setTo(FLT_MAX);

	for (int row_i = height - sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - sub_width - 1; col_j >= 0; col_j--)
		{
			int total_diff = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; sub_row_i--)
			{
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; sub_col_j--)
				{
					int row_index = row_i + sub_row_i;
					int col_index = col_j + sub_col_j;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					int template_pix = subImg.data[sub_row_i * sub_width + sub_col_j];
					total_diff += abs(bigImg_pix - template_pix);
				}
			}
			//((float*)searchImg.data)[row_i * width + col_j] = total_diff;
			if (total_diff < sum)
			{
				*x = col_j + 1;
				*y = row_i + 1;
				sum = total_diff;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用色彩进行子图匹配
//colorImg：彩色图，三通单
//subImg：模板子图，三通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (colorImg.data == NULL || subImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	int sum = INT_MAX;
	if (width <= sub_width || height <= sub_height)
	{
		printf("subgraph is bigger than original picture!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	//Mat searchImg(height, width, CV_32FC1);
	//searchImg.setTo(FLT_MAX);

	for (int row_i = height - sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - sub_width - 1; col_j >= 0; col_j--)
		{
			int total_diff = 0;
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; sub_row_i--)
			{
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; sub_col_j--)
				{
					int row_index = row_i + sub_row_i;
					int col_index = col_j + sub_col_j;
					int bigImg_pix_b = colorImg.data[3 * (row_index * width + col_index) + 0];
					int bigImg_pix_g = colorImg.data[3 * (row_index * width + col_index) + 1];
					int bigImg_pix_r = colorImg.data[3 * (row_index * width + col_index) + 2];
					int template_pix_b = subImg.data[3 * (sub_row_i * sub_width + sub_col_j) + 0];
					int template_pix_g = subImg.data[3 * (sub_row_i * sub_width + sub_col_j) + 1];
					int template_pix_r = subImg.data[3 * (sub_row_i * sub_width + sub_col_j) + 2];
					total_diff += (abs(bigImg_pix_b - template_pix_b) + abs(bigImg_pix_g - template_pix_g) + abs(bigImg_pix_r - template_pix_r));
				}
			}
			//((float*)searchImg.data)[row_i * width + col_j] = total_diff;
			if (total_diff < sum)
			{
				*x = col_j + 1;
				*y = row_i + 1;
				sum = total_diff;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用亮度相关性进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == NULL || subImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	float rtag = 0;
	if (width <= sub_width || height <= sub_height)
	{
		printf("subgraph is bigger than original picture!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	//Mat searchImg(height, width, CV_32FC1);
	//searchImg.setTo(FLT_MAX);

	for (int row_i = height - sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - sub_width - 1; col_j >= 0; col_j--)
		{
			float rtemp = 0;
			int product_sum = 0;//乘积和
			int square_sum_bigImg = 0;//大图灰度值平方和
			int square_sum_subImg = 0;//模板子图灰度值平方和
			for (int sub_row_i = sub_height - 1; sub_row_i >= 0; sub_row_i--)
			{
				for (int sub_col_j = sub_width - 1; sub_col_j >= 0; sub_col_j--)
				{
					int row_index = row_i + sub_row_i;
					int col_index = col_j + sub_col_j;
					int bigImg_pix = grayImg.data[row_index * width + col_index];//大图灰度值
					int template_pix = subImg.data[sub_row_i * sub_width + sub_col_j];//模板子图灰度值
					product_sum += bigImg_pix*template_pix;
					square_sum_bigImg += bigImg_pix*bigImg_pix;
					square_sum_subImg += template_pix*template_pix;
					//total_diff += abs(bigImg_pix - template_pix);
				}
			}
			//((float*)searchImg.data)[row_i * width + col_j] = total_diff;
			rtemp = product_sum * InvSqrt(square_sum_bigImg * square_sum_subImg);
			if (rtemp > rtag)
			{
				*x = col_j + 1;
				*y = row_i + 1;
				rtag = rtemp;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用角度值进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	*x = *y = -1;

	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int sub_nCols = subImg.cols;
	int sub_nRows = subImg.rows;

	if (sub_nCols > nCols || sub_nRows > nRows)
	{
		cout << "subImg is too big." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (sub_nCols == nCols && sub_nRows == nRows)
	{
		*x = 0;
		*y = 0;
		return SUB_IMAGE_MATCH_OK;
	}

	Mat grad_x(nRows, nCols, CV_32FC1);
	Mat grad_y(nRows, nCols, CV_32FC1);
	Mat angleImg(nRows, nCols, CV_32FC1);
	Mat magImg(nRows, nCols, CV_32FC1);

	Mat sub_grad_x(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_grad_y(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_angleImg(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_magImg(sub_nRows, sub_nCols, CV_32FC1);

	ustc_CalcGrad(grayImg, grad_x, grad_y);
	ustc_CalcAngleMag(grad_x, grad_y, angleImg, magImg);

	ustc_CalcGrad(subImg, sub_grad_x, sub_grad_y);
	ustc_CalcAngleMag(sub_grad_x, sub_grad_y, sub_angleImg, sub_magImg);

	float *data = (float *)angleImg.data;
	float *sub_data = (float *)sub_angleImg.data;

	float min_err = FLT_MAX;

	for (int i = 0; i < nRows - sub_nRows; i++)
	{
		for (int j = 0; j < nCols - sub_nCols; j++)
		{
			float total_err = 0;
			for (int sub_i = 1; sub_i < sub_nRows - 1; sub_i++)
			{
				for (int sub_j = 1; sub_j < sub_nCols - 1; sub_j++)
				{
					int index = (i + sub_i)*nCols + (j + sub_j);
					int sub_index = sub_i*sub_nCols + sub_j;
					int err = data[index] - sub_data[sub_index];
					total_err += (err & 0x80000000) ? -err : err;
				}
			}

			if (min_err > total_err) {
				min_err = total_err;
				*x = i;
				*y = j;
			}
		}
	}

	if (*x == -1 == *y) {
		cout << "match failed." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	else
	{
		return SUB_IMAGE_MATCH_OK;
	}
}

//函数功能：利用幅值进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	*x = *y = -1;

	int nCols = grayImg.cols;
	int nRows = grayImg.rows;
	int sub_nCols = subImg.cols;
	int sub_nRows = subImg.rows;

	if (sub_nCols > nCols || sub_nRows > nRows)
	{
		cout << "subImg is too big." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if (sub_nCols == nCols && sub_nRows == nRows)
	{
		*x = 0;
		*y = 0;
		return SUB_IMAGE_MATCH_OK;
	}

	Mat grad_x(nRows, nCols, CV_32FC1);
	Mat grad_y(nRows, nCols, CV_32FC1);
	Mat angleImg(nRows, nCols, CV_32FC1);
	Mat magImg(nRows, nCols, CV_32FC1);

	Mat sub_grad_x(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_grad_y(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_angleImg(sub_nRows, sub_nCols, CV_32FC1);
	Mat sub_magImg(sub_nRows, sub_nCols, CV_32FC1);

	ustc_CalcGrad(grayImg, grad_x, grad_y);
	ustc_CalcAngleMag(grad_x, grad_y, angleImg, magImg);

	ustc_CalcGrad(subImg, sub_grad_x, sub_grad_y);
	ustc_CalcAngleMag(sub_grad_x, sub_grad_y, sub_angleImg, sub_magImg);

	float *data = (float *)magImg.data;
	float *sub_data = (float *)sub_magImg.data;

	float min_err = FLT_MAX;

	for (int i = 0; i < nRows - sub_nRows; i++)
	{
		for (int j = 0; j < nCols - sub_nCols; j++)
		{
			float total_err = 0;
			for (int sub_i = 1; sub_i < sub_nRows - 1; sub_i++)
			{
				for (int sub_j = 1; sub_j < sub_nCols - 1; sub_j++)
				{
					int index = (i + sub_i)*nCols + (j + sub_j);
					int sub_index = sub_i*sub_nCols + sub_j;
					int err = data[index] - sub_data[sub_index];
					total_err += (err & 0x80000000) ? -err : err;
				}
			}

			if (min_err > total_err) {
				min_err = total_err;
				*x = i;
				*y = j;
			}
		}
	}

	if (*x == -1 == *y) {
		cout << "match failed." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	else
	{
		return SUB_IMAGE_MATCH_OK;
	}
}


//函数功能：利用直方图进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (grayImg.data == NULL || subImg.data == NULL)
	{
		printf("image is NULL!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width <= sub_width || height <= sub_height)
	{
		printf("subgraph is bigger than original picture!\n");
		return SUB_IMAGE_MATCH_FAIL;
	}

	int sum = INT_MAX;
	int* hist_temp = new int[255];
	memset(hist_temp, 0, sizeof(int) * 255);

	int* sub_hist = new int[255];
	memset(sub_hist, 0, sizeof(int) * 255);

	//统计子图直方图
	for (int row_i = sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = sub_width - 1; col_j >= 0; col_j--)
		{
			int pixVal = subImg.data[row_i * sub_width + col_j];
			sub_hist[pixVal]++;
		}
	}

	for (int row_i = height - sub_height - 1; row_i >= 0; row_i--)
	{
		for (int col_j = width - sub_width - 1; col_j >= 0; col_j--)
		{
			memset(hist_temp, 0, sizeof(int) * 255);

			for (int row_x = sub_height - 1; row_x >= 0; row_x--)
			{
				for (int col_y = sub_width - 1; col_y >= 0; col_y--)
				{
					int row_index = row_i + row_x;
					int col_index = col_j + col_y;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					hist_temp[bigImg_pix]++;
				}
			}

			int total_diff = 0;
			for (int count = 254; count >= 0; count--)
			{
				total_diff += abs(hist_temp[count] - sub_hist[count]);
			}

			if (total_diff < sum)
			{
				*x = col_j + 1;
				*y = row_i + 1;
				sum = total_diff;
			}
		}
	}

	delete[] hist_temp;
	delete[] sub_hist;

	return SUB_IMAGE_MATCH_OK;
}

//测试彩色图像与灰度图像转换
void test1()
{
	Mat bgrImg = imread("D:\\twogirls.jpg");
	Mat grayImg(bgrImg.rows, bgrImg.cols, CV_8UC1);
	if (bgrImg.data == NULL)
	{
		printf("Image read failed!\n");
		return;
	}
#ifdef IMG_SHOW
	namedWindow("bgrImg", WINDOW_NORMAL);
	imshow("bgrImag", bgrImg);
	waitKey(1000);
#endif


	//time_t start = clock();
	//for (int test_num = 1000; test_num > 0; test_num--)
	//{
	int flag = ustc_ConvertBgr2Gray(bgrImg, grayImg);
	//}
	//time_t end = clock();
	//printf("time:%d\n", (end - start) / 1000);
}

//测试灰度图计算x方向和y方向梯度
void test2()
{
	Mat grayImg = imread("D:\\m9.jpg", 0);
	if (grayImg.data == NULL)
	{
		printf("image read failed!\n");
		return;
	}

#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(1000);
#endif

	Mat gradImg_x(grayImg.rows, grayImg.cols, CV_32FC1);
	Mat gradImg_y(grayImg.rows, grayImg.cols, CV_32FC1);

	int flag = ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
}

//测试梯度计算图像的角度和幅值
void test3()
{
	Mat grayImg = imread("D:\\m9.jpg", 0);
	if (grayImg.data == NULL)
	{
		printf("image read failed!\n");
		return;
	}
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(100);
#endif

	Mat gradImg_x(grayImg.rows, grayImg.cols, CV_32FC1);
	Mat gradImg_y(grayImg.rows, grayImg.cols, CV_32FC1);
	Mat angleImg(grayImg.rows, grayImg.cols, CV_32FC1);
	Mat magImg(grayImg.rows, grayImg.cols, CV_32FC1);

	int flag = ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);

	int tag = ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
}

//测试灰度图像的二值化
void test4()
{
	Mat grayImg = imread("D:\\twogirls.jpg", 0);
	if (grayImg.data == NULL)
	{
		printf("read image fail!\n");
		return;
	}
	int th = 200;
	Mat binaryImg(grayImg.rows, grayImg.cols, CV_8UC1);

#ifdef IMG_SHOW
	namedWindow("bgrImg", WINDOW_NORMAL);
	imshow("bgrImag", grayImg);
	waitKey(1000);
#endif

	int flag = ustc_Threshold(grayImg, binaryImg, th);
}


//测试灰度图像直方图
void test5()
{
	Mat grayImg = imread("D:\\m9.jpg", 0);
	if (grayImg.data == NULL)
	{
		printf("read image fail!\n");
		return;
	}
	Mat Grey(grayImg.rows, grayImg.cols, CV_8UC1);
	int hist[255];

#ifdef IMG_SHOW
	namedWindow("bgrImg", WINDOW_NORMAL);
	imshow("bgrImag", grayImg);
	waitKey(1000);
#endif

	int flag = ustc_CalcHist(grayImg, hist, 255);
	CvHistogram* cvCreateHist(int dims, int* sizes, int type, float** ranges = NULL, int uniform = 1);
	IplImage *imgHist = cvCreateImage(cvSize(256 * 1, 64 *1), 8, 1);
	cvZero(imgHist);
	int histsize = grayImg.rows;

	for (int h = 0; h<256; h++)
	{
		line(Grey, Point(h, hist[h]), Point(h, 0), Scalar::all(0));
	}

	namedWindow("Greys", WINDOW_NORMAL);
	imshow("Greys", Grey);
	waitKey(1000);




}




//测试灰度比较的子块匹配
void test6()
{
	Mat grayImg = imread("D:\\IMG_1.jpg", 0);
	Mat subImg = imread("D:\\IMG_2.jpg", 0);
	int fix_x, fix_y;
	int* x, *y;
	x = &fix_x;
	y = &fix_y;
	int flag = ustc_SubImgMatch_corr(grayImg, subImg, x, y);
	printf("x=%d,y=%d\n", fix_x, fix_y);
	getchar();
}

//测试色彩比较的子块匹配
void test7()
{
	Mat colorImg = imread("D:\\IMG_1.jpg");
	Mat subImg = imread("D:\\IMG_2.jpg");
	int fix_x, fix_y;
	int* x, *y;
	x = &fix_x;
	y = &fix_y;
	int flag = ustc_SubImgMatch_bgr(colorImg, subImg, x, y);
	printf("x=%d,y=%d\n", fix_x, fix_y);
	getchar();
}

//测试灰度相关的子块匹配
void test8()
{
	Mat grayImg = imread("D:\\IMG_1.jpg", 0);
	Mat subImg = imread("D:\\IMG_2.jpg", 0);
	int fix_x, fix_y;
	int* x, *y;
	x = &fix_x;
	y = &fix_y;
	int flag = ustc_SubImgMatch_gray(grayImg, subImg, x, y);
	printf("x=%d,y=%d\n", fix_x, fix_y);
	getchar();
}

//测试角度比较的子块匹配
void test9()
{
	Mat grayImg = imread("D:\\IMG_1.jpg", 0);
	Mat subImg = imread("D:\\IMG_2.jpg", 0);
	int fix_x, fix_y;
	int* x, *y;
	x = &fix_x;
	y = &fix_y;
	int flag = ustc_SubImgMatch_angle(grayImg, subImg, x, y);
	printf("x=%d,y=%d\n", fix_x, fix_y);
	getchar();
}

//测试幅值比较的子块匹配
void test10()
{
	Mat grayImg = imread("D:\\IMG_1.jpg", 0);
	Mat subImg = imread("D:\\IMG_2.jpg", 0);
	int fix_x, fix_y;
	int* x, *y;
	x = &fix_x;
	y = &fix_y;
	int flag = ustc_SubImgMatch_mag(grayImg, subImg, x, y);
	printf("x=%d,y=%d\n", fix_x, fix_y);
	getchar();
}

//测试直方图比较的子块匹配
void test11()
{
	Mat grayImg = imread("D:\\IMG_1.jpg", 0);
	Mat subImg = imread("D:\\IMG_2.jpg", 0);
	int fix_x, fix_y;
	int* x, *y;
	x = &fix_x;
	y = &fix_y;
	int flag = ustc_SubImgMatch_hist(grayImg, subImg, x, y);
	printf("x=%d,y=%d\n", fix_x, fix_y);
	getchar();
}

//快速开方算法
float SqrtByRSQRTSS(float a)
{
	float b = a;
	__m128 in = _mm_load_ss(&b);
	__m128 out = _mm_rsqrt_ss(in);
	_mm_store_ss(&b, out);

	return a*b;
}
//快速开方求到数
float InvSqrt(float x)
{
	float xhalf = 0.5f * x;
	int i = *(int *)& x;
	i = 0x5f3759df - (i >> 1);
	x = *(float *)& i;
	x = x * (1.5f - xhalf * x * x);

	return x;
}

int _tmain(int argc, _TCHAR* argv[])
{
	test9();
	getchar();
	return 0;
}
