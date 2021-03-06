#include "SubImageMatch.h"
#include <math.h>

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	int i, total, j;
	uchar* data, data_f;
	data = bgrImg.data;
	if (data == NULL)
		return SUB_IMAGE_MATCH_FAIL;

	total = bgrImg.rows* bgrImg.cols;
	for (i = 0, j = 0; i < total; i++,j+=3)
	{
		grayImg.data[i] = (data[j] * 117 + data[j + 1] * 601 + data[j + 2] * 306) >> 10;	
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	int i, j, row, col, temp = 0, row1, row3;
	uchar* data;
	row = grayImg.rows;
	col = grayImg.cols;
	data = grayImg.data;
	if (data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	for (i = 0; i < col; i++)
	{
		((float*)gradImg_x.data)[i] = *(data + i);
		((float*)gradImg_y.data)[i] = *(data + i);
	}
	for (i = 1; i < row - 1; i++)
	{
		temp = i * col;
		((float*)gradImg_x.data)[temp] = *(data + temp);
		((float*)gradImg_y.data)[temp] = *(data + temp);
		for (j = 1; j < col - 1; j++)
		{
			row1 = temp - col;
			row3 = temp + col;
			((float*)gradImg_x.data)[temp + j] = -*(data + row1 + j - 1) + *(data + row3 + j - 1)
										         - *(data + row1 + j) * 2 + *(data + row3 + j) * 

2 
										         - *(data + row1 + j + 1) + *(data + row3 + j + 

1);
			((float*)gradImg_y.data)[temp + j] = -*(data + row1 + j - 1) - *(data + temp + j - 1) * 2 - *(data + row3 + j - 

1)
				                                 + *(data + row1 + j + 1) + *(data + temp + j + 1) * 2 + *(data + row3 + 

j + 1);

		}
		temp = temp + col - 1;
		((float*)gradImg_x.data)[temp] = *(data + temp);
		((float*)gradImg_y.data)[temp] = *(data + temp);
	}
	temp = temp + 1;
	for (i = 0; i < col; i++)
	{
		((float*)gradImg_x.data)[temp + i] = *(data + temp + i);
		((float*)gradImg_y.data)[temp + i] = *(data + temp + i);
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	int total, i, t;
	float*datax, *datay;
	datax = (float*)gradImg_x.data;
	datay = (float*)gradImg_y.data;
	if (datax&&datay == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	total = gradImg_x.rows*gradImg_x.cols;
	for (i = 0; i < total; i++)
	{
		((float*)angleImg.data)[i] = atan2(datay[i], datax[i]) * 180 / 3.1415926 + 180;
		((float*)magImg.data)[i] = sqrt(datay[i] * datay[i] + datax[i] * datax[i]);
	}
	
	return SUB_IMAGE_MATCH_OK;

}

int ustc_Threshold(Mat grayImg, Mat & binaryImg, int th)
{
	int total, i, temp;
	temp = th;
	uchar*data;
	data = grayImg.data;
	if (data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	total = grayImg.rows*grayImg.cols;
	for (i = 0; i < total; i++)
	{
		if (data[i] -temp>>31)
			binaryImg.data[i]= 255;
		else
			binaryImg.data[i]= 0;
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcHist(Mat grayImg, int * hist, int hist_len)
{
	uchar* data;
	int i, total;
	data = grayImg.data;
	if (data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	total = grayImg.rows*grayImg.cols;
	
	memset(hist, 0, sizeof(int)*hist_len);

	for (i = 0; i < total; i++)
		hist[data[i]] += 1;
	return SUB_IMAGE_MATCH_OK;
		
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int * x, int * y)
{
	int row, col, sub_row, sub_col, min, total_diff;
	int i, j, sub_i, sub_j, start, sub_rows;
	uchar* data, *sub_data;
	data = grayImg.data;
	sub_data = subImg.data;
	if (data&&sub_data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	sub_row = subImg.rows;
	sub_col = subImg.cols;
	min = 255 * sub_row * sub_col;
	row = grayImg.rows - sub_row;
	col = grayImg.cols - sub_col;
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
		{
			total_diff = 0;
			for (sub_i = 0; sub_i < sub_row; sub_i++)
			{
				sub_rows = sub_i*sub_col;
				start = (i + sub_i)*grayImg.cols + j;
				for (sub_j = 0; sub_j < sub_col; sub_j++)
				{
					total_diff += abs(data[start + sub_j] - sub_data[sub_rows + sub_j]);
				}
			}
			if (total_diff < min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int * x, int * y)
{
	int row, col, sub_row, sub_col, min, total_diff;
	int i, j, sub_i, sub_j, start, sub_rows, sub_cols;
	uchar* data, *sub_data;
	data = colorImg.data;
	sub_data = subImg.data;
	if (data&&sub_data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	sub_row = subImg.rows;
	sub_col = subImg.cols;
	min = 255 * sub_row * sub_col * 3;
	row = colorImg.rows - sub_row;
	col = colorImg.cols - sub_col;
	sub_cols = subImg.cols * 3;
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
		{
			total_diff = 0;
			for (sub_i = 0; sub_i < sub_row; sub_i++)
			{
				sub_rows = sub_i*sub_cols;
				start = ((i + sub_i)*colorImg.cols + j) * 3;
				for (sub_j = 0; sub_j < sub_cols; sub_j+=3)
				{
					total_diff += abs(data[start + sub_j] - sub_data[sub_rows + sub_j]) + abs(data[start + sub_j + 1] 

- sub_data[sub_rows + sub_j + 1]) + abs(data[start + sub_j + 2] - sub_data[sub_rows + sub_j + 2]);
				}
			}
			if (total_diff < min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int * x, int * y)
{
	int row, col, sub_row, sub_col, total;
	int i, j, sub_i, sub_j, start, sub_rows;
	float r = 0, r_max = 0;
	int sum = 0, sub_sum = 0, aver = 0, sub_aver = 0, xiang = 0;
	uchar* data, *sub_data;
	data = grayImg.data;
	sub_data = subImg.data;
	if (data&&sub_data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	sub_row = subImg.rows;
	sub_col = subImg.cols;
	total = sub_row*sub_col;
	row = grayImg.rows - sub_row;
	col = grayImg.cols - sub_col;

	for (i = 0; i < total; i++)
	{
		sub_sum += (int)sub_data[i] * (int)sub_data[i];
	}
	sub_aver = sqrt(sub_sum);

	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
		{
			xiang = 0;
			sum = 0;
			aver = 0;
			for (sub_i = 0; sub_i < sub_row; sub_i++)
			{
				sub_rows = sub_i*sub_col;
				start = (i + sub_i)*grayImg.cols + j;
				for (sub_j = 0; sub_j < sub_col; sub_j++)
				{
					xiang += (int)data[start + sub_j] * (int)sub_data[sub_rows + sub_j];
					sum += (int)data[start + sub_j] * (int)data[start + sub_j];
				}
			}
			aver = sqrt(sum);
			r = xiang / (aver*sub_aver);
			if (r > r_max)
			{
				r_max = r;
				*x = j;
				*y = i;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int * x, int * y)
{
	if (grayImg.data&&subImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	
	Mat gradImg_x(grayImg.rows, grayImg.cols, CV_32FC1);
	Mat gradImg_y(grayImg.rows, grayImg.cols, CV_32FC1);
	Mat angleImg(grayImg.rows, grayImg.cols, CV_32FC1);
	Mat sub_gradImg_x(subImg.rows, subImg.cols, CV_32FC1);
	Mat sub_gradImg_y(subImg.rows, subImg.cols, CV_32FC1);
	Mat sub_angleImg(subImg.rows, subImg.cols, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	int total, i, sub_total, sub_i;
	float*datax, *datay, *sub_datax, *sub_datay;
	datax = (float*)gradImg_x.data;
	datay = (float*)gradImg_y.data;
	sub_datax = (float*)sub_gradImg_x.data;
	sub_datay = (float*)sub_gradImg_y.data;
	total = gradImg_x.rows*gradImg_x.cols;
	sub_total = sub_gradImg_x.rows*sub_gradImg_x.cols;

	for (i = 0; i < total; i++)
	{
		((float*)angleImg.data)[i] = atan2(datay[i], datax[i]) * 180 / 3.1415926 + 180;
	}
	for (sub_i = 0; sub_i < sub_total; sub_i++)
	{
		((float*)sub_angleImg.data)[sub_i] = atan2(sub_datay[sub_i], sub_datax[sub_i]) * 180 / 3.1415926 + 180;
	}
	
	int row, col, sub_row, sub_col;
	int j, sub_j, start, sub_rows;
	float* data, *sub_data;
	int min, total_diff, diff;
	float temp, sub_temp;
	int temp_c, sub_temp_c;
	data = (float*)angleImg.data;
	sub_data = (float*)sub_angleImg.data;
	sub_row = sub_angleImg.rows;
	sub_col = sub_angleImg.cols;
	min = 255 * sub_row * sub_col * 3;
	row = angleImg.rows - sub_row;
	col = angleImg.cols - sub_col;
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
		{
			total_diff = 0;
			for (sub_i = 0; sub_i < sub_row; sub_i++)
			{
				sub_rows = sub_i*sub_col;
				start = (i + sub_i)*angleImg.cols + j;
				for (sub_j = 0; sub_j < sub_col; sub_j++)
				{
					diff = (int)data[start + sub_j] - (int)sub_data[sub_rows + sub_j];
					if (diff < -180)
						diff += 360;
					else if (diff < 0)
						diff = -diff;
					else if (diff > 180)
						diff = 360 - diff;
					total_diff += diff;
				}
			}
			if (total_diff < min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;

}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int * x, int * y)
{
	if (grayImg.data&&subImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	
	Mat gradImg_x(grayImg.rows, grayImg.cols, CV_32FC1);
	Mat gradImg_y(grayImg.rows, grayImg.cols, CV_32FC1);
	Mat magImg(grayImg.rows, grayImg.cols, CV_32FC1);
	Mat sub_gradImg_x(subImg.rows, subImg.cols, CV_32FC1);
	Mat sub_gradImg_y(subImg.rows, subImg.cols, CV_32FC1);
	Mat sub_magImg(subImg.rows, subImg.cols, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	int total, i, sub_total, sub_i;
	float*datax, *datay, *sub_datax, *sub_datay;
	datax = (float*)gradImg_x.data;
	datay = (float*)gradImg_y.data;
	sub_datax = (float*)sub_gradImg_x.data;
	sub_datay = (float*)sub_gradImg_y.data;
	total = gradImg_x.rows*gradImg_x.cols;
	sub_total = sub_gradImg_x.rows*sub_gradImg_x.cols;


	for (i = 0; i < total; i++)
	{
		((float*)magImg.data)[i] = sqrt(datay[i] * datay[i] + datax[i] * datax[i]);
	}
	for (sub_i = 0; sub_i < sub_total; sub_i++)
	{
		((float*)sub_magImg.data)[sub_i] = sqrt(sub_datay[sub_i] * sub_datay[sub_i] + sub_datax[sub_i] * sub_datax[sub_i]);
	}

	
	int row, col, sub_row, sub_col;
	int j, sub_j, start, sub_rows;
	float* data, *sub_data, min, total_diff;
	data = (float*)magImg.data;
	sub_data = (float*)sub_magImg.data;
	sub_row = sub_magImg.rows;
	sub_col = sub_magImg.cols;
	min = 255 * sub_row * sub_col * 3;
	row = magImg.rows - sub_row;
	col = magImg.cols - sub_col;
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
		{
			total_diff = 0;
			for (sub_i = 0; sub_i < sub_row; sub_i++)
			{
				sub_rows = sub_i*sub_col;
				start = (i + sub_i)*magImg.cols + j;
				for (sub_j = 0; sub_j < sub_col; sub_j++)
				{
					total_diff += abs(data[start + sub_j] - sub_data[sub_rows + sub_j]);
				}
			}
			if (total_diff < min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int * x, int * y)
{
	int row, col, sub_row, sub_col, min, total_diff;
	int i, j, sub_i, sub_j, total, start;
	uchar* data, *sub_data;
	data = grayImg.data;
	sub_data = subImg.data;
	if (data&&sub_data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	sub_row = subImg.rows;
	sub_col = subImg.cols;
	total = sub_row*sub_col;
	min = 255 * sub_row * sub_col;
	row = grayImg.rows - sub_row;
	col = grayImg.cols - sub_col;

	int *sub_hist, *hist;
	sub_hist = new int[256];
	hist = new int[256];
	memset(sub_hist, 0, sizeof(int) * 256);
	for (i = 0; i < total; i++)
	{
		sub_hist[(sub_data[i])] ++;
	}

	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
		{
			total_diff = 0;
			memset(hist, 0, sizeof(int) * 256);
			for (sub_i = 0; sub_i < sub_row; sub_i++)
			{
				start = (i + sub_i)*grayImg.cols + j;
				for (sub_j = 0; sub_j < sub_col; sub_j++)
				{
					hist[(data[start + sub_j])] ++;
				}
			}
			for (sub_i = 0; sub_i < 256; sub_i++)
			{
				total_diff += abs(hist[sub_i] - sub_hist[sub_i]);
			}
			if (total_diff < min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}
