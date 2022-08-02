#include<stdio.h>
#include<math.h>
#include<cmath>
#include <stdlib.h>
#include<ctime>
#include<windows.h>
 #include "mex.h"       
#include<Eigen/Dense>

using namespace Eigen;


void arrayToMatrix(double *x, MatrixXd &m, int mxRows, int  mxCols)
{
	//赋值给Eigen矩阵
	for (int k = 0; k < mxRows; k++)
	{
		for (int l = 0; l < mxCols; l++)
		{
			m(k, l) = x[mxRows*l + k];
		}
	}
}



void MatrixToArray(double *y, MatrixXd &m, int rows, int cols) //返回到matlab
{

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			y[rows*c + r] = m(r, c);//赋值给Y,传到Matlab中
		}
	}

}
void addDebug(char *log)
{
// 	FILE *fp = NULL;
// 
// 	fp = fopen("log.txt", "a");
// 	fprintf(fp, log);
// 	fclose(fp);
}

// double* UpdateSv(double *f, double *x, double *b, int cluNum, int SampleNum, int feaNum, double *s)
// {
// 	//addDebug("11\n");
// 	/*MatrixXd mm(row, col);
// 	arrayToMatrix(y, mm, row, col);*/
// 
// 	MatrixXd F = Map<MatrixXd>(f, SampleNum, cluNum);
// 	MatrixXd X = Map<MatrixXd>(x, feaNum, SampleNum);
// 	MatrixXd B = Map<MatrixXd>(b, SampleNum, SampleNum);
// 	
// 	MatrixXd h = MatrixXd::Zero(SampleNum, SampleNum);
// 	MatrixXd S = MatrixXd::Zero(SampleNum, SampleNum);
// 	for (int i = 0; i < SampleNum; i++)
// 	{
// 		for (int j = 0; j < SampleNum; j++)
// 		{
// 			if (i < j)
// 			{
// 				h(i, j) = (F.row(i) - F.row(j)).squaredNorm();
// 			}
// 			else if (i == j)
// 			{
// 				h(i, j) = 0;
// 			}
// 			else
// 			{
// 				h(i, j) = h(j, i);
// 			}
// 		}
// 		S.col(i) = B * (X.transpose() * X.col(i) - h.row(i).transpose());
// 	}
// 	//*s = 121;
// 	Map<MatrixXd>(s, SampleNum, SampleNum) = S;//矩阵传入数组中
// 	
// 	//s[0] = 121;
// 	//s[1] = 1211;
// 	//s = S.data();//转为数组
// 	char str[100];
// 	sprintf(str, "s[0]:%.4f s[1]:%.4f s[3]:%.4f\n", s[0], s[1], s[2]);
// 	//addDebug(str);
// 
// 
// 	//addDebug("22\n");	
// 	//cout << "det:" << mm.determinant() << endl;
// 	
// 	return s;
// }


//prhs[0] F  prhs[1] B  prhs[2] X prhs[3] lamda
void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray *prhs[])
{
    
    //获得matlab矩阵的函数和列数
    int cluNum,  SampleNum, feaNum;
	
    SampleNum = mxGetM(prhs[0]);
    cluNum = mxGetN(prhs[0]);
	feaNum = mxGetM(prhs[2]);
    int lamda = *(mxGetPr(prhs[3]));
    
	double start, stop;
    //start = GetTickCount();
    
	MatrixXd F = Map<MatrixXd>(mxGetPr(prhs[0]), SampleNum, cluNum);
	//MatrixXd X = Map<MatrixXd>(mxGetPr(prhs[2]), feaNum, SampleNum);
	//MatrixXd B = Map<MatrixXd>(mxGetPr(prhs[1]), SampleNum, SampleNum);
    MatrixXd h = MatrixXd::Zero(SampleNum, SampleNum);
	//MatrixXd S = MatrixXd::Zero(SampleNum, SampleNum);
   
   //start = GetTickCount();
	for (int i = 0; i < SampleNum; i++)
	{
		for (int j = 0; j < SampleNum; j++)
		{
			if (i < j)
			{
				h(i, j) = (F.row(i) - F.row(j)).squaredNorm();
			}
			else if (i == j)
			{
				h(i, j) = 0;
			}
			else
			{
				h(i, j) = h(j, i);
			}
		}
		//S.col(i) = B * (X.transpose() * X.col(i) - h.row(i).transpose());
	}
//     start = GetTickCount();
//     S = B * (X.transpose() * X - lamda / 2 * h.transpose()); matlab只要0.28秒，cpp要2.1秒
    // stop = GetTickCount();
	//mexPrintf(" matrix multiplication cost :%.2f\n", stop - start); 
    double *y;
    plhs[0] = mxCreateDoubleMatrix(SampleNum,SampleNum,mxREAL);
    y = mxGetPr(plhs[0]);
    
    Map<MatrixXd>(y, SampleNum, SampleNum) = h;//矩阵传入数组中
    
}



