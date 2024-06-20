#include "mex.h"
#include <cmath>
#include <algorithm>

// 辅助函数：group_soft_threshold
void group_soft_threshold(double *a, double kappa, double *z, mwSize size) {
    double norm = 0;
    for (mwSize i = 0; i < size; ++i) {
        norm += a[i] * a[i];
    }
    norm = std::sqrt(norm);

    double tmp = 1 - kappa / norm;
    if (tmp > 0) {
        for (mwSize i = 0; i < size; ++i) {
            z[i] = tmp * a[i];
        }
    } else {
        std::fill_n(z, size, 0);
    }
}

// MEX 函数主体
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {


    // 获取输入参数
    double *delta2 = mxGetPr(prhs[0]);
    double gamma = mxGetScalar(prhs[1]);
    double a = mxGetScalar(prhs[2]);
    mwSize len_l = mxGetM(prhs[0]);
    mwSize rows = mxGetN(prhs[0]);

    // 创建输出数组
    plhs[0] = mxCreateDoubleMatrix(len_l, rows, mxREAL);
    double *output = mxGetPr(plhs[0]);

    // 遍历每一行
    for (mwSize l = 0; l < len_l; ++l) {
        // 创建一个临时数组用于存储 delta2 的第 l 行
        double *row = new double[rows];
        for (mwSize i = 0; i < rows; ++i) {
            row[i] = delta2[l + i * len_l];
        }

        // 计算行的范数
        double norm = 0;
        for (mwSize i = 0; i < rows; ++i) {
            norm += row[i] * row[i];
        }
        norm = std::sqrt(norm);

        // 应用阈值操作
        if (norm <= gamma * 2) {
            group_soft_threshold(row, gamma, row, rows);
        } else if (norm > gamma * 2 && norm <= a * gamma) {
            group_soft_threshold(row, a * gamma / (a - 1), row, rows);
            double scale = 1 - 1 / (a - 1);
            for (mwSize i = 0; i < rows; ++i) {
                row[i] /= scale;
            }
        }

        // 将处理后的行数据回写到 output 数组
        for (mwSize i = 0; i < rows; ++i) {
            output[l + i * len_l] = row[i];
        }

        // 清理分配的内存
        delete[] row;
    }
}