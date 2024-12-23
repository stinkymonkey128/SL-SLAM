/*
* Ripped from https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT
* with major compatibility updates for tensorrt 10.x+
*/

#ifndef SUPERPOINT_H
#define SUPERPOINT_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <cstddef>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "logger.h"

struct SuperPointConfig {
    std::string onnxFilePath;
    std::string engineFilePath;
    std::vector<std::string> inputTensorNames;
    int dlaCore;
};

class SuperPoint {
    public:
        SuperPoint(SuperPointConfig config);

        bool build();
        bool infer(const cv::Mat& img, Eigen::Matrix<double, 259, Eigen::Dynamic>& features);
        void visualize(const std::string& img_path, const cv::Mat& img);
        void saveEngine();
        bool deserializeEngine();
    private:
        SuperPointConfig config_;

        nvinfer1::Dims inputDims_;
        nvinfer1::Dims semiDims_;
        nvinfer1::Dims descDims_;

        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        std::shared_ptr<nvinfer1::IExecutionContext> context_;

        std::vector<std::vector<int>> keypoints_;
        std::vector<std::vector<double>>  descriptors_;

        bool constructNetwork(
            std::unique_ptr<nvinfer1::IBuilder>& builder,
            std::unique_ptr<nvinfer1::INetworkDefinition>& network,
            std::unique_ptr<nvinfer1::IBuilderConfig>& config,
            std::unique_ptr<nvonnxparser::IParser>& parser
        ) const;
    
};

constexpr std::size_t operator"" _MiB(unsigned long long int value) {
    return value * (1ULL << 20);
}

#endif