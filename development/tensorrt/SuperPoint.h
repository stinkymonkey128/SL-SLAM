/*
* ripped from https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT
* with compaitibility update with TensorRT 10.7+
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
#include "buffers.h"
#include "common.h"

struct SuperPointConfig {
    std::string onnxFilePath;
    std::string engineFilePath = "SuperPoint.engine";
    std::vector<std::string> inputTensorNames = {"input"};
    std::vector<std::string> outputTensorNames = {"keypoints", "scores", "descriptors"};
    bool useDlaCore = false;
    int dlaCore = 0;
    size_t memoryPoolLimit = 512_MiB;
    tensorrt_log::Logger::Severity logSeverity = tensorrt_log::Logger::Severity::kINTERNAL_ERROR;
};

class SuperPoint {
public:
    SuperPoint(SuperPointConfig config);

    int build();

    // out features with idx0 scores idx1 & idx2 keypoints >= idx3 being descriptor
    bool infer(const cv::Mat& img, Eigen::Matrix<double, 259, Eigen::Dynamic>& features);
    static void visualize(
        const cv::Mat& img, 
        const Eigen::Matrix<double, 259, Eigen::Dynamic>& features, 
        const std::string& outImgPath
    );
    void saveEngine();
    bool deserializeEngine();
private:
    SuperPointConfig config_;

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    bool constructNetwork(
        std::unique_ptr<nvinfer1::IBuilder>& builder,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network,
        std::unique_ptr<nvinfer1::IBuilderConfig>& config,
        std::unique_ptr<nvonnxparser::IParser>& parser
    ) const;

    bool processInput(
        const tensorrt_buffer::BufferManager& buffers, 
        const cv::Mat& img
    );

    bool processOutput(
        const tensorrt_buffer::BufferManager& buffers, 
        Eigen::Matrix<double, 259, Eigen::Dynamic>& features
    );
};

#endif