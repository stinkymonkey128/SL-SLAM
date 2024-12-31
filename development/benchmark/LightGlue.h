/*
* due to issues with dynamo not being able to handle i64 
* shapes must be static so max keypoints must be predetermined
*/

#ifndef LIGHTGLUE_H
#define LIGHTGLUE_H

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

#define MAX_KEYPOINTS 1024

struct LightGlueConfig {
    std::string onnxFilePath;
    std::string engineFilePath = "LightGlue.engine";
    std::vector<std::string> inputTensorNames = {"keypoints", "descriptors"};
    std::vector<std::string> outputTensorNames = {"matches", "scores"};
    bool useDlaCore = false;
    int dlaCore = 0;
    size_t memoryPoolLimit = 512_MiB;
    tensorrt_log::Logger::Severity logSeverity = tensorrt_log::Logger::Severity::kINTERNAL_ERROR;
};

class LightGlue {
public:
    LightGlue(LightGlueConfig config);

    int build();
    bool infer(
        const std::vector<Eigen::Matrix<double, 259, Eigen::Dynamic>>& features,
        std::vector<cv::Point>& kpts0,
        std::vector<cv::Point>& kpts1,
        Eigen::VectorXd& scores,
        const int imgHeight,
        const int imgWidth
    );
    void saveEngine();
    bool deserializeEngine();

    static void visualize(
        const cv::Mat& img0,
        const cv::Mat& img1,
        const std::vector<cv::Point> kpts0, 
        const std::vector<cv::Point> kpts1,
        const Eigen::VectorXd scores,
        const std::string& outImgPath,
        const float scaleFactor = 0.9
    );

private:
    LightGlueConfig config_;

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
        const std::vector<Eigen::Matrix<double, 259, Eigen::Dynamic>>& features,
        int imgHeight,
        int imgWidth
    );

    bool processOutput(
        const tensorrt_buffer::BufferManager& buffers,
        const std::vector<Eigen::Matrix<double, 259, Eigen::Dynamic>>& features,
        std::vector<cv::Point>& kpts0,
        std::vector<cv::Point>& kpts1,
        Eigen::VectorXd& scores
    );
};

#endif