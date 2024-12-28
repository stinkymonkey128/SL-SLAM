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
    std::vector<std::string> inputTensorNames; // should be size 1 with idx 0 being input img in grayscale
    std::vector<std::string> outputTensorNames; // should be size 2 with idx 0 being score and idx 1 being descriptors
    bool useDlaCore = false;
    int dlaCore = 0;
    float confThreshold = 0.0005f;
    int removeBorders = 0;
    int maxKeypoints = 1024;
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

    bool processInput(const tensorrt_buffer::BufferManager& buffers, const cv::Mat& img);
    bool processOutput(const tensorrt_buffer::BufferManager& buffers, Eigen::Matrix<double, 259, Eigen::Dynamic>& features);

    void findHighScoreIndex(
        std::vector<std::vector<int>>& keypoints,
        std::vector<float>& scores, 
        int h, 
        int w, 
        double threshold
    );

    void topKKeypoints(
        std::vector<std::vector<int>>& keypoints, 
        std::vector<float>& scores, 
        int k
    );

    std::vector<size_t> sortIndexes(std::vector<float>& data);

    void removeBorders(
        std::vector<std::vector<int>>& keypoints, 
        std::vector<float>& scores, 
        int border, 
        int h, 
        int w
    );

    void sampleDescriptors(
        std::vector<std::vector<int>>& keypoints,
        float* descriptors,
        std::vector<std::vector<double>>& destDescriptors,
        int dim,
        int h,
        int w,
        int s = 8
    );
};

#endif