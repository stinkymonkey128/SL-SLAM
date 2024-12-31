#include "SuperPoint.h"
#include "LightGlue.h"

#include <iostream>

int main() {
    SuperPointConfig config;
    config.onnxFilePath = "SuperPoint.onnx";

    SuperPoint spoint(config);
    int buildStatus = spoint.build();
    if (!buildStatus)
        std::cout << "Successfully built or loaded SuperPoint engine!" << std::endl;
    else
        std::cout << "Failed to build SuperPoint engine... error code " << buildStatus << std::endl;

    cv::Mat image0 = cv::imread("00000.jpg");
    cv::Mat image1 = cv::imread("00001.jpg");

    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0;
    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points1;

    if (!spoint.infer(image0, feature_points0)) {
        std::cout << "Something FATAL happened while running inference!" << std::endl;
        return 1;
    }

    if (!spoint.infer(image1, feature_points1)) {
        std::cout << "Something FATAL happened while running inference!" << std::endl;
        return 1;
    }

    spoint.visualize(image0, feature_points0, "out.jpg");

    LightGlueConfig lgConfig;
    lgConfig.onnxFilePath = "LightGlue.onnx";

    LightGlue lightGlue(lgConfig);
    buildStatus = lightGlue.build();
    if (!buildStatus)
        std::cout << "Successfully built or loaded LightGlue engine!" << std::endl;
    else
        std::cout << "Failed to build LightGlue engine... error code " << buildStatus << std::endl;

    std::vector<cv::Point> kpts0;
    std::vector<cv::Point> kpts1;
    Eigen::VectorXd scores;

    if (!lightGlue.infer({feature_points0, feature_points1}, kpts0, kpts1, scores, image0.rows, image0.cols)) {
        std::cout << "Something FATAL happened while running inference!" << std::endl;
        return 1;
    }

    lightGlue.visualize(image0, image1, kpts0, kpts1, scores, "match.jpg");

    return 0;
}