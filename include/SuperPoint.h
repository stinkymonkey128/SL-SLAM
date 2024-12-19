#ifndef SUPERPOINT_H
#define SUPERPOINT_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <map>

namespace ORB_SLAM3 {
    struct SuperPointOutput {
        torch::Tensor keypoints;
        torch::Tensor scores;
        torch::Tensor descriptors;
    };

    class SuperPoint : torch::nn::Module {
        public:
            SuperPoint(
                int descriptor_dim = 256,
                int nms_radius = 4,
                int max_num_keypoints = -1,
                double detection_threshold = 0.0005,
                int remove_borders = 4
            );

            SuperPointOutput forward(const torch::Tensor& img);
        
        private:
            torch::Tensor simple_nms(torch::Tensor scores, int nms_radius);
            torch::Tensor sample_descriptors(torch::Tensor keypoints, torch::Tensor descriptors, int s = 8);

            int descriptor_dim_;
            int nms_radius_;
            int max_num_keypoints_;
            double detection_threshold_;
            int remove_borders_;

            torch::nn::Conv2d conv1a{nullptr};
            torch::nn::Conv2d conv1b{nullptr};
            torch::nn::Conv2d conv2a{nullptr};
            torch::nn::Conv2d conv2b{nullptr};
            torch::nn::Conv2d conv3a{nullptr};
            torch::nn::Conv2d conv3b{nullptr};
            torch::nn::Conv2d conv4a{nullptr};
            torch::nn::Conv2d conv4b{nullptr};
            torch::nn::Conv2d convPa{nullptr};
            torch::nn::Conv2d convPb{nullptr};
            torch::nn::Conv2d convDa{nullptr};
            torch::nn::Conv2d convDb{nullptr};
            torch::nn::MaxPool2d pool{nullptr};
    };
}

#endif