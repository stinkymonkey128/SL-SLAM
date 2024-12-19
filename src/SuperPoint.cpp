#include "SuperPoint.h"

namespace ORB_SLAM3 {
    SuperPoint::SuperPoint(
        int descriptor_dim,
        int nms_radius,
        int max_num_keypoints,
        double detection_threshold,
        int remove_borders
    ) :
        descriptor_dim_(descriptor_dim),
        nms_radius_(nms_radius),
        max_num_keypoints_(max_num_keypoints),
        detection_threshold_(detection_threshold),
        remove_borders_(remove_borders)
    {
        conv1a = register_module("conv1a", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1)));
        conv1b = register_module("conv1b", torch::nn::Conv2d(torch::nn::Conv2dOptions(64,64,3).stride(1).padding(1)));
        conv2a = register_module("conv2a", torch::nn::Conv2d(torch::nn::Conv2dOptions(64,64,3).stride(1).padding(1)));
        conv2b = register_module("conv2b", torch::nn::Conv2d(torch::nn::Conv2dOptions(64,64,3).stride(1).padding(1)));
        conv3a = register_module("conv3a", torch::nn::Conv2d(torch::nn::Conv2dOptions(64,128,3).stride(1).padding(1)));
        conv3b = register_module("conv3b", torch::nn::Conv2d(torch::nn::Conv2dOptions(128,128,3).stride(1).padding(1)));
        conv4a = register_module("conv4a", torch::nn::Conv2d(torch::nn::Conv2dOptions(128,128,3).stride(1).padding(1)));
        conv4b = register_module("conv4b", torch::nn::Conv2d(torch::nn::Conv2dOptions(128,128,3).stride(1).padding(1)));

        convPa = register_module("convPa", torch::nn::Conv2d(torch::nn::Conv2dOptions(128,256,3).stride(1).padding(1)));
        convPb = register_module("convPb", torch::nn::Conv2d(torch::nn::Conv2dOptions(256,65,1).stride(1).padding(0)));

        convDa = register_module("convDa", torch::nn::Conv2d(torch::nn::Conv2dOptions(128,256,3).stride(1).padding(1)));
        convDb = register_module("convDb", torch::nn::Conv2d(torch::nn::Conv2dOptions(256,descriptor_dim_,1).stride(1).padding(0)));

        pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
    }

    SuperPointOutput SuperPoint::forward(const torch::Tensor& img) {
        TORCH_CHECK(img.dim() == 4, "Image Dimension Mismatch!");

        // not doing grayscale conversion

        auto x = torch::relu(conv1a->forward(img));
        x = torch::relu(conv1b->forward(x));
        x = pool->forward(x);
        x = torch::relu(conv2a->forward(x));
        x = torch::relu(conv2b->forward(x));
        x = pool->forward(x);
        x = torch::relu(conv3a->forward(x));
        x = torch::relu(conv3b->forward(x));
        x = pool->forward(x);
        x = torch::relu(conv4a->forward(x));
        x = torch::relu(conv4b->forward(x));

        auto cPa = torch::relu(convPa->forward(x));
        auto scores = convPb->forward(cPa);
        scores = torch::softmax(scores, 1);
        scores = scores.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 64)});

        auto b = scores.size(0);
        auto h = scores.size(2);
        auto w = scores.size(3);

        scores = scores.permute({0,2,3,1});
        scores = scores.reshape({b, h, w, 8, 8});
        scores = scores.permute({0,1,3,2,4});
        scores = scores.reshape({b, h*8, w*8});

        scores = simple_nms(scores, nms_radius_);

        if (remove_borders_ > 0) {
            int pad = remove_borders_;
            scores.index_put_({torch::indexing::Ellipsis, torch::indexing::Slice(0,pad), torch::indexing::Ellipsis}, -1);
            scores.index_put_({torch::indexing::Ellipsis, torch::indexing::Slice(scores.size(1)-pad,torch::indexing::None), torch::indexing::Ellipsis}, -1);
            scores.index_put_({torch::indexing::Ellipsis, torch::indexing::Ellipsis, torch::indexing::Slice(0,pad)}, -1);
            scores.index_put_({torch::indexing::Ellipsis, torch::indexing::Ellipsis, torch::indexing::Slice(scores.size(2)-pad,torch::indexing::None)}, -1);
        }

        auto mask = scores > detection_threshold_;
        auto indices = mask.nonzero();
        auto selected_scores = scores.index({indices.select(1,0), indices.select(1,1), indices.select(1,2)});

        std::vector<torch::Tensor> keypoints_vec;
        std::vector<torch::Tensor> scores_vec;

        for (int i = 0; i < b; i++) {
            auto batch_mask = (indices.select(1,0) == i);
            auto yx = indices.index({batch_mask});
            auto kp_y = yx.select(1,1).to(torch::kFloat32);
            auto kp_x = yx.select(1,2).to(torch::kFloat32);
            auto kp = torch::stack({kp_x, kp_y}, 1);
            auto sc = selected_scores.index({batch_mask});

            if (max_num_keypoints_ > 0 && kp.size(0) > max_num_keypoints_) {
                auto topk = torch::topk(sc, max_num_keypoints_, 0, true, true);
                auto idx = std::get<1>(topk);
                kp = kp.index_select(0, idx);
                sc = sc.index_select(0, idx);
            }

            keypoints_vec.push_back(kp);
            scores_vec.push_back(sc);
        }

        auto cDa = torch::relu(convDa->forward(x));
        auto descriptors = convDb->forward(cDa);
        descriptors = torch::nn::functional::normalize(descriptors, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

        std::vector<torch::Tensor> desc_vec;
        for (int i = 0; i < b; i++) {
            auto kp = keypoints_vec[i].unsqueeze(0);
            auto d = descriptors.index({i}).unsqueeze(0);
            auto sampled = sample_descriptors(kp, d, 8);
            sampled = sampled.squeeze(0).transpose(1,2).contiguous();
            desc_vec.push_back(sampled);
        }

        auto kp_stacked = torch::stack(keypoints_vec, 0);
        auto sc_stacked = torch::stack(scores_vec, 0);
        auto desc_stacked = torch::stack(desc_vec, 0);

        SuperPointOutput output;
        output.keypoints = kp_stacked;
        output.scores = sc_stacked;
        output.descriptors = desc_stacked;
        return output;
    }

    torch::Tensor SuperPoint::simple_nms(torch::Tensor scores, int nms_radius) {
        if (nms_radius < 0)
            return scores;

        int pool_k = nms_radius * 2 + 1;
        auto zeros = torch::zeros_like(scores);

        auto max_pool = [&](torch::Tensor x) {
            return torch::max_pool2d(
                x.unsqueeze(1),
                {pool_k, pool_k},
                {1, 1},
                {nms_radius, nms_radius}
            ).unsqueeze(1);
        };

        auto max_mask = scores.eq(max_pool(scores));
        for (int i = 0; i < 2; i++) {
            auto supp_mask = max_pool(max_mask.to(torch::kFloat)) > 0;
            auto supp_scores = torch::where(supp_mask, zeros, scores);
            auto new_max_mask = supp_scores.eq(max_pool(supp_scores));
            max_mask = max_mask | (new_max_mask & (~supp_mask));
        }

        return torch::where(max_mask, scores, zeros);
    }

    torch::Tensor SuperPoint::sample_descriptors(torch::Tensor keypoints, torch::Tensor descriptors, int s) {
        auto b = descriptors.size(0);
        auto c = descriptors.size(1);
        auto h = descriptors.size(2);
        auto w = descriptors.size(3);

        auto kp = keypoints.clone();
        kp = kp - s/2.0 + 0.5;
        auto denom = torch::tensor({(double)(w*s - s/2.0 - 0.5), (double)(h*s - s/2.0 - 0.5)}).to(kp.device());
        kp = kp / denom.unsqueeze(0);
        kp = kp * 2 - 1;

        kp = kp.view({b,1,-1,2});
        auto sampled = torch::grid_sampler_2d(descriptors, kp, 0, 0, true);
        sampled = sampled.reshape({b,c,-1});
        sampled = torch::nn::functional::normalize(sampled, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        return sampled;
    }
}