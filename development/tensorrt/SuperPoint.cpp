#include "SuperPoint.h"

using namespace tensorrt_log;

SuperPoint::SuperPoint(SuperPointConfig config) :
    config_(config),
    engine_(nullptr)
{
    setReportableSeverity(Logger::Severity::kVERBOSE);
}

int SuperPoint::build() {
    if (deserializeEngine())
        return 0;

    std::unique_ptr<nvinfer1::IBuilder> builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
        return 1;

    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
        return 2;

    std::unique_ptr<nvinfer1::IBuilderConfig> config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
        return 3;

    std::unique_ptr<nvonnxparser::IParser> parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
        return 4;

    auto profile = builder->createOptimizationProfile();
    if (!profile)
        return 5;

    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 100, 100));
    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, 500, 500));
    profile->setDimensions(config_.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, 1500, 1500));
    config->addOptimizationProfile(profile);

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
        return 6;

    cudaStream_t profileStream;
    cudaStreamCreate(&profileStream);
    if (!profileStream)
        return 7;

    config->setProfileStream(profileStream);

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
        return 8;

    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime)
        return 9;

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_)
        return 10;
    
    saveEngine();
    return 0;
}

bool SuperPoint::infer(const cv::Mat& img, Eigen::Matrix<double, 259, Eigen::Dynamic>& features) {
    if (!context_) {
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_)
            return false;
    }

    tensorrt_buffer::BufferManager buffers(engine_);

    for (int32_t i = 0, e = engine_->getNbIOTensors(); i < e; i++) {
        auto const name = engine_->getIOTensorName(i);
        context_->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    ASSERT(config_.inputTensorNames.size() == 1);

    if (!processInput(buffers, img))
        return false;

    buffers.copyInputToDevice();
    
    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status)
        return false;

    buffers.copyOutputToHost();
    if (!processOutput(buffers, features))
        return false;
    
    return true;
}

bool SuperPoint::deserializeEngine() {
    std::ifstream file(config_.engineFilePath.c_str(), std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        char *model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        if (runtime == nullptr) return false;
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
        if (engine_ == nullptr) return false;
        return true;
    }

    return false;
}

bool SuperPoint::constructNetwork(
    std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network,
    std::unique_ptr<nvinfer1::IBuilderConfig>& config,
    std::unique_ptr<nvonnxparser::IParser>& parser
) const {
    auto parsed = parser->parseFromFile(config_.onnxFilePath.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
        return false;

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512_MiB);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    tensorrt_common::enableDLA(builder.get(), config.get(), config_.dlaCore);

    return true;
}

void SuperPoint::saveEngine() {
    if (engine_) {
        nvinfer1::IHostMemory* data = engine_->serialize();
        std::ofstream file(config_.engineFilePath, std::ios::binary);

        if (!file)
            return;

        file.write(reinterpret_cast<const char*>(data->data()), data->size());
    }
}

bool SuperPoint::processInput(const tensorrt_buffer::BufferManager& buffers, const cv::Mat& img) {
    inputDims_.d[2] = img.rows;
    inputDims_.d[3] = img.cols;
    semiDims_.d[1] = img.rows;
    semiDims_.d[2] = img.cols;
    descDims_.d[1] = 256;
    descDims_.d[2] = img.rows / 8;
    descDims_.d[3] = img.cols / 8;
    auto* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(config_.inputTensorNames[0]));

    for (int row = 0; row < img.rows; ++row)
        for (int col = 0; col < img.cols; ++col)
            hostDataBuffer[row * img.cols + col] = float(img.at<unsigned char>(row, col)) / 255.f;

    return true;
}

bool SuperPoint::processOutput(const tensorrt_buffer::BufferManager& buffers, Eigen::Matrix<double, 259, Eigen::Dynamic>& features) {
    keypoints_.clear();
    descriptors_.clear();

    float* outScore = static_cast<float*>(buffers.getHostBuffer(config_.outputTensorNames[0]));
    float* outDesc = static_cast<float*>(buffers.getHostBuffer(config_.outputTensorNames[1]));

    int semiFeatureMapH = semiDims_.d[1];
    int semiFeatureMapW = semiDims_.d[2];

    std::vector<float> scoresVec(outScore, outScore + semiFeatureMapH * semiFeatureMapW);
    findHighScoreIndex(keypoints_, scoresVec, semiFeatureMapH, semiFeatureMapW, config_.confThreshold);
    removeBorders(keypoints_, scoresVec, config_.removeBorders, semiFeatureMapH, semiFeatureMapW);
    topKKeypoints(keypoints_, scoresVec, config_.maxKeypoints);

    features.resize(259, scoresVec.size());
    int descFeatureDim = descDims_.d[1];
    int descFeatureMapH = descDims_.d[2];
    int descFeatureMapW = descDims_.d[3];

    sampleDescriptors(keypoints_, outDesc, descriptors_, descFeatureDim, descFeatureMapH, descFeatureMapW);

    for (int i = 0; i < scoresVec.size(); i++)
        features(0, i) = scoresVec[i];

    for (int i = 1; i < 3; ++i)
        for (int j = 0; j < keypoints_.size(); ++j)
            features(i, j) = keypoints_[j][i - 1];

    for (int m = 3; m < 259; ++m)
        for (int n = 0; n < descriptors_.size(); ++n)
            features(m, n) = descriptors_[n][m-3];

    return true;
}

void SuperPoint::findHighScoreIndex(
    std::vector<std::vector<int>>& keypoints, 
    std::vector<float>& scores, 
    int h, 
    int w, 
    double threshold
) {
    std::vector<float> newScores;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            std::vector<int> location = {int(i / w), i % w};
            keypoints.emplace_back(location);
            newScores.push_back(scores[i]);
        }
    }

    scores.swap(newScores);
}

void SuperPoint::topKKeypoints(
    std::vector<std::vector<int>>& keypoints, 
    std::vector<float>& scores, 
    int k
) {
    if (k < keypoints.size() && k != -1) {
        std::vector<std::vector<int>> newKeypoints;
        std::vector<float> newScores;
        std::vector<size_t> indexes = sortIndexes(scores);

        for (int i = 0; i < k; ++i) {
            newKeypoints.push_back(keypoints[indexes[i]]);
            newScores.push_back(scores[indexes[i]]);
        }

        keypoints.swap(newKeypoints);
        scores.swap(newScores);
    }
}

std::vector<size_t> SuperPoint::sortIndexes(std::vector<float>& data) {
    std::vector<size_t> indexes(data.size());
    iota(indexes.begin(), indexes.end(), 0);
    sort(indexes.begin(), indexes.end(), [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    return indexes;
}

void SuperPoint::removeBorders(
    std::vector<std::vector<int>>& keypoints, 
    std::vector<float>& scores, 
    int border, 
    int h, 
    int w
) {
    std::vector<std::vector<int>> newKeypoints;
    std::vector<float> newScores;

    for (int i = 0; i < keypoints.size(); ++i) {
        bool flag_h = (keypoints[i][0] >= border) && (keypoints[i][0] < (h - border));
        bool flag_w = (keypoints[i][1] >= border) && (keypoints[i][1] < (w - border));

        if (flag_h && flag_w) {
            newKeypoints.push_back(std::vector<int>(keypoints[i][1], keypoints[i][0]));
            newScores.push_back(scores[i]);
        }
    }

    keypoints.swap(newKeypoints);
    scores.swap(newScores);
}

void normalize_keypoints(
    const std::vector<std::vector<int>> &keypoints, 
    std::vector<std::vector<double>> &keypoints_norm,
    int h, 
    int w, 
    int s
) {
    for (auto &keypoint : keypoints) {
        std::vector<double> kp = {keypoint[0] - s / 2 + 0.5, keypoint[1] - s / 2 + 0.5};
        kp[0] = kp[0] / (w * s - s / 2 - 0.5);
        kp[1] = kp[1] / (h * s - s / 2 - 0.5);
        kp[0] = kp[0] * 2 - 1;
        kp[1] = kp[1] * 2 - 1;
        keypoints_norm.push_back(kp);
    }
}

int clip(int val, int max) {
    if (val < 0) return 0;
    return std::min(val, max - 1);
}

void grid_sample(
    const float *input, std::vector<std::vector<double>> &grid,
    std::vector<std::vector<double>> &output, 
    int dim, 
    int h, 
    int w
) {
    // descriptors 1, 256, image_height/8, image_width/8
    // keypoints 1, 1, number, 2
    // out 1, 256, 1, number
    for (auto &g : grid) {
        double ix = ((g[0] + 1) / 2) * (w - 1);
        double iy = ((g[1] + 1) / 2) * (h - 1);

        int ix_nw = clip(std::floor(ix), w);
        int iy_nw = clip(std::floor(iy), h);

        int ix_ne = clip(ix_nw + 1, w);
        int iy_ne = clip(iy_nw, h);

        int ix_sw = clip(ix_nw, w);
        int iy_sw = clip(iy_nw + 1, h);

        int ix_se = clip(ix_nw + 1, w);
        int iy_se = clip(iy_nw + 1, h);

        double nw = (ix_se - ix) * (iy_se - iy);
        double ne = (ix - ix_sw) * (iy_sw - iy);
        double sw = (ix_ne - ix) * (iy - iy_ne);
        double se = (ix - ix_nw) * (iy - iy_nw);

        std::vector<double> descriptor;
        for (int i = 0; i < dim; ++i) {
            // 256x60x106 dhw
            // x * height * depth + y * depth + z
            float nw_val = input[i * h * w + iy_nw * w + ix_nw];
            float ne_val = input[i * h * w + iy_ne * w + ix_ne];
            float sw_val = input[i * h * w + iy_sw * w + ix_sw];
            float se_val = input[i * h * w + iy_se * w + ix_se];
            descriptor.push_back(nw_val * nw + ne_val * ne + sw_val * sw + se_val * se);
        }
        output.push_back(descriptor);
    }
}

template<typename Iter_T>
double vector_normalize(Iter_T first, Iter_T last) {
    return sqrt(inner_product(first, last, first, 0.0));
}

void normalize_descriptors(std::vector<std::vector<double>> &dest_descriptors) {
    for (auto &descriptor : dest_descriptors) {
        double norm_inv = 1.0 / vector_normalize(descriptor.begin(), descriptor.end());
        std::transform(descriptor.begin(), descriptor.end(), descriptor.begin(),
                       std::bind1st(std::multiplies<double>(), norm_inv));
    }
}

void SuperPoint::sampleDescriptors(
    std::vector<std::vector<int>>& keypoints,
    float* descriptors,
    std::vector<std::vector<double>>& destDescriptors,
    int dim,
    int h,
    int w,
    int s
) {
    std::vector<std::vector<double>> normKeypoints;
    normalize_keypoints(keypoints, normKeypoints, h, w, s);
    grid_sample(descriptors, normKeypoints, destDescriptors, dim, h, w);
    normalize_descriptors(destDescriptors);
}