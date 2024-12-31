#include "LightGlue.h"

using namespace tensorrt_log;

LightGlue::LightGlue(LightGlueConfig config) :
    config_(config),
    engine_(nullptr)
{
    setReportableSeverity(config_.logSeverity);
}

int LightGlue::build() {
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

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
        return 6;

    auto profileStream = tensorrt_common::makeCudaStream();
    if (!profileStream)
        return 7;

    config->setProfileStream(*profileStream);

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

bool LightGlue::deserializeEngine() {
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

bool LightGlue::constructNetwork(
    std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network,
    std::unique_ptr<nvinfer1::IBuilderConfig>& config,
    std::unique_ptr<nvonnxparser::IParser>& parser
) const {
    auto parsed = parser->parseFromFile(config_.onnxFilePath.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
        return false;

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config_.memoryPoolLimit);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    if (config_.useDlaCore)
        tensorrt_common::enableDLA(builder.get(), config.get(), config_.dlaCore);

    return true;
}

void LightGlue::saveEngine() {
    if (engine_) {
        nvinfer1::IHostMemory* data = engine_->serialize();
        std::ofstream file(config_.engineFilePath, std::ios::binary);

        if (!file)
            return;

        file.write(reinterpret_cast<const char*>(data->data()), data->size());
    }
}

bool LightGlue::infer(
    const std::vector<Eigen::Matrix<double, 259, Eigen::Dynamic>>& features,
    std::vector<cv::Point>& kpts0,
    std::vector<cv::Point>& kpts1,
    Eigen::VectorXd& scores,
    const int imgHeight,
    const int imgWidth
) {
    if (!context_) {
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_)
            return false;
    }

    std::unordered_map<std::string, size_t> memoryMapping; // force allocation to max keypoints
    memoryMapping[config_.outputTensorNames[0]] = 3 * MAX_KEYPOINTS; // assume idx 0 is matches
    memoryMapping[config_.outputTensorNames[1]] = MAX_KEYPOINTS;
    
    tensorrt_buffer::BufferManager buffers(engine_, memoryMapping, 0, context_.get());

    for (int32_t i = 0, e = engine_->getNbIOTensors(); i < e; i++) {
        auto const name = engine_->getIOTensorName(i);
        context_->setTensorAddress(name, buffers.getDeviceBuffer(name));    
    }

    ASSERT(config_.inputTensorNames.size() == 2);

    if (!processInput(buffers, features, imgHeight, imgWidth))
        return false;

    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status)
        return false;

    buffers.copyOutputToHost();

    if (!processOutput(buffers, features, kpts0, kpts1, scores))
        return false;

    return true;
}

bool LightGlue::processInput(
    const tensorrt_buffer::BufferManager& buffers, 
    const std::vector<Eigen::Matrix<double, 259, Eigen::Dynamic>>& features,
    int imgHeight,
    int imgWidth
) {
    float* keypoints = static_cast<float*>(buffers.getHostBuffer(config_.inputTensorNames[0]));
    float* descriptors = static_cast<float*>(buffers.getHostBuffer(config_.inputTensorNames[1]));
    // keypoints [2, 1024, 2]
    // descriptors [2, 1024, 256]

    float shiftX = imgWidth / 2.f;
    float shiftY = imgHeight / 2.f;
    float scale = (imgHeight > imgWidth ? imgHeight : imgWidth) / 2;

    for (int i = 0; i < 2; i++) {
        for (int point = 0; point < MAX_KEYPOINTS; point++) {
            double x = features[i](1, point);
            double y = features[i](2, point);

            float normX = (static_cast<float>(x) - shiftX) / scale;
            float normY = (static_cast<float>(y) - shiftY) / scale;

            keypoints[i * MAX_KEYPOINTS * 2 + point * 2] = normX;
            keypoints[i * MAX_KEYPOINTS * 2 + point * 2 + 1] = normY;

            for (int d = 0; d < 256; d++)
                descriptors[i * MAX_KEYPOINTS * 256 + point * 256 + d] = features[i](3 + d, point);
        }
    }

    return true;
}

bool LightGlue::processOutput(
    const tensorrt_buffer::BufferManager& buffers, 
    const std::vector<Eigen::Matrix<double, 259, Eigen::Dynamic>>& features,
    std::vector<cv::Point>& kpts0,
    std::vector<cv::Point>& kpts1,
    Eigen::VectorXd& scores
) {
    float* outMatches = static_cast<float*>(buffers.getHostBuffer(config_.outputTensorNames[0]));
    float* outScores = static_cast<float*>(buffers.getHostBuffer(config_.outputTensorNames[1]));

    kpts0.clear();
    kpts1.clear();
    scores.resize(0);

    int match = 0;

    for (match = 0; match < MAX_KEYPOINTS; match++) {
        if (outMatches[match * 3] != 0)
            break;

        float score = outScores[match];
        int img0KptIdx = static_cast<int>(outMatches[match * 3 + 1]);
        int img1KptIdx = static_cast<int>(outMatches[match * 3 + 2]);

        cv::Point kp0(
            static_cast<int>(features[0](1, img0KptIdx)),
            static_cast<int>(features[0](2, img0KptIdx))
        );

        cv::Point kp1(
            static_cast<int>(features[1](1, img1KptIdx)),
            static_cast<int>(features[1](2, img1KptIdx))
        );

        kpts0.emplace_back(kp0);
        kpts1.emplace_back(kp1);

        scores.conservativeResize(scores.size() + 1);
        scores[scores.size() - 1] = score;
    }

    return true;
}

cv::Scalar getJetColor(double score, double minScore, double maxScore) {
    double normalized = (score - minScore) / (maxScore - minScore);
    normalized = std::max(0.0, std::min(1.0, normalized));

    double r = std::max(0.0, std::min(1.0, 1.5 - std::abs(4.0 * normalized - 3.0)));
    double g = std::max(0.0, std::min(1.0, 1.5 - std::abs(4.0 * normalized - 2.0)));
    double b = std::max(0.0, std::min(1.0, 1.5 - std::abs(4.0 * normalized - 1.0)));

    return cv::Scalar(b * 255, g * 255, r * 255);
}

void LightGlue::visualize(
    const cv::Mat& img0,
    const cv::Mat& img1,
    const std::vector<cv::Point> kpts0, 
    const std::vector<cv::Point> kpts1,
    const Eigen::VectorXd scores,
    const std::string& outImgPath,
    const float scaleFactor
) {
    cv::Mat resizedImg0, resizedImg1;
    cv::resize(img0, resizedImg0, cv::Size(), scaleFactor, scaleFactor);
    cv::resize(img1, resizedImg1, cv::Size(), scaleFactor, scaleFactor);

    std::vector<cv::Point> scaledKpts0, scaledKpts1;
    for (const auto& kp : kpts0)
        scaledKpts0.emplace_back(kp * scaleFactor);
    for (const auto& kp : kpts1)
        scaledKpts1.emplace_back(kp * scaleFactor);

    int height = std::max(resizedImg0.rows, resizedImg1.rows);
    int width = resizedImg0.cols + resizedImg1.cols;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    resizedImg0.copyTo(canvas(cv::Rect(0, 0, resizedImg0.cols, resizedImg0.rows)));
    resizedImg1.copyTo(canvas(cv::Rect(resizedImg0.cols, 0, resizedImg1.cols, resizedImg1.rows)));

    for (auto& kp : scaledKpts1) {
        kp.x += resizedImg0.cols;
    }

    double minScore = scores.minCoeff();
    double maxScore = scores.maxCoeff();

    for (size_t i = 0; i < scaledKpts0.size(); ++i) {
        if (i >= scaledKpts1.size() || i >= scores.size()) continue;

        cv::Scalar color = getJetColor(scores[i], minScore, maxScore);

        cv::circle(canvas, scaledKpts0[i], 1, color, cv::FILLED);
        cv::circle(canvas, scaledKpts1[i], 1, color, cv::FILLED);

        cv::line(canvas, scaledKpts0[i], scaledKpts1[i], color, 1);
    }

    cv::imwrite(outImgPath, canvas);
}