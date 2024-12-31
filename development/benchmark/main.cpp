#include "SuperPoint.h"
#include "LightGlue.h"

#include <iostream>
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <chrono>
#include <iomanip>

bool processImagePairs(
    const std::vector<std::string>& imagePaths,
    const std::shared_ptr<SuperPoint>& sp,
    const std::shared_ptr<LightGlue>& lg,
    int visualize,
    float imgFactor
);

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Invalid number of arguments! Usage: " << argv[0] << " <dir of jpgs> <n samples to visualize> <opt resize factor>\n";
        return 1;
    }

    std::string dir = argv[1];
    int visualize = atoi(argv[2]);
    float imgFactor = argc == 3 ? 1 : std::atof(argv[3]);

    std::vector<std::string> imgPaths;
    for (const auto& entry : std::filesystem::directory_iterator(dir))
        if (entry.path().extension() == ".jpg")
            imgPaths.push_back(entry.path().string());


    if (imgPaths.size() < 2) {
        std::cerr << "Not enough images!\n";
        return 1;
    }

    std::sort(imgPaths.begin(), imgPaths.end());

    SuperPointConfig spConfig;
    spConfig.onnxFilePath = "SuperPoint.onnx";

    std::shared_ptr<SuperPoint> sp = std::make_shared<SuperPoint>(spConfig);
    if (sp->build()) {
        std::cerr << "Something went wrong building or loading SuperPoint!\n";
        return 1;
    }

    LightGlueConfig lgConfig;
    lgConfig.onnxFilePath = "LightGlue.onnx";

    std::shared_ptr<LightGlue> lg = std::make_shared<LightGlue>(lgConfig);
    if (lg->build()) {
        std::cerr << "Something went wrong building or loading LightGlue!\n";
        return 1;
    }

    std::cout << "Running SuperPoint + LightGlue benchmark with " << imgPaths.size() << " images!\n";

    if (!processImagePairs(imgPaths, sp, lg, visualize, imgFactor)) {
        return 1;
    }

    return 0;
}

void displayProgressBar(size_t current, size_t total, size_t barWidth = 50) {
    double progress = static_cast<double>(current) / total;
    size_t pos = static_cast<size_t>(barWidth * progress);

    std::cout << "[";
    for (size_t i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% [" << current << "/" << total << "]\r";
    std::cout.flush();
}

bool processImagePairs(
    const std::vector<std::string>& imagePaths,
    const std::shared_ptr<SuperPoint>& sp,
    const std::shared_ptr<LightGlue>& lg,
    int visualize,
    float imgFactor
) {
    size_t totalPairs = 0;
    std::vector<std::chrono::milliseconds> totalElapsed;
    std::vector<int> matches;

    std::chrono::seconds benchmarkElapsed;
    auto benchStart = std::chrono::high_resolution_clock::now();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, imagePaths.size() - 2);

    cv::Mat sample = cv::imread(imagePaths[0]);
    cv::resize(sample, sample, cv::Size(), imgFactor, imgFactor);
    int h = sample.rows;
    int w = sample.cols;

    if (!std::filesystem::exists("bench_samp"))
        std::filesystem::create_directory("bench_samp");

    for (size_t i = 0; i < imagePaths.size() - 1; ++i) {
        displayProgressBar(i, imagePaths.size() - 1);

        cv::Mat img0 = cv::imread(imagePaths[i]);
        cv::resize(img0, img0, cv::Size(), imgFactor, imgFactor);
        cv::Mat img1 = cv::imread(imagePaths[i + 1]);
        cv::resize(img1, img1, cv::Size(), imgFactor, imgFactor);

        Eigen::Matrix<double, 259, Eigen::Dynamic> feature0;
        Eigen::Matrix<double, 259, Eigen::Dynamic> feature1;

        std::vector<cv::Point> kpts0;
        std::vector<cv::Point> kpts1;
        Eigen::VectorXd scores;

        if ((img0.rows != img1.rows) || (img0.rows != img1.rows)) {
            std::cerr << "Image pair size mismatch! (" << img0.rows << ", " << img0.cols << ") != (" << img1.rows << ", " << img1.cols << ")\n";
            std::cerr << "\t" << imagePaths[i] << "\n\t" << imagePaths[i + 1] << "\n";
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        if (!sp->infer(img0, feature0)) {
            std::cerr << "SuperPoint inference failed with image located at " << imagePaths[i] << "\n";
            return false;
        }

        if (!sp->infer(img1, feature1)) {
            std::cerr << "SuperPoint inference failed with image located at " << imagePaths[i + 1] << "\n";
            return false;
        }

        if (!lg->infer({feature0, feature1}, kpts0, kpts1, scores, img0.rows, img0.cols)) {
            std::cerr << "LightGlue inference failed with image located at " << imagePaths[i] << " and " << imagePaths[i + 1] << "\n";
            return false;
        }

        auto end = std::chrono::high_resolution_clock::now() - start;
        std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end);

        if (visualize > 0 && dis(gen) < visualize)
            lg->visualize(img0, img1, kpts0, kpts1, scores, "bench_samp/" + std::to_string(i) + ".jpg");

        totalPairs++;
        matches.push_back(kpts0.size());
        totalElapsed.push_back(elapsed);
    }

    benchmarkElapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - benchStart);

    displayProgressBar(imagePaths.size() - 1, imagePaths.size() - 1);
    std::cout << std::endl; 

    auto totalTime = std::accumulate(totalElapsed.begin(), totalElapsed.end(), std::chrono::milliseconds(0)).count();
    auto avgTime = totalTime / static_cast<double>(totalPairs);
    auto minTime = *std::min_element(totalElapsed.begin(), totalElapsed.end());
    auto maxTime = *std::max_element(totalElapsed.begin(), totalElapsed.end());
    double avgFps = 1.0 / (avgTime / 1e3);

    size_t totalMatches = std::accumulate(matches.begin(), matches.end(), 0);
    double avgMatches = static_cast<double>(totalMatches) / totalPairs;
    int minMatches = *std::min_element(matches.begin(), matches.end());
    int maxMatches = *std::max_element(matches.begin(), matches.end());

    std::cout << "\nBenchmark Elapsed " << benchmarkElapsed.count() << " seconds @ (" << h << ", " << w << ")\n";
    std::cout << "\tTotal Pairs         " << totalPairs;
    std::cout << "\n\tAverage Time        " << avgTime << " ms";
    std::cout << "\n\tAverage FPS         " << avgFps << " frames/second";
    std::cout << "\n\tMin/Max Time        " << minTime.count() << " / " << maxTime.count() << " ms";
    std::cout << "\n\tAverage Matches     " << avgMatches;
    std::cout << "\n\tMin/Max Matches     " << minMatches << " / " << maxMatches << std::endl;

    return true;
}