// AKAZE descriptors (61 bytes each) matched with FlannBasedMatcher(LSH).
// Every reported distance is compared with the exact Hamming distance of the same two descriptor rows.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <cstdio>
#include <cstdlib>

int main()
{
    cv::Mat img(480, 640, CV_8UC1);
    cv::RNG rng(12345);
    rng.fill(img, cv::RNG::UNIFORM, 0, 64);
    for (int i = 0; i < 300; ++i)
    {
        cv::Point p(rng.uniform(0, 640), rng.uniform(0, 480));
        cv::rectangle(img, p, p + cv::Point(rng.uniform(5, 40), rng.uniform(5, 40)), cv::Scalar(rng.uniform(128, 256)), -1);
    }
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
    cv::Mat shifted;
    cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 7.5, 0, 1, -4.25);
    cv::warpAffine(img, shifted, shift, img.size());

    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    std::vector<cv::KeyPoint> k1, k2;
    cv::Mat d1, d2;
    akaze->detectAndCompute(img, cv::noArray(), k1, d1);
    akaze->detectAndCompute(shifted, cv::noArray(), k2, d2);
    std::printf("AKAZE descriptors: %d and %d rows of %d bytes\n", d1.rows, d2.rows, d1.cols);

    cv::FlannBasedMatcher flann(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    std::vector<cv::DMatch> matches;
    flann.match(d1, d2, matches);
    int wrong = 0, maxErr = 0;
    double sumErr = 0;
    for (size_t i = 0; i < matches.size(); ++i)
    {
        const cv::DMatch& m = matches[i];
        const int exact = (int)cv::norm(d1.row(m.queryIdx), d2.row(m.trainIdx), cv::NORM_HAMMING);
        const int err = (int)m.distance - exact;
        if (err != 0)
        {
            if (wrong < 3)
                std::printf("  query %d -> train %d: FLANN distance %d, exact %d\n", m.queryIdx, m.trainIdx, (int)m.distance, exact);
            ++wrong;
            sumErr += err;
            maxErr = std::max(maxErr, std::abs(err));
        }
    }
    std::printf("FlannBasedMatcher(LSH): %d matches, %d with a wrong distance (largest error %d bits, mean %.1f)\n",
                (int)matches.size(), wrong, maxErr, wrong ? sumErr / wrong : 0.0);
    return 0;
}
