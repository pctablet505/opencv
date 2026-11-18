#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <cstdio>

int main()
{
    // a and b agree in their first 61 bytes, so their Hamming distance over 61 bytes is 0
    unsigned char a[64], b[64];
    for (int i = 0; i < 64; i++)
    {
        a[i] = (unsigned char)i;
        b[i] = (i < 61) ? a[i] : (unsigned char)~a[i];
    }
    int d = cvflann::Hamming<unsigned char>()(a, b, 61);
    std::printf("distance %d (expected 0)\n", d);
    return 0;
}
