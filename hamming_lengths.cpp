// cvflann::Hamming<unsigned char> against an exact byte-wise popcount, for vector lengths 1..64.
// The two vectors are equal in their first n bytes and differ in every bit after them,
// as the next row of a descriptor matrix usually does. The exact distance is therefore 0.
#include <opencv2/core/cvdef.h>
#ifndef CV_DbgAssert  // defined in core/base.hpp, which needs a build tree; used only by Hamming2, not tested here
#define CV_DbgAssert(expr)
#endif
#include <opencv2/flann/dist.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int exactHamming(const unsigned char* a, const unsigned char* b, size_t n)
{
    int d = 0;
    for (size_t i = 0; i < n; ++i)
        d += __builtin_popcount((unsigned)(a[i] ^ b[i]));
    return d;
}

int main()
{
#if defined(__ARM_NEON)
    std::printf("cvflann::Hamming path: NEON (__ARM_NEON defined)\n");
#else
    std::printf("cvflann::Hamming path: generic popcount (no __ARM_NEON)\n");
#endif
    cvflann::Hamming<unsigned char> hamming;
#ifndef ASAN_CASE
    std::printf("length,exact,flann,error_bits\n");
    int wrong = 0;
    for (size_t n = 1; n <= 64; ++n)
    {
        unsigned char a[96], b[96];
        for (int i = 0; i < 96; ++i) { a[i] = (unsigned char)(i * 37 + 11); b[i] = a[i]; }
        for (size_t i = n; i < 96; ++i) b[i] = (unsigned char)~a[i];
        const int exact = exactHamming(a, b, n);
        const int got = hamming(a, b, n);
        if (got != exact) ++wrong;
        std::printf("%zu,%d,%d,%d\n", n, exact, got, got - exact);
    }
    std::printf("lengths with a wrong distance: %d of 64\n", wrong);
    return 0;
#else
    // Two 61-byte vectors (the AKAZE MLDB descriptor size), each in a heap block of exactly 61 bytes.
    unsigned char* a = (unsigned char*)std::malloc(61);
    unsigned char* b = (unsigned char*)std::malloc(61);
    std::memset(a, 0x0F, 61);
    std::memset(b, 0xF0, 61);
    std::printf("exact distance %d\n", exactHamming(a, b, 61));
    std::fflush(stdout);
    const int got = hamming(a, b, 61);
    std::printf("cvflann::Hamming distance %d\n", got);
    std::free(a); std::free(b);
    return 0;
#endif
}
