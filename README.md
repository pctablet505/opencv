Checks `cvflann::Hamming<unsigned char>` (modules/flann/include/opencv2/flann/dist.h) on ARM64 against x86-64.
The NEON branch loads 16 bytes per step with no tail handling, so for lengths that are not a multiple of 16 it reads
and counts bytes past the end of each vector. See the workflow runs of this branch.
