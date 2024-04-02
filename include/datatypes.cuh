#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

/*
    Build Configurations
*/
//#define DEBUG_BUILD


// Width of kernels groups
#define CKW 256


// Default hash value for hash maps
#define DEFAULT_HASH 998244353


typedef unsigned int uint;

/*
    Type definitions for scalar and vector types in finite volume scheme
*/

#define FLOAT_PREC



#ifdef FLOAT_PREC
    typedef float scalar;
    typedef float3 vector;
    #define FVM_TOL 1e-6

    #define READ_FORMAT "f"
#else
    typedef double scalar;
    typedef double3 vector;
    #define FVM_TOL 1e-12
    #define READ_FORMAT "lf"
#endif

/*
    Type definition for a 3 by 3 tensor (derivative of vector type)
*/
typedef struct tensor {
    vector u, v, w;
} tensor;


/*
    Macros for branchless operators
*/
#define blmin(a, b) a * (a <= b) + b * (a > b)
#define blmax(a, b) a * (a >= b) + b * (a < b)

#define blsign(a) -1.0 * (a < 0.0) + (a > 0.0)

// Generic cuda macros

#define cudaExec(contents) cudaHandleError(contents, __FILE__, __LINE__);
#define cudaSync cudaExec(cudaDeviceSynchronize())

#define cudaGlobalId (blockIdx.x * blockDim.x) + threadIdx.x

#define cpuAlloc(type, size) (type*)malloc(sizeof(type) * (size))

#define gpuAlloc(loc, type, size) cudaExec(cudaMalloc((void **)&(loc), sizeof(type) * (size)))

#define kernelInput(x) x->_gpu


inline int cuda_size(uint size, uint group_size) {
    return (int)ceil(((double)size) / ((double)group_size));
}

inline void cudaHandleError(cudaError_t error, const char* file, const int line) {
    if (error != cudaSuccess) {
        const char* error_str = cudaGetErrorString(error);
        printf("Error %d in file %s line %d, error log: %s\n", error, file, line, error_str);
        exit(error);
    }
}

/*
    float & double operations
*/
__host__ __device__ inline void reset(float& a) {
    a = 0;
}
__host__ __device__ inline void reset(double& a) {
    a = 0;
}


/*
    float3 operations
*/
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator*(const float3& a, const float& b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
__host__ __device__ inline float3 operator/(const float3& a, const float& b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}
__host__ __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
__host__ __device__ inline float3& operator/=(float3& a, const float& b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}
__host__ __device__ inline float norm(const float3& a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Fast branchless sign function
__host__ __device__ inline float sign(const float& x) {
    return (float)(x >= 0.0) - (float)(x < 0.0);
}

__host__ __device__ inline float3 outer_normal(const float3& n, const float3& d) {
    return n * sign(dot(n, d));
}


__host__ __host__ __device__ inline float3 make_vector(const float& x, const float& y, const float& z) {
    return make_float3(x, y, z);
}

__host__ __device__ inline void reset(float3& a) {
    a.x = 0;
    a.y = 0;
    a.z = 0;
}




/*
    double3 operations
*/

__host__ __device__ inline double3 operator+(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline double3 operator*(const double3& a, const double& b) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}
__host__ __device__ inline double3 operator/(const double3& a, const double& b) {
    return make_double3(a.x / b, a.y / b, a.z / b);
}
__host__ __device__ inline double3& operator+=(double3& a, const double3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
__host__ __device__ inline double3& operator/=(double3& a, const double& b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}
__host__ __device__ inline double norm(const double3& a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
__host__ __device__ inline double dot(const double3& a, const double3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline double3 cross(const double3& a, const double3& b) {
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Fast branchless sign function
__host__ __device__ inline double sign(const double& x) {
    return (double)(x > 0) - (double)(x < 0);
}

__host__ __device__ inline double3 outer_normal(const double3& n, const double3& d) {
    return n * sign(dot(n, d));
}



#ifndef FLOAT_PREC
__host__ __host__ __device__ inline double3 make_vector(const double& x, const double& y, const double& z) {
    return make_double3(x, y, z);
}
#endif


__host__ __device__ inline void reset(double3& a) {
    a.x = 0;
    a.y = 0;
    a.z = 0;
}



// Vector operators

__host__ __device__ inline vector operator*(const scalar& a, const vector& b) {
    return b * a;
}
__host__ __device__ inline vector operator-(const vector& a) {
    return make_vector(
        -a.x, -a.y, -a.z
    );
}
__host__ __device__ inline vector operator*(const vector& a, const vector& b) {
    return make_vector(
        a.x * b.x, a.y * b.y, a.z * b.z
    );
}
__host__ __device__ inline vector& operator-=(vector& a, const vector& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__host__ __device__ inline vector minv(const vector& a, const vector& b) {
    return make_vector(
        fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z)
    );
}

__host__ __device__ inline vector maxv(const vector& a, const vector& b) {
    return make_vector(
        fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z)
    );
}


/*
    tensor operations
*/
/*
    Tensors are column major
        (u, v, w)
*/
__host__ __device__ inline tensor make_tensor(
    const scalar& ux, const scalar& vx, const scalar& wx,
    const scalar& uy, const scalar& vy, const scalar& wy,
    const scalar& uz, const scalar& vz, const scalar& wz
) {
    tensor t;
    t.u = make_vector(ux, uy, uz);
    t.v = make_vector(vx, vy, vz);
    t.w = make_vector(wx, wy, wz);
    return t;
}
__host__ __device__ inline tensor make_tensor(
    const vector& u, const vector& v, const vector& w
) {
    tensor t;
    t.u = u;
    t.v = v;
    t.w = w;
    return t;
}
__host__ __device__ inline vector operator*(const tensor& a, const vector& x) {
    return make_vector(
        a.u.x * x.x + a.v.x * x.y + a.w.x * x.z,
        a.u.y * x.x + a.v.y * x.y + a.w.y * x.z,
        a.u.z * x.x + a.v.z * x.y + a.w.z * x.z
    );
}
__host__ __device__ inline vector dot(const tensor& a, const vector& x) {
    return a * x;
}
__host__ __device__ inline tensor operator+(const tensor& a, const tensor& b) {
    return make_tensor(
        a.u + b.u, a.v + b.v, a.w + b.w
    );
}
__host__ __device__ inline tensor operator-(const tensor& a, const tensor& b) {
    return make_tensor(
        a.u - b.u, a.v - b.v, a.w - b.w
    );
}
__host__ __device__ inline tensor& operator+=(tensor& a, const tensor& b) {
    a.u += b.u;
    a.v += b.v;
    a.w += b.w;
    return a;
}
__host__ __device__ inline tensor operator*(const tensor& a, const scalar& b) {
    return make_tensor(
        a.u * b, a.v * b, a.w * b
    );
}
__host__ __device__ inline tensor operator*(const tensor& a, const tensor& b) {
    return make_tensor(
        a.u.x * b.u.x + a.v.x * b.u.y + a.w.x * b.u.z,
        a.u.y * b.u.x + a.v.y * b.u.y + a.w.y * b.u.z,
        a.u.z * b.u.x + a.v.z * b.u.y + a.w.z * b.u.z,

        a.u.x * b.v.x + a.v.x * b.v.y + a.w.x * b.v.z,
        a.u.y * b.v.x + a.v.y * b.v.y + a.w.y * b.v.z,
        a.u.z * b.v.x + a.v.z * b.v.y + a.w.z * b.v.z,

        a.u.x * b.w.x + a.v.x * b.w.y + a.w.x * b.w.z,
        a.u.y * b.w.x + a.v.y * b.w.y + a.w.y * b.w.z,
        a.u.z * b.w.x + a.v.z * b.w.y + a.w.z * b.w.z
    );
}
__host__ __device__ inline tensor operator/(const tensor& a, const scalar& b) {
    return make_tensor(
        a.u / b, a.v / b, a.w / b
    );
}

__host__ __device__ inline tensor outer(const vector& a, const vector& b) {
    return make_tensor(
        a * b.x, a *b.y, a * b.z
    );
}

__host__ __device__ inline tensor identity() {
    return make_tensor(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    );
}

__host__ __device__ inline scalar norm(const tensor& a) {
    return sqrt(dot(a.u, a.u) + dot(a.v, a.v) + dot(a.w, a.w));
}

__host__ __device__ inline void reset(tensor& a) {
    a.u.x = 0.0f; a.v.x = 0.0f; a.w.x = 0.0f;
    a.u.y = 0.0f; a.v.y = 0.0f; a.w.y = 0.0f;
    a.u.z = 0.0f; a.v.z = 0.0f; a.w.z = 0.0f;
}



// Limited square root
__host__ __device__ inline scalar lsqrt(const scalar& x) {
    return sqrt( fmax((scalar)1e-12, fmin((scalar)1e30, x)) );
}






