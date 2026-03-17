import { CompilationTrace } from '../compiler/types';
import { compileTGC } from '../compiler/TGCCompiler';

export interface Example {
  name: string;
  description: string;
  source: string;
  trace: CompilationTrace;
  initialMemory: number[];
  numBlocks: number;
  threadsPerBlock: number;
}

const vectorAddSource = `kernel vector_add(global int* a, global int* b, global int* c) {
    int idx = blockIdx * blockDim + threadIdx;
    c[idx] = a[idx] + b[idx];
}`;

const matrixMultiplySource = `kernel matrix_multiply(global int* A, global int* B, global int* C, int N) {
    int idx = blockIdx * blockDim + threadIdx;
    int row = idx / N;
    int col = idx - row * N;
    int sum = 0;
    for (int k = 0; k < N; k = k + 1) {
        int a_val = A[row * N + k];
        int b_val = B[k * N + col];
        sum = sum + a_val * b_val;
    }
    C[idx] = sum;
}`;

const dotProductSource = `kernel dot_product(global int* a, global int* b, global int* c) {
    int idx = blockIdx * blockDim + threadIdx;
    int ai = a[idx];
    int bi = b[idx];
    c[idx] = ai * bi;
}`;

const reluSource = `kernel relu(global int* input, global int* output) {
    int idx = blockIdx * blockDim + threadIdx;
    int val = input[idx];
    if (val > 0) {
        output[idx] = val;
    } else {
        output[idx] = 0;
    }
}`;

const saxpySource = `kernel saxpy(global int* x, global int* y, int a) {
    int idx = blockIdx * blockDim + threadIdx;
    int xi = x[idx];
    int yi = y[idx];
    y[idx] = a * xi + yi;
}`;

const conv1dSource = `kernel conv1d(global int* input, global int* weights, global int* output, int K) {
    int idx = blockIdx * blockDim + threadIdx;
    int sum = 0;
    for (int j = 0; j < K; j = j + 1) {
        int in_val = input[idx + j];
        int w_val = weights[j];
        sum = sum + in_val * w_val;
    }
    output[idx] = sum;
}`;

const vectorMaxSource = `kernel vector_max(global int* a, global int* b, global int* c) {
    int idx = blockIdx * blockDim + threadIdx;
    int ai = a[idx];
    int bi = b[idx];
    if (ai > bi) {
        c[idx] = ai;
    } else {
        c[idx] = bi;
    }
}`;

// NEW: Shared memory tiled vector add - demonstrates shared mem + syncthreads
const sharedTileAddSource = `kernel shared_tile_add(global int* a, global int* b, global int* c) {
    shared int tile_a[4];
    shared int tile_b[4];
    int idx = blockIdx * blockDim + threadIdx;
    tile_a[threadIdx] = a[idx];
    tile_b[threadIdx] = b[idx];
    __syncthreads();
    int sum = tile_a[threadIdx] + tile_b[threadIdx];
    c[idx] = sum;
}`;

// NEW: Shared memory reduction - demonstrates shared mem cooperation
const sharedReductionSource = `kernel shared_reduce(global int* input, global int* output) {
    shared int scratch[4];
    int idx = blockIdx * blockDim + threadIdx;
    scratch[threadIdx] = input[idx];
    __syncthreads();
    if (threadIdx < 2) {
        scratch[threadIdx] = scratch[threadIdx] + scratch[threadIdx + 2];
    }
    __syncthreads();
    if (threadIdx < 1) {
        scratch[0] = scratch[0] + scratch[1];
        output[blockIdx] = scratch[0];
    }
}`;

export const EXAMPLES: Example[] = [
  {
    name: 'Vector Add',
    description: 'c[i] = a[i] + b[i]',
    source: vectorAddSource,
    trace: compileTGC(vectorAddSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      for (let i = 0; i < 8; i++) mem[64 + i] = (i + 1) * 10;
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'Shared Tile Add',
    description: 'Tiled add with shared memory',
    source: sharedTileAddSource,
    trace: compileTGC(sharedTileAddSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      for (let i = 0; i < 8; i++) mem[64 + i] = (i + 1) * 10;
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'Shared Reduce',
    description: 'Parallel reduction with shared memory',
    source: sharedReductionSource,
    trace: compileTGC(sharedReductionSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      // input = [1, 2, 3, 4, 5, 6, 7, 8]
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'Matrix Multiply',
    description: 'C = A * B (2x2)',
    source: matrixMultiplySource,
    trace: compileTGC(matrixMultiplySource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      mem[0] = 1; mem[1] = 2; mem[2] = 3; mem[3] = 4;
      mem[64] = 5; mem[65] = 6; mem[66] = 7; mem[67] = 8;
      mem[192] = 2;
      return mem;
    })(),
    numBlocks: 1,
    threadsPerBlock: 4,
  },
  {
    name: '1D Convolution',
    description: 'output[i] = sum(input[i+j] * w[j])',
    source: conv1dSource,
    trace: compileTGC(conv1dSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      mem[64] = 1; mem[65] = 2; mem[66] = 1;
      mem[192] = 3;
      return mem;
    })(),
    numBlocks: 1,
    threadsPerBlock: 4,
  },
  {
    name: 'Dot Product',
    description: 'c[i] = a[i] * b[i]',
    source: dotProductSource,
    trace: compileTGC(dotProductSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      const a = [2, 3, 4, 5, 1, 2, 3, 4];
      for (let i = 0; i < a.length; i++) mem[i] = a[i];
      const b = [5, 4, 3, 2, 6, 7, 8, 9];
      for (let i = 0; i < b.length; i++) mem[64 + i] = b[i];
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'ReLU Activation',
    description: 'output[i] = max(0, input[i])',
    source: reluSource,
    trace: compileTGC(reluSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      const input = [0, 5, 0, 12, 0, 3, 0, 7];
      for (let i = 0; i < input.length; i++) mem[i] = input[i];
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'SAXPY',
    description: 'y[i] = a * x[i] + y[i]',
    source: saxpySource,
    trace: compileTGC(saxpySource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      for (let i = 0; i < 8; i++) mem[64 + i] = (i + 1) * 10;
      mem[192] = 3;
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
  {
    name: 'Vector Max',
    description: 'c[i] = max(a[i], b[i])',
    source: vectorMaxSource,
    trace: compileTGC(vectorMaxSource),
    initialMemory: (() => {
      const mem = new Array(256).fill(0);
      const a = [3, 7, 2, 9, 1, 8, 4, 6];
      for (let i = 0; i < a.length; i++) mem[i] = a[i];
      const b = [5, 4, 6, 1, 8, 3, 7, 2];
      for (let i = 0; i < b.length; i++) mem[64 + i] = b[i];
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
];
