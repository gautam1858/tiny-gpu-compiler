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

const vectorAddTrace: CompilationTrace = {
  source: vectorAddSource,
  stages: [
    {
      name: 'Frontend \u2192 TinyGPU Dialect',
      ir: `tinygpu.func @vector_add() {
  %0 = tinygpu.block_id : i8
  %1 = tinygpu.block_dim : i8
  %2 = tinygpu.mul %0, %1 : i8
  %3 = tinygpu.thread_id : i8
  %4 = tinygpu.add %2, %3 : i8
  %5 = tinygpu.load %4 : i8
  %6 = tinygpu.const 64 : i8
  %7 = tinygpu.add %6, %4 : i8
  %8 = tinygpu.load %7 : i8
  %9 = tinygpu.add %5, %8 : i8
  %10 = tinygpu.const 128 : i8
  %11 = tinygpu.add %10, %4 : i8
  tinygpu.store %11, %9 : i8
  tinygpu.ret
}`,
    },
    {
      name: 'Register Allocation',
      ir: `tinygpu.func @vector_add() {
  %0 = tinygpu.block_id {rd=13} : i8
  %1 = tinygpu.block_dim {rd=14} : i8
  %2 = tinygpu.mul %0, %1 {rd=0, rs=13, rt=14} : i8
  %3 = tinygpu.thread_id {rd=15} : i8
  %4 = tinygpu.add %2, %3 {rd=0, rs=0, rt=15} : i8
  %5 = tinygpu.load %4 {rd=1, rs=0} : i8
  %6 = tinygpu.const 64 {rd=2} : i8
  %7 = tinygpu.add %6, %4 {rd=2, rs=2, rt=0} : i8
  %8 = tinygpu.load %7 {rd=2, rs=2} : i8
  %9 = tinygpu.add %5, %8 {rd=1, rs=1, rt=2} : i8
  %10 = tinygpu.const 128 {rd=2} : i8
  %11 = tinygpu.add %10, %4 {rd=2, rs=2, rt=0} : i8
  tinygpu.store %11, %9 {rs=2, rt=1} : i8
  tinygpu.ret
}`,
    },
  ],
  binary: {
    instructions: [
      { addr: 0, hex: '0x50DE', asm: 'MUL R0, %blockIdx, %blockDim', bits: '0101 0000 1101 1110' },
      { addr: 1, hex: '0x300F', asm: 'ADD R0, R0, %threadIdx', bits: '0011 0000 0000 1111' },
      { addr: 2, hex: '0x7100', asm: 'LDR R1, [R0]', bits: '0111 0001 0000 0000' },
      { addr: 3, hex: '0x9240', asm: 'CONST R2, #64', bits: '1001 0010 0100 0000' },
      { addr: 4, hex: '0x3220', asm: 'ADD R2, R2, R0', bits: '0011 0010 0010 0000' },
      { addr: 5, hex: '0x7220', asm: 'LDR R2, [R2]', bits: '0111 0010 0010 0000' },
      { addr: 6, hex: '0x3112', asm: 'ADD R1, R1, R2', bits: '0011 0001 0001 0010' },
      { addr: 7, hex: '0x9280', asm: 'CONST R2, #128', bits: '1001 0010 1000 0000' },
      { addr: 8, hex: '0x3220', asm: 'ADD R2, R2, R0', bits: '0011 0010 0010 0000' },
      { addr: 9, hex: '0x8021', asm: 'STR [R2], R1', bits: '1000 0000 0010 0001' },
      { addr: 10, hex: '0xF000', asm: 'RET', bits: '1111 0000 0000 0000' },
    ],
  },
};

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

const matrixMultiplyTrace: CompilationTrace = {
  source: matrixMultiplySource,
  stages: [
    {
      name: 'Frontend \u2192 TinyGPU Dialect',
      ir: `tinygpu.func @matrix_multiply() {
  %bid = tinygpu.block_id : i8
  %bdim = tinygpu.block_dim : i8
  %tid = tinygpu.thread_id : i8
  %idx = tinygpu.mul %bid, %bdim : i8
  %idx2 = tinygpu.add %idx, %tid : i8
  // N loaded from address 192
  %naddr = tinygpu.const 192 : i8
  %N = tinygpu.load %naddr : i8
  %row = tinygpu.div %idx2, %N : i8
  %tmp = tinygpu.mul %row, %N : i8
  %col = tinygpu.sub %idx2, %tmp : i8
  %sum = tinygpu.const 0 : i8
  // ... loop body with branches ...
  tinygpu.ret
}`,
    },
    {
      name: 'Register Allocation',
      ir: `tinygpu.func @matrix_multiply() {
  // R0=idx, R1=N, R2=row, R3=col, R4=sum, R5=k, R6-R8=temps
  // R13=%blockIdx, R14=%blockDim, R15=%threadIdx
  // ... register-allocated IR ...
  tinygpu.ret
}`,
    },
  ],
  binary: {
    instructions: [
      { addr: 0, hex: '0x50DE', asm: 'MUL R0, %blockIdx, %blockDim', bits: '0101 0000 1101 1110' },
      { addr: 1, hex: '0x300F', asm: 'ADD R0, R0, %threadIdx', bits: '0011 0000 0000 1111' },
      { addr: 2, hex: '0x91C0', asm: 'CONST R1, #192', bits: '1001 0001 1100 0000' },
      { addr: 3, hex: '0x7110', asm: 'LDR R1, [R1]', bits: '0111 0001 0001 0000' },
      { addr: 4, hex: '0x6201', asm: 'DIV R2, R0, R1', bits: '0110 0010 0000 0001' },
      { addr: 5, hex: '0x5321', asm: 'MUL R3, R2, R1', bits: '0101 0011 0010 0001' },
      { addr: 6, hex: '0x4303', asm: 'SUB R3, R0, R3', bits: '0100 0011 0000 0011' },
      { addr: 7, hex: '0x9400', asm: 'CONST R4, #0', bits: '1001 0100 0000 0000' },
      { addr: 8, hex: '0x9500', asm: 'CONST R5, #0', bits: '1001 0101 0000 0000' },
      { addr: 9, hex: '0x2051', asm: 'CMP R5, R1', bits: '0010 0000 0101 0001' },
      { addr: 10, hex: '0x1418', asm: 'BRnzp 2, #24', bits: '0001 0100 0001 1000' },
      { addr: 11, hex: '0x5621', asm: 'MUL R6, R2, R1', bits: '0101 0110 0010 0001' },
      { addr: 12, hex: '0x3665', asm: 'ADD R6, R6, R5', bits: '0011 0110 0110 0101' },
      { addr: 13, hex: '0x7660', asm: 'LDR R6, [R6]', bits: '0111 0110 0110 0000' },
      { addr: 14, hex: '0x5751', asm: 'MUL R7, R5, R1', bits: '0101 0111 0101 0001' },
      { addr: 15, hex: '0x3773', asm: 'ADD R7, R7, R3', bits: '0011 0111 0111 0011' },
      { addr: 16, hex: '0x9840', asm: 'CONST R8, #64', bits: '1001 1000 0100 0000' },
      { addr: 17, hex: '0x3787', asm: 'ADD R7, R8, R7', bits: '0011 0111 1000 0111' },
      { addr: 18, hex: '0x7770', asm: 'LDR R7, [R7]', bits: '0111 0111 0111 0000' },
      { addr: 19, hex: '0x5667', asm: 'MUL R6, R6, R7', bits: '0101 0110 0110 0111' },
      { addr: 20, hex: '0x3446', asm: 'ADD R4, R4, R6', bits: '0011 0100 0100 0110' },
      { addr: 21, hex: '0x9601', asm: 'CONST R6, #1', bits: '1001 0110 0000 0001' },
      { addr: 22, hex: '0x3556', asm: 'ADD R5, R5, R6', bits: '0011 0101 0101 0110' },
      { addr: 23, hex: '0x1E09', asm: 'JMP #9', bits: '0001 1110 0000 1001' },
      { addr: 24, hex: '0x9680', asm: 'CONST R6, #128', bits: '1001 0110 1000 0000' },
      { addr: 25, hex: '0x3660', asm: 'ADD R6, R6, R0', bits: '0011 0110 0110 0000' },
      { addr: 26, hex: '0x8064', asm: 'STR [R6], R4', bits: '1000 0000 0110 0100' },
      { addr: 27, hex: '0xF000', asm: 'RET', bits: '1111 0000 0000 0000' },
    ],
  },
};

// New examples -- compiled at load time using the in-browser compiler

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

export const EXAMPLES: Example[] = [
  {
    name: 'Vector Add',
    description: 'c[i] = a[i] + b[i]',
    source: vectorAddSource,
    trace: vectorAddTrace,
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
    name: 'Matrix Multiply',
    description: 'C = A * B (2x2)',
    source: matrixMultiplySource,
    trace: matrixMultiplyTrace,
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
      // input = [1, 2, 3, 4, 5, 6, 7, 8] at addresses 0-63
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      // weights (kernel) = [1, 2, 1] at addresses 64-127
      mem[64] = 1; mem[65] = 2; mem[66] = 1;
      // K = 3 (kernel size, scalar at address 192)
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
      // a = [2, 3, 4, 5, 1, 2, 3, 4]
      const a = [2, 3, 4, 5, 1, 2, 3, 4];
      for (let i = 0; i < a.length; i++) mem[i] = a[i];
      // b = [5, 4, 3, 2, 6, 7, 8, 9]
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
      // input with mix of zero and positive values (unsigned 8-bit)
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
      // x = [1, 2, 3, 4, 5, 6, 7, 8]
      for (let i = 0; i < 8; i++) mem[i] = i + 1;
      // y = [10, 20, 30, 40, 50, 60, 70, 80]
      for (let i = 0; i < 8; i++) mem[64 + i] = (i + 1) * 10;
      // a = 3 (scalar at address 192)
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
      // a = [3, 7, 2, 9, 1, 8, 4, 6]
      const a = [3, 7, 2, 9, 1, 8, 4, 6];
      for (let i = 0; i < a.length; i++) mem[i] = a[i];
      // b = [5, 4, 6, 1, 8, 3, 7, 2]
      const b = [5, 4, 6, 1, 8, 3, 7, 2];
      for (let i = 0; i < b.length; i++) mem[64 + i] = b[i];
      return mem;
    })(),
    numBlocks: 2,
    threadsPerBlock: 4,
  },
];
