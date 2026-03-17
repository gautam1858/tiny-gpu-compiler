/** Represents a single 16-bit instruction */
export interface Instruction {
  addr: number;
  hex: string;
  asm: string;
  bits: string;
}

/** A single compilation stage showing IR at that point */
export interface CompilationStage {
  name: string;
  ir: string;
}

/** Warp divergence information for a branch instruction */
export interface DivergenceInfo {
  instructionAddr: number;
  type: 'branch' | 'converge';
  branchTaken: boolean[];  // per-thread: did they take the branch?
  description: string;
}

/** Memory coalescing analysis for a memory instruction */
export interface CoalescingInfo {
  instructionAddr: number;
  accessPattern: 'coalesced' | 'strided' | 'scattered';
  addresses: number[];       // per-thread addresses accessed
  transactionsNeeded: number;
  description: string;
}

/** Performance profiling metrics */
export interface PerformanceMetrics {
  totalInstructions: number;
  registersUsed: number;
  sharedMemoryBytes: number;
  branchInstructions: number;
  memoryInstructions: number;
  computeInstructions: number;
  barrierCount: number;
  estimatedCycles: number;
  computeToMemoryRatio: number;
  optimizationSummary: string;
}

/** Full analysis results from the compiler */
export interface AnalysisResult {
  divergence: DivergenceInfo[];
  coalescing: CoalescingInfo[];
  metrics: PerformanceMetrics;
}

/** Full compilation trace from the compiler */
export interface CompilationTrace {
  source: string;
  stages: CompilationStage[];
  binary: {
    instructions: Instruction[];
  };
  analysis?: AnalysisResult;
}

/** Opcode definitions matching tiny-gpu's decoder.sv */
export enum Opcode {
  NOP   = 0b0000,
  BRnzp = 0b0001,
  CMP   = 0b0010,
  ADD   = 0b0011,
  SUB   = 0b0100,
  MUL   = 0b0101,
  DIV   = 0b0110,
  LDR   = 0b0111,
  STR   = 0b1000,
  CONST = 0b1001,
  SLDR  = 0b1010,
  SSTR  = 0b1011,
  BAR   = 0b1100,
  RET   = 0b1111,
}

export const OPCODE_NAMES: Record<number, string> = {
  [Opcode.NOP]: 'NOP',
  [Opcode.BRnzp]: 'BRnzp',
  [Opcode.CMP]: 'CMP',
  [Opcode.ADD]: 'ADD',
  [Opcode.SUB]: 'SUB',
  [Opcode.MUL]: 'MUL',
  [Opcode.DIV]: 'DIV',
  [Opcode.LDR]: 'LDR',
  [Opcode.STR]: 'STR',
  [Opcode.CONST]: 'CONST',
  [Opcode.SLDR]: 'SLDR',
  [Opcode.SSTR]: 'SSTR',
  [Opcode.BAR]: 'BAR',
  [Opcode.RET]: 'RET',
};

/** Pipeline stages for each core's state machine */
export enum PipelineStage {
  FETCH   = 'FETCH',
  DECODE  = 'DECODE',
  REQUEST = 'REQUEST',
  WAIT    = 'WAIT',
  EXECUTE = 'EXECUTE',
  UPDATE  = 'UPDATE',
  BARRIER = 'BARRIER',
  DONE    = 'DONE',
}

/** Per-thread execution state */
export interface ThreadState {
  threadId: number;
  blockId: number;
  pc: number;
  registers: number[];      // R0-R15
  nzp: number;              // 3-bit NZP flags
  stage: PipelineStage;
  done: boolean;
  currentInstruction: string;
  divergent?: boolean;       // true if this thread diverged from majority
}

/** Full GPU simulation state at one cycle */
export interface SimulationState {
  cycle: number;
  threads: ThreadState[];
  memory: number[];          // 256-byte data memory
  sharedMemory: number[];    // 64-byte shared memory per block
  currentBlock: number;
  totalBlocks: number;
}
