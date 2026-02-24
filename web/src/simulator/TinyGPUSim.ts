import {
  Instruction,
  Opcode,
  OPCODE_NAMES,
  PipelineStage,
  ThreadState,
  SimulationState,
} from '../compiler/types';

/**
 * JavaScript implementation of the tiny-gpu hardware simulator.
 * Faithfully mirrors the Verilog: dispatcher → cores → threads → ALU/LSU.
 */
export class TinyGPUSim {
  private program: number[] = [];     // Program memory (16-bit instructions)
  private memory: number[] = [];      // Data memory (256 bytes)
  private threads: ThreadState[] = [];
  private numBlocks: number;
  private threadsPerBlock: number;
  private currentBlock = 0;
  private cycle = 0;
  private blockDone = false;

  constructor(
    instructions: Instruction[],
    initialMemory: number[],
    numBlocks: number,
    threadsPerBlock: number
  ) {
    this.numBlocks = numBlocks;
    this.threadsPerBlock = threadsPerBlock;

    // Load program memory
    this.program = instructions.map((i) => parseInt(i.hex, 16));

    // Initialize data memory
    this.memory = new Array(256).fill(0);
    for (let i = 0; i < initialMemory.length && i < 256; i++) {
      this.memory[i] = initialMemory[i] & 0xff;
    }

    // Initialize threads for first block
    this.initBlock(0);
  }

  private initBlock(blockId: number) {
    this.currentBlock = blockId;
    this.blockDone = false;
    this.threads = [];

    for (let t = 0; t < this.threadsPerBlock; t++) {
      const regs = new Array(16).fill(0);
      regs[13] = blockId;                 // %blockIdx
      regs[14] = this.threadsPerBlock;    // %blockDim
      regs[15] = t;                       // %threadIdx
      this.threads.push({
        threadId: t,
        blockId,
        pc: 0,
        registers: regs,
        nzp: 0b010, // Zero flag set initially
        stage: PipelineStage.FETCH,
        done: false,
        currentInstruction: '',
      });
    }
  }

  /** Get current state snapshot (for visualization) */
  getState(): SimulationState {
    return {
      cycle: this.cycle,
      threads: this.threads.map((t) => ({ ...t, registers: [...t.registers] })),
      memory: [...this.memory],
      currentBlock: this.currentBlock,
      totalBlocks: this.numBlocks,
    };
  }

  /** Check if entire simulation is complete */
  isDone(): boolean {
    return this.currentBlock >= this.numBlocks;
  }

  /** Advance simulation by one cycle (all threads in lockstep) */
  step(): SimulationState {
    if (this.isDone()) return this.getState();

    // Check if current block is done
    if (this.threads.every((t) => t.done)) {
      this.currentBlock++;
      if (this.currentBlock < this.numBlocks) {
        this.initBlock(this.currentBlock);
      }
      this.cycle++;
      return this.getState();
    }

    // Execute one pipeline cycle for all active threads
    for (const thread of this.threads) {
      if (thread.done) continue;
      this.executeThread(thread);
    }

    this.cycle++;
    return this.getState();
  }

  /** Execute one pipeline stage for a single thread */
  private executeThread(thread: ThreadState) {
    switch (thread.stage) {
      case PipelineStage.FETCH:
        this.fetch(thread);
        break;
      case PipelineStage.DECODE:
        this.decode(thread);
        break;
      case PipelineStage.REQUEST:
        this.request(thread);
        break;
      case PipelineStage.WAIT:
        this.waitStage(thread);
        break;
      case PipelineStage.EXECUTE:
        this.execute(thread);
        break;
      case PipelineStage.UPDATE:
        this.update(thread);
        break;
    }
  }

  // Decoded instruction state per thread (stored temporarily)
  private decoded = new Map<
    number,
    {
      opcode: number;
      rd: number;
      rs: number;
      rt: number;
      imm: number;
      nzpMask: number;
      result?: number;
      memAddr?: number;
      memData?: number;
      memRead?: number;
    }
  >();

  private fetch(thread: ThreadState) {
    if (thread.pc >= this.program.length) {
      thread.done = true;
      thread.stage = PipelineStage.DONE;
      return;
    }
    // Fetch instruction from program memory
    const instruction = this.program[thread.pc];
    const opcode = (instruction >> 12) & 0xf;
    thread.currentInstruction =
      OPCODE_NAMES[opcode] || `UNK(${opcode.toString(2).padStart(4, '0')})`;
    thread.stage = PipelineStage.DECODE;
  }

  private decode(thread: ThreadState) {
    const instruction = this.program[thread.pc];
    const opcode = (instruction >> 12) & 0xf;
    const rd = (instruction >> 8) & 0xf;
    const rs = (instruction >> 4) & 0xf;
    const rt = instruction & 0xf;
    const imm = instruction & 0xff;
    const nzpMask = (instruction >> 9) & 0x7;

    this.decoded.set(thread.threadId + thread.blockId * 1000, {
      opcode,
      rd,
      rs,
      rt,
      imm,
      nzpMask,
    });

    thread.stage = PipelineStage.REQUEST;
  }

  private request(thread: ThreadState) {
    const key = thread.threadId + thread.blockId * 1000;
    const d = this.decoded.get(key)!;

    // If LDR, prepare memory read address
    if (d.opcode === Opcode.LDR) {
      d.memAddr = thread.registers[d.rs] & 0xff;
    }
    // If STR, prepare memory write
    if (d.opcode === Opcode.STR) {
      d.memAddr = thread.registers[d.rs] & 0xff;
      d.memData = thread.registers[d.rt] & 0xff;
    }

    thread.stage = PipelineStage.WAIT;
  }

  private waitStage(thread: ThreadState) {
    const key = thread.threadId + thread.blockId * 1000;
    const d = this.decoded.get(key)!;

    // Perform memory operations
    if (d.opcode === Opcode.LDR && d.memAddr !== undefined) {
      d.memRead = this.memory[d.memAddr] & 0xff;
    }
    if (d.opcode === Opcode.STR && d.memAddr !== undefined && d.memData !== undefined) {
      this.memory[d.memAddr] = d.memData;
    }

    thread.stage = PipelineStage.EXECUTE;
  }

  private execute(thread: ThreadState) {
    const key = thread.threadId + thread.blockId * 1000;
    const d = this.decoded.get(key)!;
    const regs = thread.registers;

    switch (d.opcode) {
      case Opcode.NOP:
        break;
      case Opcode.ADD:
        d.result = (regs[d.rs] + regs[d.rt]) & 0xff;
        break;
      case Opcode.SUB:
        d.result = (regs[d.rs] - regs[d.rt]) & 0xff;
        break;
      case Opcode.MUL:
        d.result = (regs[d.rs] * regs[d.rt]) & 0xff;
        break;
      case Opcode.DIV:
        d.result = regs[d.rt] !== 0 ? Math.floor(regs[d.rs] / regs[d.rt]) & 0xff : 0;
        break;
      case Opcode.CMP: {
        const diff = regs[d.rs] - regs[d.rt];
        const n = diff < 0 ? 1 : 0;
        const z = diff === 0 ? 1 : 0;
        const p = diff > 0 ? 1 : 0;
        d.result = (n << 2) | (z << 1) | p;
        break;
      }
      case Opcode.CONST:
        d.result = d.imm & 0xff;
        break;
      case Opcode.LDR:
        d.result = d.memRead ?? 0;
        break;
      case Opcode.STR:
        // Already handled in WAIT
        break;
      case Opcode.BRnzp:
        // Branch logic
        break;
      case Opcode.RET:
        break;
    }

    thread.stage = PipelineStage.UPDATE;
  }

  private update(thread: ThreadState) {
    const key = thread.threadId + thread.blockId * 1000;
    const d = this.decoded.get(key)!;

    switch (d.opcode) {
      case Opcode.ADD:
      case Opcode.SUB:
      case Opcode.MUL:
      case Opcode.DIV:
      case Opcode.CONST:
      case Opcode.LDR:
        if (d.result !== undefined) {
          thread.registers[d.rd] = d.result & 0xff;
        }
        thread.pc++;
        break;
      case Opcode.CMP:
        if (d.result !== undefined) {
          thread.nzp = d.result & 0x7;
        }
        thread.pc++;
        break;
      case Opcode.STR:
      case Opcode.NOP:
        thread.pc++;
        break;
      case Opcode.BRnzp: {
        const taken = (thread.nzp & d.nzpMask) !== 0;
        if (taken) {
          thread.pc = d.imm;
        } else {
          thread.pc++;
        }
        break;
      }
      case Opcode.RET:
        thread.done = true;
        thread.stage = PipelineStage.DONE;
        this.decoded.delete(key);
        return;
    }

    this.decoded.delete(key);
    thread.stage = PipelineStage.FETCH;
  }

  /** Run entire simulation to completion */
  runToEnd(maxCycles = 10000): SimulationState[] {
    const history: SimulationState[] = [this.getState()];
    while (!this.isDone() && this.cycle < maxCycles) {
      history.push(this.step());
    }
    return history;
  }
}
