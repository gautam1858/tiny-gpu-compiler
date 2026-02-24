#include "tiny-gpu-compiler/CodeGen/TinyGPUEmitter.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Format.h"

using namespace mlir;

namespace tgc {

/// Opcode encoding for tiny-gpu's 16-bit ISA
namespace Opcode {
constexpr uint16_t NOP = 0b0000;
constexpr uint16_t BRnzp = 0b0001;
constexpr uint16_t CMP = 0b0010;
constexpr uint16_t ADD = 0b0011;
constexpr uint16_t SUB = 0b0100;
constexpr uint16_t MUL = 0b0101;
constexpr uint16_t DIV = 0b0110;
constexpr uint16_t LDR = 0b0111;
constexpr uint16_t STR = 0b1000;
constexpr uint16_t CONST = 0b1001;
constexpr uint16_t RET = 0b1111;
} // namespace Opcode

/// Helper to get register number from an attribute
static int getReg(Operation *op, StringRef attrName) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return 0;
}

/// Encode: opcode[15:12] | rd[11:8] | rs[7:4] | rt[3:0]
static uint16_t encodeRRR(uint16_t opcode, int rd, int rs, int rt) {
  return (opcode << 12) | ((rd & 0xF) << 8) | ((rs & 0xF) << 4) | (rt & 0xF);
}

/// Encode: opcode[15:12] | rd[11:8] | imm[7:0]
static uint16_t encodeRI(uint16_t opcode, int rd, int imm) {
  return (opcode << 12) | ((rd & 0xF) << 8) | (imm & 0xFF);
}

/// Encode: opcode[15:12] | nzp[11:9] | 0 | target[7:0]
static uint16_t encodeBranch(uint16_t opcode, int nzp, int target) {
  return (opcode << 12) | ((nzp & 0x7) << 9) | (target & 0xFF);
}

/// Convert register number to assembly name
static std::string regName(int reg) {
  if (reg == 13) return "%blockIdx";
  if (reg == 14) return "%blockDim";
  if (reg == 15) return "%threadIdx";
  return "R" + std::to_string(reg);
}

std::vector<Instruction> emitBinary(Operation *funcOp) {
  std::vector<Instruction> instructions;
  llvm::DenseMap<Block *, int> blockAddresses;

  // First pass: compute block addresses
  int addr = 0;
  for (Block &block : funcOp->getRegion(0)) {
    blockAddresses[&block] = addr;
    for (Operation &op : block) {
      // Count instructions this op will generate
      if (isa<tinygpu::AddOp, tinygpu::SubOp, tinygpu::MulOp, tinygpu::DivOp,
              tinygpu::LoadOp, tinygpu::StoreOp, tinygpu::ConstOp,
              tinygpu::CmpOp, tinygpu::ReturnOp, tinygpu::BranchOp,
              tinygpu::JumpOp>(&op)) {
        addr++;
      }
      // Special register reads (thread_id, block_id, block_dim) don't emit
      // instructions — they reference hardware registers directly
    }
  }

  // Second pass: emit instructions
  addr = 0;
  for (Block &block : funcOp->getRegion(0)) {
    for (Operation &op : block) {
      Instruction inst;
      inst.address = addr;

      if (auto addOp = dyn_cast<tinygpu::AddOp>(&op)) {
        int rd = getReg(&op, "rd"), rs = getReg(&op, "rs"),
            rt = getReg(&op, "rt");
        inst.binary = encodeRRR(Opcode::ADD, rd, rs, rt);
        inst.assembly =
            "ADD " + regName(rd) + ", " + regName(rs) + ", " + regName(rt);
        instructions.push_back(inst);
        addr++;
      } else if (auto subOp = dyn_cast<tinygpu::SubOp>(&op)) {
        int rd = getReg(&op, "rd"), rs = getReg(&op, "rs"),
            rt = getReg(&op, "rt");
        inst.binary = encodeRRR(Opcode::SUB, rd, rs, rt);
        inst.assembly =
            "SUB " + regName(rd) + ", " + regName(rs) + ", " + regName(rt);
        instructions.push_back(inst);
        addr++;
      } else if (auto mulOp = dyn_cast<tinygpu::MulOp>(&op)) {
        int rd = getReg(&op, "rd"), rs = getReg(&op, "rs"),
            rt = getReg(&op, "rt");
        inst.binary = encodeRRR(Opcode::MUL, rd, rs, rt);
        inst.assembly =
            "MUL " + regName(rd) + ", " + regName(rs) + ", " + regName(rt);
        instructions.push_back(inst);
        addr++;
      } else if (auto divOp = dyn_cast<tinygpu::DivOp>(&op)) {
        int rd = getReg(&op, "rd"), rs = getReg(&op, "rs"),
            rt = getReg(&op, "rt");
        inst.binary = encodeRRR(Opcode::DIV, rd, rs, rt);
        inst.assembly =
            "DIV " + regName(rd) + ", " + regName(rs) + ", " + regName(rt);
        instructions.push_back(inst);
        addr++;
      } else if (auto loadOp = dyn_cast<tinygpu::LoadOp>(&op)) {
        int rd = getReg(&op, "rd"), rs = getReg(&op, "rs");
        inst.binary = encodeRRR(Opcode::LDR, rd, rs, 0);
        inst.assembly = "LDR " + regName(rd) + ", [" + regName(rs) + "]";
        instructions.push_back(inst);
        addr++;
      } else if (auto storeOp = dyn_cast<tinygpu::StoreOp>(&op)) {
        int rs = getReg(&op, "rs"), rt = getReg(&op, "rt");
        inst.binary = encodeRRR(Opcode::STR, 0, rs, rt);
        inst.assembly = "STR [" + regName(rs) + "], " + regName(rt);
        instructions.push_back(inst);
        addr++;
      } else if (auto constOp = dyn_cast<tinygpu::ConstOp>(&op)) {
        int rd = getReg(&op, "rd");
        int imm = constOp.getValue().getZExtValue();
        inst.binary = encodeRI(Opcode::CONST, rd, imm);
        inst.assembly =
            "CONST " + regName(rd) + ", #" + std::to_string(imm);
        instructions.push_back(inst);
        addr++;
      } else if (auto cmpOp = dyn_cast<tinygpu::CmpOp>(&op)) {
        int rs = getReg(&op, "rs"), rt = getReg(&op, "rt");
        inst.binary = encodeRRR(Opcode::CMP, 0, rs, rt);
        inst.assembly = "CMP " + regName(rs) + ", " + regName(rt);
        instructions.push_back(inst);
        addr++;
      } else if (auto branchOp = dyn_cast<tinygpu::BranchOp>(&op)) {
        int nzp = branchOp.getConditionMask().getZExtValue();
        Block *target = branchOp.getTarget();
        int targetAddr = blockAddresses.lookup(target);
        inst.binary = encodeBranch(Opcode::BRnzp, nzp, targetAddr);
        inst.assembly =
            "BRnzp " + std::to_string(nzp) + ", #" + std::to_string(targetAddr);
        instructions.push_back(inst);
        addr++;
      } else if (auto jumpOp = dyn_cast<tinygpu::JumpOp>(&op)) {
        // Unconditional jump = BRnzp with mask 0b111 (always taken)
        Block *target = jumpOp.getTarget();
        int targetAddr = blockAddresses.lookup(target);
        inst.binary = encodeBranch(Opcode::BRnzp, 0b111, targetAddr);
        inst.assembly = "JMP #" + std::to_string(targetAddr);
        instructions.push_back(inst);
        addr++;
      } else if (isa<tinygpu::ReturnOp>(&op)) {
        inst.binary = encodeRRR(Opcode::RET, 0, 0, 0);
        inst.assembly = "RET";
        instructions.push_back(inst);
        addr++;
      }
      // Skip thread_id/block_id/block_dim — they just reference hardware regs
    }
  }

  return instructions;
}

void emitHex(const std::vector<Instruction> &instructions,
             llvm::raw_ostream &os) {
  for (auto &inst : instructions) {
    os << llvm::format("0x%04X", inst.binary) << "\n";
  }
}

void emitRawBinary(const std::vector<Instruction> &instructions,
                   llvm::raw_ostream &os) {
  for (auto &inst : instructions) {
    // Big-endian: high byte first (matching tiny-gpu's memory layout)
    uint8_t hi = (inst.binary >> 8) & 0xFF;
    uint8_t lo = inst.binary & 0xFF;
    os.write((char)hi);
    os.write((char)lo);
  }
}

void emitAssembly(const std::vector<Instruction> &instructions,
                  llvm::raw_ostream &os) {
  for (auto &inst : instructions) {
    os << llvm::format("%3d: 0x%04X  %-40s  ; ", inst.address, inst.binary,
                       inst.assembly.c_str());
    // Print binary representation
    for (int bit = 15; bit >= 0; bit--) {
      os << ((inst.binary >> bit) & 1);
      if (bit == 12 || bit == 8 || bit == 4)
        os << " ";
    }
    os << "\n";
  }
}

void emitJsonTrace(const CompilationTrace &trace, llvm::raw_ostream &os) {
  os << "{\n";
  os << "  \"source\": \"";
  // Escape source code for JSON
  for (char c : trace.sourceCode) {
    if (c == '"')
      os << "\\\"";
    else if (c == '\\')
      os << "\\\\";
    else if (c == '\n')
      os << "\\n";
    else if (c == '\t')
      os << "\\t";
    else
      os << c;
  }
  os << "\",\n";

  // IR stages
  os << "  \"stages\": [\n";
  for (size_t i = 0; i < trace.irStages.size(); i++) {
    os << "    {\"name\": \"" << trace.irStages[i].first << "\", \"ir\": \"";
    for (char c : trace.irStages[i].second) {
      if (c == '"')
        os << "\\\"";
      else if (c == '\\')
        os << "\\\\";
      else if (c == '\n')
        os << "\\n";
      else if (c == '\t')
        os << "\\t";
      else
        os << c;
    }
    os << "\"}";
    if (i + 1 < trace.irStages.size())
      os << ",";
    os << "\n";
  }
  os << "  ],\n";

  // Binary instructions
  os << "  \"binary\": {\n";
  os << "    \"instructions\": [\n";
  for (size_t i = 0; i < trace.instructions.size(); i++) {
    auto &inst = trace.instructions[i];
    os << "      {\"addr\": " << inst.address;
    os << ", \"hex\": \"" << llvm::format("0x%04X", inst.binary) << "\"";
    os << ", \"asm\": \"" << inst.assembly << "\"";
    os << ", \"bits\": \"";
    for (int bit = 15; bit >= 0; bit--) {
      os << ((inst.binary >> bit) & 1);
      if (bit == 12 || bit == 8 || bit == 4)
        os << " ";
    }
    os << "\"}";
    if (i + 1 < trace.instructions.size())
      os << ",";
    os << "\n";
  }
  os << "    ]\n";
  os << "  }\n";
  os << "}\n";
}

} // namespace tgc
