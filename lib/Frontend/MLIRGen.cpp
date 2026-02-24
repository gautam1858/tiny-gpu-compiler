#include "tiny-gpu-compiler/Frontend/MLIRGen.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace tgc {

/// MLIRGen implementation that walks the AST and emits MLIR.
/// Directly emits TinyGPU dialect ops (skipping the gpu dialect intermediate
/// for simplicity, since we're the only frontend and tiny-gpu is 1D only).
class MLIRGenImpl {
public:
  MLIRGenImpl(MLIRContext &context) : builder(&context), context(context) {}

  OwningOpRef<ModuleOp> generate(const Program &program) {
    module = ModuleOp::create(builder.getUnknownLoc());

    for (auto &kernel : program.kernels) {
      if (failed(genKernel(*kernel)))
        return nullptr;
    }

    if (failed(verify(*module))) {
      module->emitError("module verification failed");
      return nullptr;
    }

    return std::move(module);
  }

private:
  OpBuilder builder;
  MLIRContext &context;
  OwningOpRef<ModuleOp> module;

  // Symbol table for local variables
  llvm::ScopedHashTable<llvm::StringRef, Value> symbolTable;
  using ScopeTy = llvm::ScopedHashTableScope<llvm::StringRef, Value>;

  // Kernel parameter base addresses (for global int* params)
  llvm::StringMap<int> paramBaseAddrs;

  // Owned strings for symbol table keys
  std::vector<std::unique_ptr<std::string>> ownedNames;

  llvm::StringRef ownName(const std::string &name) {
    ownedNames.push_back(std::make_unique<std::string>(name));
    return *ownedNames.back();
  }

  Location loc(tgc::Location l) {
    return mlir::FileLineColLoc::get(builder.getStringAttr("<kernel>"), l.line,
                                     l.col);
  }

  Type i8Ty() { return builder.getIntegerType(8); }

  LogicalResult genKernel(const KernelDef &kernel) {
    ScopeTy scope(symbolTable);

    // Create tinygpu.func with no arguments (params are memory-mapped)
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<tinygpu::FuncOp>(loc(kernel.loc), kernel.name,
                                                   funcType);
    Block *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Map parameters to base addresses in data memory.
    // Convention: each global int* param gets a consecutive 64-byte region.
    // param[0] starts at address 0, param[1] at 64, param[2] at 128, etc.
    // Scalar int params are placed at fixed addresses after pointer params.
    paramBaseAddrs.clear();
    int ptrIdx = 0;
    int scalarAddr = 192; // Scalars start at address 192
    for (auto &param : kernel.params) {
      if (param.isGlobalPtr) {
        paramBaseAddrs[param.name] = ptrIdx * 64;
        ptrIdx++;
      } else {
        paramBaseAddrs[param.name] = scalarAddr;
        // Load the scalar value into a variable
        auto addrVal =
            builder.create<tinygpu::ConstOp>(loc(param.loc), (uint8_t)scalarAddr);
        auto val = builder.create<tinygpu::LoadOp>(loc(param.loc), addrVal);
        symbolTable.insert(ownName(param.name), val);
        scalarAddr++;
      }
    }

    // Generate body
    for (auto &stmt : kernel.body) {
      if (failed(genStmt(*stmt)))
        return failure();
    }

    // Add return if not already terminated
    if (entryBlock->empty() ||
        !entryBlock->back().hasTrait<OpTrait::IsTerminator>()) {
      builder.create<tinygpu::ReturnOp>(loc(kernel.loc));
    }

    module->push_back(funcOp);
    return success();
  }

  LogicalResult genStmt(const Stmt &stmt) {
    switch (stmt.kind) {
    case StmtKind::VarDecl:
      return genVarDecl(static_cast<const VarDeclStmt &>(stmt));
    case StmtKind::Assignment:
      return genAssignment(static_cast<const AssignmentStmt &>(stmt));
    case StmtKind::ArrayStore:
      return genArrayStore(static_cast<const ArrayStoreStmt &>(stmt));
    case StmtKind::For:
      return genFor(static_cast<const ForStmt &>(stmt));
    case StmtKind::If:
      return genIf(static_cast<const IfStmt &>(stmt));
    default:
      llvm::errs() << "unhandled statement kind\n";
      return failure();
    }
  }

  LogicalResult genVarDecl(const VarDeclStmt &stmt) {
    auto val = genExpr(*stmt.init);
    if (!val)
      return failure();
    symbolTable.insert(ownName(stmt.name), val);
    return success();
  }

  LogicalResult genAssignment(const AssignmentStmt &stmt) {
    auto val = genExpr(*stmt.value);
    if (!val)
      return failure();
    symbolTable.insert(ownName(stmt.name), val);
    return success();
  }

  LogicalResult genArrayStore(const ArrayStoreStmt &stmt) {
    auto index = genExpr(*stmt.index);
    auto value = genExpr(*stmt.value);
    if (!index || !value)
      return failure();

    // Compute address: base + index
    auto it = paramBaseAddrs.find(stmt.array);
    if (it == paramBaseAddrs.end()) {
      llvm::errs() << "unknown array: " << stmt.array << "\n";
      return failure();
    }

    Value addr;
    if (it->second == 0) {
      addr = index;
    } else {
      auto base =
          builder.create<tinygpu::ConstOp>(loc(stmt.loc), (uint8_t)it->second);
      addr = builder.create<tinygpu::AddOp>(loc(stmt.loc), base, index);
    }

    builder.create<tinygpu::StoreOp>(loc(stmt.loc), addr, value);
    return success();
  }

  LogicalResult genFor(const ForStmt &stmt) {
    // Generate init
    if (failed(genVarDecl(*stmt.init)))
      return failure();

    // Create loop blocks
    Block *condBlock = new Block();
    Block *bodyBlock = new Block();
    Block *exitBlock = new Block();

    auto parentOp = builder.getBlock()->getParent();
    parentOp->push_back(condBlock);
    parentOp->push_back(bodyBlock);
    parentOp->push_back(exitBlock);

    // Jump to condition check
    builder.create<tinygpu::JumpOp>(loc(stmt.loc), condBlock);

    // Condition block: evaluate condition, branch
    builder.setInsertionPointToStart(condBlock);
    auto condVal = genExpr(*stmt.condition);
    if (!condVal)
      return failure();

    // The condition expression should produce an NZP-like value.
    // For comparisons, the cmp result is already NZP.
    // Branch to body if condition is true (positive), exit otherwise.
    // We use condition_mask = 0b101 (N or P, i.e., not zero) for "not equal"
    // or 0b001 (P only) for "less than result is positive meaning lhs > rhs"
    // For simplicity: branch to body if condVal indicates the comparison is true.
    auto zeroConst = builder.create<tinygpu::ConstOp>(loc(stmt.loc), (uint8_t)0);
    auto nzp = builder.create<tinygpu::CmpOp>(loc(stmt.loc), condVal, zeroConst);
    // Branch if positive (condition was nonzero = true)
    builder.create<tinygpu::BranchOp>(loc(stmt.loc), nzp, (uint8_t)0b001,
                                       bodyBlock);

    // Body block
    builder.setInsertionPointToStart(bodyBlock);
    for (auto &s : stmt.body) {
      if (failed(genStmt(*s)))
        return failure();
    }

    // Update iterator
    auto updateVal = genExpr(*stmt.iterUpdate);
    if (!updateVal)
      return failure();
    symbolTable.insert(ownName(stmt.iterVar), updateVal);

    // Jump back to condition
    builder.create<tinygpu::JumpOp>(loc(stmt.loc), condBlock);

    // Continue in exit block
    builder.setInsertionPointToStart(exitBlock);
    return success();
  }

  LogicalResult genIf(const IfStmt &stmt) {
    auto condVal = genExpr(*stmt.condition);
    if (!condVal)
      return failure();

    Block *thenBlock = new Block();
    Block *mergeBlock = new Block();

    auto parentOp = builder.getBlock()->getParent();
    parentOp->push_back(thenBlock);

    Block *elseBlock = nullptr;
    if (!stmt.elseBody.empty()) {
      elseBlock = new Block();
      parentOp->push_back(elseBlock);
    }
    parentOp->push_back(mergeBlock);

    // Branch on condition
    auto zeroConst = builder.create<tinygpu::ConstOp>(loc(stmt.loc), (uint8_t)0);
    auto nzp = builder.create<tinygpu::CmpOp>(loc(stmt.loc), condVal, zeroConst);
    builder.create<tinygpu::BranchOp>(loc(stmt.loc), nzp, (uint8_t)0b001,
                                       thenBlock);

    // Then block
    builder.setInsertionPointToStart(thenBlock);
    for (auto &s : stmt.thenBody) {
      if (failed(genStmt(*s)))
        return failure();
    }
    builder.create<tinygpu::JumpOp>(loc(stmt.loc), mergeBlock);

    // Else block (if present)
    if (elseBlock) {
      builder.setInsertionPointToStart(elseBlock);
      for (auto &s : stmt.elseBody) {
        if (failed(genStmt(*s)))
          return failure();
      }
      builder.create<tinygpu::JumpOp>(loc(stmt.loc), mergeBlock);
    }

    // Continue in merge block
    builder.setInsertionPointToStart(mergeBlock);
    return success();
  }

  Value genExpr(const Expr &expr) {
    switch (expr.kind) {
    case ExprKind::IntLiteral:
      return genIntLiteral(static_cast<const IntLiteralExpr &>(expr));
    case ExprKind::Identifier:
      return genIdentifier(static_cast<const IdentifierExpr &>(expr));
    case ExprKind::BuiltinVar:
      return genBuiltinVar(static_cast<const BuiltinVarExpr &>(expr));
    case ExprKind::BinaryOp:
      return genBinaryOp(static_cast<const BinaryOpExpr &>(expr));
    case ExprKind::ArrayIndex:
      return genArrayIndex(static_cast<const ArrayIndexExpr &>(expr));
    }
    llvm_unreachable("unhandled expr kind");
  }

  Value genIntLiteral(const IntLiteralExpr &expr) {
    if (expr.value < 0 || expr.value > 255) {
      llvm::errs() << "integer literal " << expr.value
                   << " out of 8-bit range [0, 255]\n";
      return nullptr;
    }
    return builder.create<tinygpu::ConstOp>(loc(expr.loc),
                                            (uint8_t)expr.value);
  }

  Value genIdentifier(const IdentifierExpr &expr) {
    Value val = symbolTable.lookup(expr.name);
    if (!val) {
      llvm::errs() << "undefined variable: " << expr.name << "\n";
      return nullptr;
    }
    return val;
  }

  Value genBuiltinVar(const BuiltinVarExpr &expr) {
    switch (expr.var) {
    case BuiltinVar::ThreadIdx:
      return builder.create<tinygpu::ThreadIdOp>(loc(expr.loc));
    case BuiltinVar::BlockIdx:
      return builder.create<tinygpu::BlockIdOp>(loc(expr.loc));
    case BuiltinVar::BlockDim:
      return builder.create<tinygpu::BlockDimOp>(loc(expr.loc));
    }
    llvm_unreachable("unhandled builtin var");
  }

  Value genBinaryOp(const BinaryOpExpr &expr) {
    auto lhs = genExpr(*expr.lhs);
    auto rhs = genExpr(*expr.rhs);
    if (!lhs || !rhs)
      return nullptr;

    switch (expr.op) {
    case BinOp::Add:
      return builder.create<tinygpu::AddOp>(loc(expr.loc), lhs, rhs);
    case BinOp::Sub:
      return builder.create<tinygpu::SubOp>(loc(expr.loc), lhs, rhs);
    case BinOp::Mul:
      return builder.create<tinygpu::MulOp>(loc(expr.loc), lhs, rhs);
    case BinOp::Div:
      return builder.create<tinygpu::DivOp>(loc(expr.loc), lhs, rhs);
    case BinOp::Lt:
    case BinOp::Gt:
    case BinOp::Leq:
    case BinOp::Geq:
    case BinOp::Eq:
    case BinOp::Neq:
      // All comparisons use CMP (lhs - rhs), caller interprets NZP flags
      return builder.create<tinygpu::CmpOp>(loc(expr.loc), lhs, rhs);
    }
    llvm_unreachable("unhandled binary op");
  }

  Value genArrayIndex(const ArrayIndexExpr &expr) {
    auto index = genExpr(*expr.index);
    if (!index)
      return nullptr;

    auto it = paramBaseAddrs.find(expr.array);
    if (it == paramBaseAddrs.end()) {
      llvm::errs() << "unknown array: " << expr.array << "\n";
      return nullptr;
    }

    Value addr;
    if (it->second == 0) {
      addr = index;
    } else {
      auto base =
          builder.create<tinygpu::ConstOp>(loc(expr.loc), (uint8_t)it->second);
      addr = builder.create<tinygpu::AddOp>(loc(expr.loc), base, index);
    }

    return builder.create<tinygpu::LoadOp>(loc(expr.loc), addr);
  }
};

OwningOpRef<ModuleOp> mlirGen(MLIRContext &context, const Program &program) {
  MLIRGenImpl gen(context);
  return gen.generate(program);
}

} // namespace tgc
