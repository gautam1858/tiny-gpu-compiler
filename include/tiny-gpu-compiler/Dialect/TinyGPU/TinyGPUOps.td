#ifndef TINYGPU_OPS_TD
#define TINYGPU_OPS_TD

include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for all TinyGPU ops
class TinyGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TinyGPU_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Kernel function
//===----------------------------------------------------------------------===//

def TinyGPU_FuncOp : TinyGPU_Op<"func", [
    IsolatedFromAbove,
    FunctionOpInterface,
    Symbol
]> {
  let summary = "GPU kernel function targeting tiny-gpu hardware";
  let description = [{
    Represents a GPU kernel that will be compiled to tiny-gpu binary.
    Parameters are mapped to base addresses in data memory.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    /// Returns the region on the function operation.
    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Special register reads (read-only hardware registers)
//===----------------------------------------------------------------------===//

def TinyGPU_ThreadIdOp : TinyGPU_Op<"thread_id", [Pure]> {
  let summary = "Read thread index within block (R15)";
  let results = (outs I8:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TinyGPU_BlockIdOp : TinyGPU_Op<"block_id", [Pure]> {
  let summary = "Read block index (R13)";
  let results = (outs I8:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TinyGPU_BlockDimOp : TinyGPU_Op<"block_dim", [Pure]> {
  let summary = "Read threads per block (R14)";
  let results = (outs I8:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Arithmetic operations (map to ALU)
//===----------------------------------------------------------------------===//

def TinyGPU_AddOp : TinyGPU_Op<"add", [Pure, Commutative]> {
  let summary = "8-bit unsigned addition";
  let arguments = (ins I8:$lhs, I8:$rhs);
  let results = (outs I8:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def TinyGPU_SubOp : TinyGPU_Op<"sub", [Pure]> {
  let summary = "8-bit unsigned subtraction";
  let arguments = (ins I8:$lhs, I8:$rhs);
  let results = (outs I8:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def TinyGPU_MulOp : TinyGPU_Op<"mul", [Pure, Commutative]> {
  let summary = "8-bit unsigned multiplication";
  let arguments = (ins I8:$lhs, I8:$rhs);
  let results = (outs I8:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def TinyGPU_DivOp : TinyGPU_Op<"div", [Pure]> {
  let summary = "8-bit unsigned division";
  let arguments = (ins I8:$lhs, I8:$rhs);
  let results = (outs I8:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Memory operations
//===----------------------------------------------------------------------===//

def TinyGPU_LoadOp : TinyGPU_Op<"load"> {
  let summary = "Load 8-bit value from data memory";
  let description = [{
    Loads a value from the tiny-gpu's 256-byte data memory at the given address.
    Maps to the LDR instruction.
  }];
  let arguments = (ins I8:$addr);
  let results = (outs I8:$result);
  let assemblyFormat = "$addr attr-dict `:` type($result)";
}

def TinyGPU_StoreOp : TinyGPU_Op<"store"> {
  let summary = "Store 8-bit value to data memory";
  let description = [{
    Stores a value to the tiny-gpu's 256-byte data memory at the given address.
    Maps to the STR instruction.
  }];
  let arguments = (ins I8:$addr, I8:$value);
  let assemblyFormat = "$addr `,` $value attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

def TinyGPU_ConstOp : TinyGPU_Op<"const", [Pure, ConstantLike]> {
  let summary = "Load 8-bit immediate constant";
  let arguments = (ins I8Attr:$value);
  let results = (outs I8:$result);
  let assemblyFormat = "$value attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Comparison and branching
//===----------------------------------------------------------------------===//

def TinyGPU_CmpOp : TinyGPU_Op<"cmp", [Pure]> {
  let summary = "Compare two values, set NZP flags";
  let description = [{
    Compares lhs and rhs by computing lhs - rhs, then sets NZP flags:
    - N (bit 2): result is negative
    - Z (bit 1): result is zero
    - P (bit 0): result is positive
    Maps to the CMP instruction.
  }];
  let arguments = (ins I8:$lhs, I8:$rhs);
  let results = (outs I8:$nzp);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($nzp)";
}

def TinyGPU_BranchOp : TinyGPU_Op<"branch", [Terminator]> {
  let summary = "Conditional branch based on NZP flags";
  let description = [{
    Branches to `target` block if (nzp_reg & condition_mask) != 0.
    The condition_mask is a 3-bit value selecting which NZP flags to test.
    Maps to the BRnzp instruction.
  }];
  let arguments = (ins I8:$nzp, I8Attr:$condition_mask, AnySuccessor:$target);
  let assemblyFormat = "$nzp `,` $condition_mask `,` $target attr-dict";
}

//===----------------------------------------------------------------------===//
// Terminator
//===----------------------------------------------------------------------===//

def TinyGPU_ReturnOp : TinyGPU_Op<"ret", [Terminator]> {
  let summary = "Signal thread completion";
  let description = [{
    Indicates the current thread has finished execution.
    Maps to the RET instruction.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Unconditional jump (for control flow lowering)
//===----------------------------------------------------------------------===//

def TinyGPU_JumpOp : TinyGPU_Op<"jump", [Terminator]> {
  let summary = "Unconditional jump to target block";
  let description = [{
    Unconditional branch. Lowered to BRnzp with condition_mask=0b111 (always taken).
  }];
  let arguments = (ins AnySuccessor:$target);
  let assemblyFormat = "$target attr-dict";
}

#endif // TINYGPU_OPS_TD
