#ifndef TGC_FRONTEND_AST_H
#define TGC_FRONTEND_AST_H

#include <memory>
#include <string>
#include <vector>

namespace tgc {

struct Location {
  int line;
  int col;
};

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

enum class ExprKind {
  IntLiteral,
  Identifier,
  BuiltinVar,
  BinaryOp,
  ArrayIndex,
};

struct Expr {
  ExprKind kind;
  Location loc;
  virtual ~Expr() = default;
};

struct IntLiteralExpr : Expr {
  int value;
  IntLiteralExpr(int val, Location loc) : value(val) {
    kind = ExprKind::IntLiteral;
    this->loc = loc;
  }
};

struct IdentifierExpr : Expr {
  std::string name;
  IdentifierExpr(std::string name, Location loc) : name(std::move(name)) {
    kind = ExprKind::Identifier;
    this->loc = loc;
  }
};

enum class BuiltinVar { ThreadIdx, BlockIdx, BlockDim };

struct BuiltinVarExpr : Expr {
  BuiltinVar var;
  BuiltinVarExpr(BuiltinVar var, Location loc) : var(var) {
    kind = ExprKind::BuiltinVar;
    this->loc = loc;
  }
};

enum class BinOp { Add, Sub, Mul, Div, Eq, Neq, Lt, Gt, Leq, Geq };

struct BinaryOpExpr : Expr {
  BinOp op;
  std::unique_ptr<Expr> lhs, rhs;
  BinaryOpExpr(BinOp op, std::unique_ptr<Expr> lhs, std::unique_ptr<Expr> rhs,
               Location loc)
      : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {
    kind = ExprKind::BinaryOp;
    this->loc = loc;
  }
};

struct ArrayIndexExpr : Expr {
  std::string array;
  std::unique_ptr<Expr> index;
  ArrayIndexExpr(std::string array, std::unique_ptr<Expr> index, Location loc)
      : array(std::move(array)), index(std::move(index)) {
    kind = ExprKind::ArrayIndex;
    this->loc = loc;
  }
};

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

enum class StmtKind {
  VarDecl,
  Assignment,
  ArrayStore,
  For,
  If,
  ExprStmt,
};

struct Stmt {
  StmtKind kind;
  Location loc;
  virtual ~Stmt() = default;
};

struct VarDeclStmt : Stmt {
  std::string name;
  std::unique_ptr<Expr> init;
  VarDeclStmt(std::string name, std::unique_ptr<Expr> init, Location loc)
      : name(std::move(name)), init(std::move(init)) {
    kind = StmtKind::VarDecl;
    this->loc = loc;
  }
};

struct AssignmentStmt : Stmt {
  std::string name;
  std::unique_ptr<Expr> value;
  AssignmentStmt(std::string name, std::unique_ptr<Expr> value, Location loc)
      : name(std::move(name)), value(std::move(value)) {
    kind = StmtKind::Assignment;
    this->loc = loc;
  }
};

struct ArrayStoreStmt : Stmt {
  std::string array;
  std::unique_ptr<Expr> index;
  std::unique_ptr<Expr> value;
  ArrayStoreStmt(std::string array, std::unique_ptr<Expr> index,
                 std::unique_ptr<Expr> value, Location loc)
      : array(std::move(array)), index(std::move(index)),
        value(std::move(value)) {
    kind = StmtKind::ArrayStore;
    this->loc = loc;
  }
};

struct ForStmt : Stmt {
  std::unique_ptr<VarDeclStmt> init;
  std::unique_ptr<Expr> condition;
  std::string iterVar;
  std::unique_ptr<Expr> iterUpdate;
  std::vector<std::unique_ptr<Stmt>> body;
  ForStmt(Location loc) {
    kind = StmtKind::For;
    this->loc = loc;
  }
};

struct IfStmt : Stmt {
  std::unique_ptr<Expr> condition;
  std::vector<std::unique_ptr<Stmt>> thenBody;
  std::vector<std::unique_ptr<Stmt>> elseBody;
  IfStmt(Location loc) {
    kind = StmtKind::If;
    this->loc = loc;
  }
};

//===----------------------------------------------------------------------===//
// Kernel parameters and top-level
//===----------------------------------------------------------------------===//

struct KernelParam {
  std::string name;
  bool isGlobalPtr; // true = "global int*", false = "int"
  Location loc;
};

struct KernelDef {
  std::string name;
  std::vector<KernelParam> params;
  std::vector<std::unique_ptr<Stmt>> body;
  Location loc;
};

struct Program {
  std::vector<std::unique_ptr<KernelDef>> kernels;
};

} // namespace tgc

#endif // TGC_FRONTEND_AST_H
