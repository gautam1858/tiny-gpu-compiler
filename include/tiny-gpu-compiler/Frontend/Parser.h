#ifndef TGC_FRONTEND_PARSER_H
#define TGC_FRONTEND_PARSER_H

#include "tiny-gpu-compiler/Frontend/AST.h"
#include "tiny-gpu-compiler/Frontend/Lexer.h"

#include <memory>

namespace tgc {

class Parser {
public:
  explicit Parser(Lexer &lexer);

  std::unique_ptr<Program> parseProgram();

private:
  // Top-level
  std::unique_ptr<KernelDef> parseKernelDef();
  KernelParam parseParam();

  // Statements
  std::unique_ptr<Stmt> parseStatement();
  std::unique_ptr<VarDeclStmt> parseVarDecl();
  std::unique_ptr<Stmt> parseAssignmentOrExpr();
  std::unique_ptr<ForStmt> parseForStmt();
  std::unique_ptr<IfStmt> parseIfStmt();
  std::vector<std::unique_ptr<Stmt>> parseBlock();

  // Expressions (precedence climbing)
  std::unique_ptr<Expr> parseExpr();
  std::unique_ptr<Expr> parseComparison();
  std::unique_ptr<Expr> parseAdditive();
  std::unique_ptr<Expr> parseMultiplicative();
  std::unique_ptr<Expr> parsePrimary();

  // Utilities
  Token consume(TokenKind expected, const char *msg);
  Token advance();
  Token peek();
  bool check(TokenKind kind);
  bool match(TokenKind kind);
  Location currentLoc();
  [[noreturn]] void error(const char *msg);

  Lexer &lexer;
};

} // namespace tgc

#endif // TGC_FRONTEND_PARSER_H
