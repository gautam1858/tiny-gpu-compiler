#include "tiny-gpu-compiler/Frontend/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include <stdexcept>

namespace tgc {

Parser::Parser(Lexer &lexer) : lexer(lexer) {}

Token Parser::advance() { return lexer.nextToken(); }

Token Parser::peek() { return lexer.peekToken(); }

bool Parser::check(TokenKind kind) { return peek().kind == kind; }

bool Parser::match(TokenKind kind) {
  if (check(kind)) {
    advance();
    return true;
  }
  return false;
}

Token Parser::consume(TokenKind expected, const char *msg) {
  Token tok = advance();
  if (tok.kind != expected) {
    llvm::errs() << "Parse error at line " << tok.line << ":" << tok.col
                 << ": " << msg << " (got '" << tok.text << "')\n";
    std::exit(1);
  }
  return tok;
}

Location Parser::currentLoc() {
  Token tok = peek();
  return {tok.line, tok.col};
}

void Parser::error(const char *msg) {
  Token tok = peek();
  llvm::errs() << "Parse error at line " << tok.line << ":" << tok.col << ": "
               << msg << " (got '" << tok.text << "')\n";
  std::exit(1);
}

//===----------------------------------------------------------------------===//
// Top-level
//===----------------------------------------------------------------------===//

std::unique_ptr<Program> Parser::parseProgram() {
  auto program = std::make_unique<Program>();
  while (!check(TokenKind::Eof)) {
    program->kernels.push_back(parseKernelDef());
  }
  return program;
}

std::unique_ptr<KernelDef> Parser::parseKernelDef() {
  auto kernel = std::make_unique<KernelDef>();
  kernel->loc = currentLoc();

  consume(TokenKind::Kernel, "expected 'kernel'");
  Token name = consume(TokenKind::Identifier, "expected kernel name");
  kernel->name = name.text;

  consume(TokenKind::LParen, "expected '('");

  // Parse parameters
  if (!check(TokenKind::RParen)) {
    kernel->params.push_back(parseParam());
    while (match(TokenKind::Comma)) {
      kernel->params.push_back(parseParam());
    }
  }

  consume(TokenKind::RParen, "expected ')'");

  // Parse body
  kernel->body = parseBlock();

  return kernel;
}

KernelParam Parser::parseParam() {
  KernelParam param;
  param.loc = currentLoc();

  if (check(TokenKind::Global)) {
    advance(); // consume 'global'
    consume(TokenKind::Int, "expected 'int' after 'global'");
    consume(TokenKind::Star, "expected '*' after 'int'");
    Token name = consume(TokenKind::Identifier, "expected parameter name");
    param.name = name.text;
    param.isGlobalPtr = true;
  } else {
    consume(TokenKind::Int, "expected 'int' or 'global'");
    Token name = consume(TokenKind::Identifier, "expected parameter name");
    param.name = name.text;
    param.isGlobalPtr = false;
  }

  return param;
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

std::vector<std::unique_ptr<Stmt>> Parser::parseBlock() {
  consume(TokenKind::LBrace, "expected '{'");
  std::vector<std::unique_ptr<Stmt>> stmts;
  while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
    stmts.push_back(parseStatement());
  }
  consume(TokenKind::RBrace, "expected '}'");
  return stmts;
}

std::unique_ptr<Stmt> Parser::parseStatement() {
  if (check(TokenKind::Int))
    return parseVarDecl();
  if (check(TokenKind::For))
    return parseForStmt();
  if (check(TokenKind::If))
    return parseIfStmt();
  return parseAssignmentOrExpr();
}

std::unique_ptr<VarDeclStmt> Parser::parseVarDecl() {
  Location loc = currentLoc();
  consume(TokenKind::Int, "expected 'int'");
  Token name = consume(TokenKind::Identifier, "expected variable name");
  consume(TokenKind::Equal, "expected '='");
  auto init = parseExpr();
  consume(TokenKind::Semicolon, "expected ';'");
  return std::make_unique<VarDeclStmt>(name.text, std::move(init), loc);
}

std::unique_ptr<Stmt> Parser::parseAssignmentOrExpr() {
  Location loc = currentLoc();
  Token name = consume(TokenKind::Identifier, "expected identifier");

  // Array store: name[index] = value;
  if (check(TokenKind::LBracket)) {
    advance(); // consume '['
    auto index = parseExpr();
    consume(TokenKind::RBracket, "expected ']'");
    consume(TokenKind::Equal, "expected '='");
    auto value = parseExpr();
    consume(TokenKind::Semicolon, "expected ';'");
    return std::make_unique<ArrayStoreStmt>(name.text, std::move(index),
                                            std::move(value), loc);
  }

  // Simple assignment: name = value;
  consume(TokenKind::Equal, "expected '='");
  auto value = parseExpr();
  consume(TokenKind::Semicolon, "expected ';'");
  return std::make_unique<AssignmentStmt>(name.text, std::move(value), loc);
}

std::unique_ptr<ForStmt> Parser::parseForStmt() {
  Location loc = currentLoc();
  auto stmt = std::make_unique<ForStmt>(loc);

  consume(TokenKind::For, "expected 'for'");
  consume(TokenKind::LParen, "expected '('");

  // Init: int i = 0;
  stmt->init = parseVarDecl();

  // Condition: i < N
  stmt->condition = parseExpr();
  consume(TokenKind::Semicolon, "expected ';'");

  // Update: i = i + 1
  Token iterName = consume(TokenKind::Identifier, "expected iterator variable");
  stmt->iterVar = iterName.text;
  consume(TokenKind::Equal, "expected '='");
  stmt->iterUpdate = parseExpr();

  consume(TokenKind::RParen, "expected ')'");

  stmt->body = parseBlock();

  return stmt;
}

std::unique_ptr<IfStmt> Parser::parseIfStmt() {
  Location loc = currentLoc();
  auto stmt = std::make_unique<IfStmt>(loc);

  consume(TokenKind::If, "expected 'if'");
  consume(TokenKind::LParen, "expected '('");
  stmt->condition = parseExpr();
  consume(TokenKind::RParen, "expected ')'");

  stmt->thenBody = parseBlock();

  if (check(TokenKind::Else)) {
    advance();
    stmt->elseBody = parseBlock();
  }

  return stmt;
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

std::unique_ptr<Expr> Parser::parseExpr() { return parseComparison(); }

std::unique_ptr<Expr> Parser::parseComparison() {
  auto lhs = parseAdditive();

  while (check(TokenKind::EqualEqual) || check(TokenKind::BangEqual) ||
         check(TokenKind::Less) || check(TokenKind::Greater) ||
         check(TokenKind::LessEqual) || check(TokenKind::GreaterEqual)) {
    Location loc = currentLoc();
    Token op = advance();
    auto rhs = parseAdditive();

    BinOp binOp;
    switch (op.kind) {
    case TokenKind::EqualEqual:
      binOp = BinOp::Eq;
      break;
    case TokenKind::BangEqual:
      binOp = BinOp::Neq;
      break;
    case TokenKind::Less:
      binOp = BinOp::Lt;
      break;
    case TokenKind::Greater:
      binOp = BinOp::Gt;
      break;
    case TokenKind::LessEqual:
      binOp = BinOp::Leq;
      break;
    case TokenKind::GreaterEqual:
      binOp = BinOp::Geq;
      break;
    default:
      error("unexpected comparison operator");
    }

    lhs =
        std::make_unique<BinaryOpExpr>(binOp, std::move(lhs), std::move(rhs), loc);
  }

  return lhs;
}

std::unique_ptr<Expr> Parser::parseAdditive() {
  auto lhs = parseMultiplicative();

  while (check(TokenKind::Plus) || check(TokenKind::Minus)) {
    Location loc = currentLoc();
    Token op = advance();
    auto rhs = parseMultiplicative();
    BinOp binOp = (op.kind == TokenKind::Plus) ? BinOp::Add : BinOp::Sub;
    lhs =
        std::make_unique<BinaryOpExpr>(binOp, std::move(lhs), std::move(rhs), loc);
  }

  return lhs;
}

std::unique_ptr<Expr> Parser::parseMultiplicative() {
  auto lhs = parsePrimary();

  while (check(TokenKind::Star) || check(TokenKind::Slash)) {
    Location loc = currentLoc();
    Token op = advance();
    auto rhs = parsePrimary();
    BinOp binOp = (op.kind == TokenKind::Star) ? BinOp::Mul : BinOp::Div;
    lhs =
        std::make_unique<BinaryOpExpr>(binOp, std::move(lhs), std::move(rhs), loc);
  }

  return lhs;
}

std::unique_ptr<Expr> Parser::parsePrimary() {
  Location loc = currentLoc();

  // Built-in variables
  if (check(TokenKind::ThreadIdx)) {
    advance();
    return std::make_unique<BuiltinVarExpr>(BuiltinVar::ThreadIdx, loc);
  }
  if (check(TokenKind::BlockIdx)) {
    advance();
    return std::make_unique<BuiltinVarExpr>(BuiltinVar::BlockIdx, loc);
  }
  if (check(TokenKind::BlockDim)) {
    advance();
    return std::make_unique<BuiltinVarExpr>(BuiltinVar::BlockDim, loc);
  }

  // Integer literal
  if (check(TokenKind::IntLiteral)) {
    Token tok = advance();
    int val = std::stoi(tok.text);
    return std::make_unique<IntLiteralExpr>(val, loc);
  }

  // Parenthesized expression
  if (check(TokenKind::LParen)) {
    advance();
    auto expr = parseExpr();
    consume(TokenKind::RParen, "expected ')'");
    return expr;
  }

  // Identifier or array index
  if (check(TokenKind::Identifier)) {
    Token name = advance();
    if (check(TokenKind::LBracket)) {
      advance(); // consume '['
      auto index = parseExpr();
      consume(TokenKind::RBracket, "expected ']'");
      return std::make_unique<ArrayIndexExpr>(name.text, std::move(index), loc);
    }
    return std::make_unique<IdentifierExpr>(name.text, loc);
  }

  error("expected expression");
}

} // namespace tgc
