#ifndef TGC_FRONTEND_LEXER_H
#define TGC_FRONTEND_LEXER_H

#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace tgc {

enum class TokenKind {
  // Keywords
  Kernel,
  Global,
  Int,
  For,
  If,
  Else,

  // Built-in variables
  ThreadIdx,
  BlockIdx,
  BlockDim,

  // Literals & identifiers
  Identifier,
  IntLiteral,

  // Operators
  Plus,
  Minus,
  Star,
  Slash,
  Equal,
  EqualEqual,
  BangEqual,
  Less,
  Greater,
  LessEqual,
  GreaterEqual,

  // Delimiters
  LParen,
  RParen,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  Semicolon,
  Comma,

  // Special
  Eof,
  Error,
};

struct Token {
  TokenKind kind;
  std::string text;
  int line;
  int col;
};

class Lexer {
public:
  explicit Lexer(llvm::StringRef source);

  Token nextToken();
  Token peekToken();

  const std::string &getSource() const { return source; }

private:
  void skipWhitespaceAndComments();
  Token makeToken(TokenKind kind, int startPos, int length);
  Token lexIdentifierOrKeyword();
  Token lexNumber();

  std::string source;
  int pos = 0;
  int line = 1;
  int col = 1;
  Token peeked;
  bool hasPeeked = false;
};

} // namespace tgc

#endif // TGC_FRONTEND_LEXER_H
