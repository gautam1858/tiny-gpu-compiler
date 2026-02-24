#include "tiny-gpu-compiler/Frontend/Lexer.h"

#include <cctype>

namespace tgc {

Lexer::Lexer(llvm::StringRef src) : source(src.str()) {}

void Lexer::skipWhitespaceAndComments() {
  while (pos < (int)source.size()) {
    // Skip whitespace
    if (std::isspace(source[pos])) {
      if (source[pos] == '\n') {
        line++;
        col = 1;
      } else {
        col++;
      }
      pos++;
      continue;
    }
    // Skip // line comments
    if (pos + 1 < (int)source.size() && source[pos] == '/' &&
        source[pos + 1] == '/') {
      while (pos < (int)source.size() && source[pos] != '\n')
        pos++;
      continue;
    }
    // Skip /* block comments */
    if (pos + 1 < (int)source.size() && source[pos] == '/' &&
        source[pos + 1] == '*') {
      pos += 2;
      col += 2;
      while (pos + 1 < (int)source.size() &&
             !(source[pos] == '*' && source[pos + 1] == '/')) {
        if (source[pos] == '\n') {
          line++;
          col = 1;
        } else {
          col++;
        }
        pos++;
      }
      if (pos + 1 < (int)source.size()) {
        pos += 2;
        col += 2;
      }
      continue;
    }
    break;
  }
}

Token Lexer::makeToken(TokenKind kind, int startPos, int length) {
  Token tok;
  tok.kind = kind;
  tok.text = source.substr(startPos, length);
  tok.line = line;
  tok.col = col - length;
  return tok;
}

Token Lexer::lexIdentifierOrKeyword() {
  int start = pos;
  int startCol = col;
  while (pos < (int)source.size() &&
         (std::isalnum(source[pos]) || source[pos] == '_')) {
    pos++;
    col++;
  }

  std::string text = source.substr(start, pos - start);

  TokenKind kind = TokenKind::Identifier;
  if (text == "kernel")
    kind = TokenKind::Kernel;
  else if (text == "global")
    kind = TokenKind::Global;
  else if (text == "int")
    kind = TokenKind::Int;
  else if (text == "for")
    kind = TokenKind::For;
  else if (text == "if")
    kind = TokenKind::If;
  else if (text == "else")
    kind = TokenKind::Else;
  else if (text == "threadIdx")
    kind = TokenKind::ThreadIdx;
  else if (text == "blockIdx")
    kind = TokenKind::BlockIdx;
  else if (text == "blockDim")
    kind = TokenKind::BlockDim;

  Token tok;
  tok.kind = kind;
  tok.text = text;
  tok.line = line;
  tok.col = startCol;
  return tok;
}

Token Lexer::lexNumber() {
  int start = pos;
  int startCol = col;
  while (pos < (int)source.size() && std::isdigit(source[pos])) {
    pos++;
    col++;
  }

  Token tok;
  tok.kind = TokenKind::IntLiteral;
  tok.text = source.substr(start, pos - start);
  tok.line = line;
  tok.col = startCol;
  return tok;
}

Token Lexer::nextToken() {
  if (hasPeeked) {
    hasPeeked = false;
    return peeked;
  }

  skipWhitespaceAndComments();

  if (pos >= (int)source.size()) {
    Token tok;
    tok.kind = TokenKind::Eof;
    tok.line = line;
    tok.col = col;
    return tok;
  }

  char c = source[pos];

  if (std::isalpha(c) || c == '_')
    return lexIdentifierOrKeyword();

  if (std::isdigit(c))
    return lexNumber();

  // Two-character operators
  if (pos + 1 < (int)source.size()) {
    char c2 = source[pos + 1];
    if (c == '=' && c2 == '=') {
      pos += 2;
      col += 2;
      return makeToken(TokenKind::EqualEqual, pos - 2, 2);
    }
    if (c == '!' && c2 == '=') {
      pos += 2;
      col += 2;
      return makeToken(TokenKind::BangEqual, pos - 2, 2);
    }
    if (c == '<' && c2 == '=') {
      pos += 2;
      col += 2;
      return makeToken(TokenKind::LessEqual, pos - 2, 2);
    }
    if (c == '>' && c2 == '=') {
      pos += 2;
      col += 2;
      return makeToken(TokenKind::GreaterEqual, pos - 2, 2);
    }
  }

  // Single-character tokens
  pos++;
  col++;
  switch (c) {
  case '+':
    return makeToken(TokenKind::Plus, pos - 1, 1);
  case '-':
    return makeToken(TokenKind::Minus, pos - 1, 1);
  case '*':
    return makeToken(TokenKind::Star, pos - 1, 1);
  case '/':
    return makeToken(TokenKind::Slash, pos - 1, 1);
  case '=':
    return makeToken(TokenKind::Equal, pos - 1, 1);
  case '<':
    return makeToken(TokenKind::Less, pos - 1, 1);
  case '>':
    return makeToken(TokenKind::Greater, pos - 1, 1);
  case '(':
    return makeToken(TokenKind::LParen, pos - 1, 1);
  case ')':
    return makeToken(TokenKind::RParen, pos - 1, 1);
  case '{':
    return makeToken(TokenKind::LBrace, pos - 1, 1);
  case '}':
    return makeToken(TokenKind::RBrace, pos - 1, 1);
  case '[':
    return makeToken(TokenKind::LBracket, pos - 1, 1);
  case ']':
    return makeToken(TokenKind::RBracket, pos - 1, 1);
  case ';':
    return makeToken(TokenKind::Semicolon, pos - 1, 1);
  case ',':
    return makeToken(TokenKind::Comma, pos - 1, 1);
  default: {
    Token tok;
    tok.kind = TokenKind::Error;
    tok.text = std::string(1, c);
    tok.line = line;
    tok.col = col - 1;
    return tok;
  }
  }
}

Token Lexer::peekToken() {
  if (!hasPeeked) {
    peeked = nextToken();
    hasPeeked = true;
  }
  return peeked;
}

} // namespace tgc
