/**
 * In-browser tiny-gpu compiler.
 * A lightweight JavaScript reimplementation of the C++ compiler pipeline.
 * This allows the web visualizer to compile kernels without a server.
 *
 * The full C++ MLIR-based compiler is the canonical implementation.
 * This JS version covers the core subset for interactive visualization.
 */

import { CompilationTrace, Instruction } from './types';

// =============================================================================
// Lexer
// =============================================================================

type TokenKind =
  | 'kernel' | 'global' | 'int' | 'for' | 'if' | 'else'
  | 'threadIdx' | 'blockIdx' | 'blockDim'
  | 'ident' | 'number'
  | '+' | '-' | '*' | '/' | '=' | '==' | '!=' | '<' | '>' | '<=' | '>='
  | '(' | ')' | '{' | '}' | '[' | ']' | ';' | ','
  | 'eof' | 'error';

interface Token {
  kind: TokenKind;
  text: string;
  line: number;
  col: number;
}

function lex(source: string): Token[] {
  const tokens: Token[] = [];
  let pos = 0, line = 1, col = 1;
  const keywords = new Set(['kernel', 'global', 'int', 'for', 'if', 'else']);
  const builtins = new Set(['threadIdx', 'blockIdx', 'blockDim']);

  while (pos < source.length) {
    // Skip whitespace
    if (/\s/.test(source[pos])) {
      if (source[pos] === '\n') { line++; col = 1; } else { col++; }
      pos++;
      continue;
    }
    // Skip comments
    if (source[pos] === '/' && source[pos + 1] === '/') {
      while (pos < source.length && source[pos] !== '\n') pos++;
      continue;
    }
    if (source[pos] === '/' && source[pos + 1] === '*') {
      pos += 2; col += 2;
      while (pos + 1 < source.length && !(source[pos] === '*' && source[pos + 1] === '/')) {
        if (source[pos] === '\n') { line++; col = 1; } else { col++; }
        pos++;
      }
      pos += 2; col += 2;
      continue;
    }

    const startCol = col;

    // Identifier or keyword
    if (/[a-zA-Z_]/.test(source[pos])) {
      let text = '';
      while (pos < source.length && /[a-zA-Z0-9_]/.test(source[pos])) {
        text += source[pos]; pos++; col++;
      }
      let kind: TokenKind = 'ident';
      if (keywords.has(text)) kind = text as TokenKind;
      if (builtins.has(text)) kind = text as TokenKind;
      tokens.push({ kind, text, line, col: startCol });
      continue;
    }

    // Number
    if (/\d/.test(source[pos])) {
      let text = '';
      while (pos < source.length && /\d/.test(source[pos])) {
        text += source[pos]; pos++; col++;
      }
      tokens.push({ kind: 'number', text, line, col: startCol });
      continue;
    }

    // Two-char operators
    const two = source.slice(pos, pos + 2);
    if (['==', '!=', '<=', '>='].includes(two)) {
      tokens.push({ kind: two as TokenKind, text: two, line, col: startCol });
      pos += 2; col += 2;
      continue;
    }

    // Single-char tokens
    const singleOps = '+-*/=<>(){}[];,';
    if (singleOps.includes(source[pos])) {
      tokens.push({ kind: source[pos] as TokenKind, text: source[pos], line, col: startCol });
      pos++; col++;
      continue;
    }

    tokens.push({ kind: 'error', text: source[pos], line, col: startCol });
    pos++; col++;
  }

  tokens.push({ kind: 'eof', text: '', line, col });
  return tokens;
}

// =============================================================================
// AST
// =============================================================================

type Expr =
  | { type: 'int'; value: number }
  | { type: 'ident'; name: string }
  | { type: 'builtin'; name: 'threadIdx' | 'blockIdx' | 'blockDim' }
  | { type: 'binop'; op: string; lhs: Expr; rhs: Expr }
  | { type: 'index'; array: string; index: Expr };

type Stmt =
  | { type: 'vardecl'; name: string; init: Expr }
  | { type: 'assign'; name: string; value: Expr }
  | { type: 'store'; array: string; index: Expr; value: Expr }
  | { type: 'for'; init: Stmt; cond: Expr; iterVar: string; iterExpr: Expr; body: Stmt[] }
  | { type: 'if'; cond: Expr; then: Stmt[]; else: Stmt[] };

interface Param { name: string; isPtr: boolean; }
interface Kernel { name: string; params: Param[]; body: Stmt[]; }

// =============================================================================
// Parser
// =============================================================================

function parse(tokens: Token[]): Kernel {
  let pos = 0;
  const peek = () => tokens[pos];
  const advance = () => tokens[pos++];
  const expect = (kind: TokenKind) => {
    const t = advance();
    if (t.kind !== kind) throw new Error(`Expected ${kind}, got ${t.kind} "${t.text}" at ${t.line}:${t.col}`);
    return t;
  };
  const match = (kind: TokenKind) => { if (peek().kind === kind) { advance(); return true; } return false; };

  expect('kernel');
  const name = expect('ident').text;
  expect('(');
  const params: Param[] = [];
  if (peek().kind !== ')') {
    do {
      if (peek().kind === 'global') {
        advance(); expect('int'); expect('*');
        params.push({ name: expect('ident').text, isPtr: true });
      } else {
        expect('int');
        params.push({ name: expect('ident').text, isPtr: false });
      }
    } while (match(','));
  }
  expect(')');

  function parseBlock(): Stmt[] {
    expect('{');
    const stmts: Stmt[] = [];
    while (peek().kind !== '}' && peek().kind !== 'eof') stmts.push(parseStmt());
    expect('}');
    return stmts;
  }

  function parseStmt(): Stmt {
    if (peek().kind === 'int') {
      advance();
      const vname = expect('ident').text;
      expect('=');
      const init = parseExpr();
      expect(';');
      return { type: 'vardecl', name: vname, init };
    }
    if (peek().kind === 'for') {
      advance(); expect('(');
      expect('int');
      const initName = expect('ident').text;
      expect('=');
      const initExpr = parseExpr();
      expect(';');
      const init: Stmt = { type: 'vardecl', name: initName, init: initExpr };
      const cond = parseExpr();
      expect(';');
      const iterVar = expect('ident').text;
      expect('=');
      const iterExpr = parseExpr();
      expect(')');
      const body = parseBlock();
      return { type: 'for', init, cond, iterVar, iterExpr, body };
    }
    if (peek().kind === 'if') {
      advance(); expect('(');
      const cond = parseExpr();
      expect(')');
      const thenBody = parseBlock();
      let elseBody: Stmt[] = [];
      if (match('else')) elseBody = parseBlock();
      return { type: 'if', cond, then: thenBody, else: elseBody };
    }
    // Assignment or array store
    const ident = expect('ident');
    if (peek().kind === '[') {
      advance();
      const index = parseExpr();
      expect(']'); expect('=');
      const value = parseExpr();
      expect(';');
      return { type: 'store', array: ident.text, index, value };
    }
    expect('=');
    const value = parseExpr();
    expect(';');
    return { type: 'assign', name: ident.text, value };
  }

  function parseExpr(): Expr { return parseComparison(); }

  function parseComparison(): Expr {
    let lhs = parseAdditive();
    while (['==', '!=', '<', '>', '<=', '>='].includes(peek().kind)) {
      const op = advance().text;
      lhs = { type: 'binop', op, lhs, rhs: parseAdditive() };
    }
    return lhs;
  }

  function parseAdditive(): Expr {
    let lhs = parseMultiplicative();
    while (peek().kind === '+' || peek().kind === '-') {
      const op = advance().text;
      lhs = { type: 'binop', op, lhs, rhs: parseMultiplicative() };
    }
    return lhs;
  }

  function parseMultiplicative(): Expr {
    let lhs = parsePrimary();
    while (peek().kind === '*' || peek().kind === '/') {
      const op = advance().text;
      lhs = { type: 'binop', op, lhs, rhs: parsePrimary() };
    }
    return lhs;
  }

  function parsePrimary(): Expr {
    if (peek().kind === 'number') {
      return { type: 'int', value: parseInt(advance().text) };
    }
    if (peek().kind === 'threadIdx' || peek().kind === 'blockIdx' || peek().kind === 'blockDim') {
      return { type: 'builtin', name: advance().kind as 'threadIdx' | 'blockIdx' | 'blockDim' };
    }
    if (peek().kind === '(') {
      advance();
      const e = parseExpr();
      expect(')');
      return e;
    }
    if (peek().kind === 'ident') {
      const name = advance().text;
      if (peek().kind === '[') {
        advance();
        const index = parseExpr();
        expect(']');
        return { type: 'index', array: name, index };
      }
      return { type: 'ident', name };
    }
    throw new Error(`Unexpected token: ${peek().kind} "${peek().text}" at ${peek().line}:${peek().col}`);
  }

  const body = parseBlock();
  return { name, params, body };
}

// =============================================================================
// MLIR Generation (produces string IR + instruction list)
// =============================================================================

interface IRInstruction {
  op: string;
  rd?: number;
  rs?: number;
  rt?: number;
  imm?: number;
  nzp?: number;
  target?: number;
  asm: string;
}

function compile(kernel: Kernel): { ir: string; regIR: string; instructions: Instruction[] } {
  const paramBases: Record<string, number> = {};
  let ptrIdx = 0;
  let scalarAddr = 192;
  for (const p of kernel.params) {
    if (p.isPtr) { paramBases[p.name] = ptrIdx * 64; ptrIdx++; }
    else { paramBases[p.name] = scalarAddr++; }
  }

  // SSA value tracking
  let nextSSA = 0;
  const vars: Record<string, number> = {};
  const irLines: string[] = [];
  const irOps: IRInstruction[] = [];

  // Register allocation
  let nextReg = 0;
  const ssaToReg: Record<number, number> = {};

  function allocReg(ssa: number): number {
    if (ssaToReg[ssa] !== undefined) return ssaToReg[ssa];
    const r = nextReg++;
    ssaToReg[ssa] = r;
    return r;
  }

  function regOf(ssa: number): number {
    return ssaToReg[ssa] ?? ssa;
  }

  function regName(r: number): string {
    if (r === 13) return '%blockIdx';
    if (r === 14) return '%blockDim';
    if (r === 15) return '%threadIdx';
    return `R${r}`;
  }

  function genExpr(expr: Expr): number {
    switch (expr.type) {
      case 'int': {
        const ssa = nextSSA++;
        irLines.push(`  %${ssa} = tinygpu.const ${expr.value} : i8`);
        const rd = allocReg(ssa);
        irOps.push({ op: 'CONST', rd, imm: expr.value & 0xff, asm: `CONST ${regName(rd)}, #${expr.value}` });
        return ssa;
      }
      case 'builtin': {
        const ssa = nextSSA++;
        const opName = expr.name === 'threadIdx' ? 'thread_id' : expr.name === 'blockIdx' ? 'block_id' : 'block_dim';
        irLines.push(`  %${ssa} = tinygpu.${opName} : i8`);
        const fixedReg = expr.name === 'threadIdx' ? 15 : expr.name === 'blockIdx' ? 13 : 14;
        ssaToReg[ssa] = fixedReg;
        return ssa;
      }
      case 'ident': {
        if (vars[expr.name] !== undefined) return vars[expr.name];
        // Check if it's a scalar param
        if (paramBases[expr.name] !== undefined) {
          const addrSSA = nextSSA++;
          irLines.push(`  %${addrSSA} = tinygpu.const ${paramBases[expr.name]} : i8`);
          const addrReg = allocReg(addrSSA);
          irOps.push({ op: 'CONST', rd: addrReg, imm: paramBases[expr.name], asm: `CONST ${regName(addrReg)}, #${paramBases[expr.name]}` });

          const valSSA = nextSSA++;
          irLines.push(`  %${valSSA} = tinygpu.load %${addrSSA} : i8`);
          // Reuse address register (address is consumed by LDR)
          ssaToReg[valSSA] = addrReg;
          irOps.push({ op: 'LDR', rd: addrReg, rs: addrReg, asm: `LDR ${regName(addrReg)}, [${regName(addrReg)}]` });
          vars[expr.name] = valSSA;
          return valSSA;
        }
        throw new Error(`Undefined variable: ${expr.name}`);
      }
      case 'binop': {
        const lhs = genExpr(expr.lhs);
        const rhs = genExpr(expr.rhs);
        const ssa = nextSSA++;
        const opMap: Record<string, string> = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div' };
        const cmpOps = ['==', '!=', '<', '>', '<=', '>='];
        if (cmpOps.includes(expr.op)) {
          irLines.push(`  %${ssa} = tinygpu.cmp %${lhs}, %${rhs} : i8`);
          // CMP only sets NZP flags, no register output needed
          ssaToReg[ssa] = -1;
          irOps.push({ op: 'CMP', rs: regOf(lhs), rt: regOf(rhs), asm: `CMP ${regName(regOf(lhs))}, ${regName(regOf(rhs))}` });
        } else {
          const mlirOp = opMap[expr.op] || 'add';
          irLines.push(`  %${ssa} = tinygpu.${mlirOp} %${lhs}, %${rhs} : i8`);
          // Reuse an operand register if it's a temporary (not a named variable)
          const liveVarSSAs = new Set(Object.values(vars));
          let rd: number;
          if (!liveVarSSAs.has(lhs) && regOf(lhs) >= 0 && regOf(lhs) < 13) {
            rd = regOf(lhs);
            ssaToReg[ssa] = rd;
          } else if (!liveVarSSAs.has(rhs) && regOf(rhs) >= 0 && regOf(rhs) < 13) {
            rd = regOf(rhs);
            ssaToReg[ssa] = rd;
          } else {
            rd = allocReg(ssa);
          }
          const asmOp = mlirOp.toUpperCase();
          irOps.push({ op: asmOp, rd, rs: regOf(lhs), rt: regOf(rhs), asm: `${asmOp} ${regName(rd)}, ${regName(regOf(lhs))}, ${regName(regOf(rhs))}` });
        }
        return ssa;
      }
      case 'index': {
        const indexSSA = genExpr(expr.index);
        const base = paramBases[expr.array] ?? 0;
        let addrSSA = indexSSA;
        if (base !== 0) {
          const baseSSA = nextSSA++;
          irLines.push(`  %${baseSSA} = tinygpu.const ${base} : i8`);
          const baseReg = allocReg(baseSSA);
          irOps.push({ op: 'CONST', rd: baseReg, imm: base, asm: `CONST ${regName(baseReg)}, #${base}` });

          const sumSSA = nextSSA++;
          irLines.push(`  %${sumSSA} = tinygpu.add %${baseSSA}, %${indexSSA} : i8`);
          // Reuse base register (base constant is consumed by ADD)
          ssaToReg[sumSSA] = baseReg;
          irOps.push({ op: 'ADD', rd: baseReg, rs: baseReg, rt: regOf(indexSSA), asm: `ADD ${regName(baseReg)}, ${regName(baseReg)}, ${regName(regOf(indexSSA))}` });
          addrSSA = sumSSA;
        }

        const valSSA = nextSSA++;
        irLines.push(`  %${valSSA} = tinygpu.load %${addrSSA} : i8`);
        // Reuse address register if it's a temporary (not a named variable)
        const liveVarSSAs = new Set(Object.values(vars));
        const addrReg = regOf(addrSSA);
        if (!liveVarSSAs.has(addrSSA) && addrReg >= 0 && addrReg < 13) {
          ssaToReg[valSSA] = addrReg;
          irOps.push({ op: 'LDR', rd: addrReg, rs: addrReg, asm: `LDR ${regName(addrReg)}, [${regName(addrReg)}]` });
        } else {
          const valReg = allocReg(valSSA);
          irOps.push({ op: 'LDR', rd: valReg, rs: addrReg, asm: `LDR ${regName(valReg)}, [${regName(addrReg)}]` });
        }
        return valSSA;
      }
    }
  }

  function genStmt(stmt: Stmt): void {
    switch (stmt.type) {
      case 'vardecl': {
        const ssa = genExpr(stmt.init);
        vars[stmt.name] = ssa;
        break;
      }
      case 'assign': {
        const ssa = genExpr(stmt.value);
        vars[stmt.name] = ssa;
        break;
      }
      case 'store': {
        const indexSSA = genExpr(stmt.index);
        const valueSSA = genExpr(stmt.value);
        const base = paramBases[stmt.array] ?? 0;
        let addrSSA = indexSSA;
        if (base !== 0) {
          const baseSSA = nextSSA++;
          irLines.push(`  %${baseSSA} = tinygpu.const ${base} : i8`);
          const baseReg = allocReg(baseSSA);
          irOps.push({ op: 'CONST', rd: baseReg, imm: base, asm: `CONST ${regName(baseReg)}, #${base}` });

          const sumSSA = nextSSA++;
          irLines.push(`  %${sumSSA} = tinygpu.add %${baseSSA}, %${indexSSA} : i8`);
          // Reuse base register (base constant is consumed by ADD)
          ssaToReg[sumSSA] = baseReg;
          irOps.push({ op: 'ADD', rd: baseReg, rs: baseReg, rt: regOf(indexSSA), asm: `ADD ${regName(baseReg)}, ${regName(baseReg)}, ${regName(regOf(indexSSA))}` });
          addrSSA = sumSSA;
        }
        irLines.push(`  tinygpu.store %${addrSSA}, %${valueSSA} : i8`);
        irOps.push({ op: 'STR', rs: regOf(addrSSA), rt: regOf(valueSSA), asm: `STR [${regName(regOf(addrSSA))}], ${regName(regOf(valueSSA))}` });
        break;
      }
      case 'for': {
        genStmt(stmt.init);

        // Snapshot variable -> register mappings before the loop
        // so we can fix loop-carried variables after the body
        const preLoopRegs: Record<string, number> = {};
        for (const [name, ssa] of Object.entries(vars)) {
          preLoopRegs[name] = regOf(ssa);
        }

        const loopStart = irOps.length;
        // Condition
        const condSSA = genExpr(stmt.cond);
        const branchIdx = irOps.length;
        irOps.push({ op: 'BRnzp', nzp: 0b010, target: 0, rs: regOf(condSSA), asm: 'BRnzp (exit)' }); // placeholder

        // Body
        for (const s of stmt.body) genStmt(s);

        // Fix loop-carried variables: any variable that existed before the loop
        // and was reassigned in the body must write back to its original register,
        // so the next iteration reads the updated value from the same register.
        for (const [name, origReg] of Object.entries(preLoopRegs)) {
          if (name === stmt.iterVar) continue; // handled separately in update below
          const currentSSA = vars[name];
          if (currentSSA === undefined) continue;
          const currentReg = regOf(currentSSA);
          if (currentReg !== origReg) {
            for (let i = irOps.length - 1; i > branchIdx; i--) {
              if (irOps[i].rd === currentReg) {
                irOps[i].rd = origReg;
                irOps[i].asm = irOps[i].asm.replace(/^(\w+)\s+R\d+/, `$1 ${regName(origReg)}`);
                ssaToReg[currentSSA] = origReg;
                break;
              }
            }
          }
        }

        // Update -- must write result back to the original loop variable's register
        // so that the CMP at loopStart (which references the original register) sees the new value
        const origLoopVarSSA = vars[stmt.iterVar];
        const origReg = regOf(origLoopVarSSA);
        const updateSSA = genExpr(stmt.iterExpr);
        // Patch the last emitted instruction to write to the original register
        const lastOp = irOps[irOps.length - 1];
        if (lastOp.rd !== undefined) {
          lastOp.rd = origReg;
          lastOp.asm = lastOp.asm.replace(/^(\w+)\s+R\d+/, `$1 ${regName(origReg)}`);
        }
        ssaToReg[updateSSA] = origReg;
        vars[stmt.iterVar] = updateSSA;

        // Jump back
        irOps.push({ op: 'JMP', target: loopStart, nzp: 0b111, asm: `JMP #${loopStart}` });

        // Patch branch target
        const exitAddr = irOps.length;
        irOps[branchIdx].target = exitAddr;
        irOps[branchIdx].asm = `BRnzp 2, #${exitAddr}`;
        break;
      }
      case 'if': {
        const condSSA = genExpr(stmt.cond);
        const branchIdx = irOps.length;
        irOps.push({ op: 'BRnzp', nzp: 0b010, target: 0, rs: regOf(condSSA), asm: 'BRnzp (else)' });

        for (const s of stmt.then) genStmt(s);

        if (stmt.else.length > 0) {
          const jmpIdx = irOps.length;
          irOps.push({ op: 'JMP', target: 0, nzp: 0b111, asm: 'JMP (end)' });
          const elseAddr = irOps.length;
          irOps[branchIdx].target = elseAddr;
          irOps[branchIdx].asm = `BRnzp 2, #${elseAddr}`;
          for (const s of stmt.else) genStmt(s);
          const endAddr = irOps.length;
          irOps[jmpIdx].target = endAddr;
          irOps[jmpIdx].asm = `JMP #${endAddr}`;
        } else {
          const endAddr = irOps.length;
          irOps[branchIdx].target = endAddr;
          irOps[branchIdx].asm = `BRnzp 2, #${endAddr}`;
        }
        break;
      }
    }
  }

  // Generate IR
  irLines.push(`tinygpu.func @${kernel.name}() {`);
  for (const stmt of kernel.body) genStmt(stmt);
  irLines.push('  tinygpu.ret');
  irOps.push({ op: 'RET', asm: 'RET' });
  irLines.push('}');

  // Encode binary
  const opcodeMap: Record<string, number> = {
    NOP: 0, BRnzp: 1, CMP: 2, ADD: 3, SUB: 4, MUL: 5, DIV: 6, LDR: 7, STR: 8, CONST: 9, RET: 15, JMP: 1,
  };

  const instructions: Instruction[] = irOps.map((op, i) => {
    const opcode = opcodeMap[op.op] ?? 0;
    let binary = 0;

    switch (op.op) {
      case 'ADD': case 'SUB': case 'MUL': case 'DIV':
        binary = (opcode << 12) | ((op.rd! & 0xf) << 8) | ((op.rs! & 0xf) << 4) | (op.rt! & 0xf);
        break;
      case 'CONST':
        binary = (opcode << 12) | ((op.rd! & 0xf) << 8) | (op.imm! & 0xff);
        break;
      case 'LDR':
        binary = (opcode << 12) | ((op.rd! & 0xf) << 8) | ((op.rs! & 0xf) << 4);
        break;
      case 'STR':
        binary = (opcode << 12) | ((op.rs! & 0xf) << 4) | (op.rt! & 0xf);
        break;
      case 'CMP':
        binary = (opcode << 12) | ((op.rs! & 0xf) << 4) | (op.rt! & 0xf);
        break;
      case 'BRnzp':
        binary = (opcode << 12) | ((op.nzp! & 0x7) << 9) | (op.target! & 0xff);
        break;
      case 'JMP':
        binary = (1 << 12) | (0b111 << 9) | (op.target! & 0xff);
        break;
      case 'RET':
        binary = 0xf000;
        break;
    }

    const bits = binary.toString(2).padStart(16, '0');
    const formattedBits = `${bits.slice(0, 4)} ${bits.slice(4, 8)} ${bits.slice(8, 12)} ${bits.slice(12, 16)}`;

    return {
      addr: i,
      hex: `0x${binary.toString(16).toUpperCase().padStart(4, '0')}`,
      asm: op.asm,
      bits: formattedBits,
    };
  });

  // Build register-allocated IR string
  const regIRLines = [`tinygpu.func @${kernel.name}() {`];
  for (const op of irOps) {
    if (op.op === 'RET') {
      regIRLines.push('  tinygpu.ret');
    } else {
      regIRLines.push(`  // ${op.asm}`);
    }
  }
  regIRLines.push('}');

  return {
    ir: irLines.join('\n'),
    regIR: regIRLines.join('\n'),
    instructions,
  };
}

/** Compile a .tgc source string and return the full trace */
export function compileTGC(source: string): CompilationTrace {
  try {
    const tokens = lex(source);
    const kernel = parse(tokens);
    const { ir, regIR, instructions } = compile(kernel);

    return {
      source,
      stages: [
        { name: 'Frontend \u2192 TinyGPU Dialect', ir },
        { name: 'Register Allocation', ir: regIR },
      ],
      binary: { instructions },
    };
  } catch (e) {
    return {
      source,
      stages: [{ name: 'Error', ir: `Compilation error: ${(e as Error).message}` }],
      binary: { instructions: [] },
    };
  }
}
