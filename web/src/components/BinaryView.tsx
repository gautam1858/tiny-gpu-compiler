import { Instruction } from '../compiler/types';

interface BinaryViewProps {
  instructions: Instruction[];
  highlightAddr?: number;
}

/** Color-coded 16-bit instruction view */
export function BinaryView({ instructions, highlightAddr }: BinaryViewProps) {
  if (instructions.length === 0) {
    return (
      <div style={{ padding: '16px', color: '#666', fontSize: '13px' }}>
        No instructions generated yet. Write a kernel above to compile.
      </div>
    );
  }

  return (
    <div style={{ overflow: 'auto', fontSize: '12px', fontFamily: 'monospace' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ background: '#111', color: '#888', textAlign: 'left' }}>
            <th style={{ padding: '6px 8px', width: '40px' }}>Addr</th>
            <th style={{ padding: '6px 8px', width: '60px' }}>Hex</th>
            <th style={{ padding: '6px 8px', width: '180px' }}>Binary</th>
            <th style={{ padding: '6px 8px' }}>Assembly</th>
          </tr>
        </thead>
        <tbody>
          {instructions.map((inst) => {
            const isHighlighted = inst.addr === highlightAddr;
            const binary = parseInt(inst.hex, 16) || 0;
            const opcode = (binary >> 12) & 0xf;
            const field1 = (binary >> 8) & 0xf;
            const field2 = (binary >> 4) & 0xf;
            const field3 = binary & 0xf;

            return (
              <tr
                key={inst.addr}
                style={{
                  background: isHighlighted ? '#2d3a2d' : 'transparent',
                  borderLeft: isHighlighted ? '3px solid #4ec9b0' : '3px solid transparent',
                  transition: 'background 0.15s',
                }}
              >
                <td style={{ padding: '4px 8px', color: '#666' }}>{inst.addr}</td>
                <td style={{ padding: '4px 8px', color: '#b5cea8' }}>{inst.hex}</td>
                <td style={{ padding: '4px 8px' }}>
                  <BinaryBits
                    opcode={opcode}
                    field1={field1}
                    field2={field2}
                    field3={field3}
                  />
                </td>
                <td style={{ padding: '4px 8px', color: '#e0e0e0' }}>{inst.asm}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Render 16 bits with color coding per field */
function BinaryBits({
  opcode,
  field1,
  field2,
  field3,
}: {
  opcode: number;
  field1: number;
  field2: number;
  field3: number;
}) {
  const toBin4 = (n: number) => n.toString(2).padStart(4, '0');

  return (
    <span>
      <span style={{ color: '#e06c75' }} title="Opcode [15:12]">
        {toBin4(opcode)}
      </span>
      {' '}
      <span style={{ color: '#61afef' }} title="rd / NZP [11:8]">
        {toBin4(field1)}
      </span>
      {' '}
      <span style={{ color: '#98c379' }} title="rs [7:4]">
        {toBin4(field2)}
      </span>
      {' '}
      <span style={{ color: '#d19a66' }} title="rt / imm[3:0] [3:0]">
        {toBin4(field3)}
      </span>
    </span>
  );
}
