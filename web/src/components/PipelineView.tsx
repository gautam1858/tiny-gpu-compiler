import { useState } from 'react';
import { CompilationStage } from '../compiler/types';

interface PipelineViewProps {
  stages: CompilationStage[];
  source: string;
}

export function PipelineView({ stages, source }: PipelineViewProps) {
  const [selectedStage, setSelectedStage] = useState(0);

  const allStages = [
    { name: 'Source Code', ir: source },
    ...stages,
  ];

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Stage tabs */}
      <div
        style={{
          display: 'flex',
          gap: '2px',
          padding: '4px',
          background: '#111',
          overflowX: 'auto',
          flexShrink: 0,
        }}
      >
        {allStages.map((stage, i) => (
          <button
            key={i}
            onClick={() => setSelectedStage(i)}
            style={{
              padding: '6px 12px',
              fontSize: '11px',
              fontWeight: selectedStage === i ? 700 : 400,
              background: selectedStage === i ? '#2d5a3d' : '#1a1a2e',
              color: selectedStage === i ? '#7fff7f' : '#888',
              border: selectedStage === i ? '1px solid #4a8a5a' : '1px solid #333',
              borderRadius: '4px',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
              transition: 'all 0.15s',
            }}
          >
            {i === 0 ? '\u2460' : i === 1 ? '\u2461' : '\u2462'} {stage.name}
          </button>
        ))}
      </div>

      {/* Arrow indicator */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '4px 8px',
          background: '#111',
          fontSize: '11px',
          color: '#666',
          gap: '4px',
          flexShrink: 0,
        }}
      >
        {allStages.map((_, i) => (
          <span key={i} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: i <= selectedStage ? '#4ec9b0' : '#333',
                transition: 'background 0.3s',
              }}
            />
            {i < allStages.length - 1 && (
              <span style={{ color: i < selectedStage ? '#4ec9b0' : '#333' }}>{'\u2192'}</span>
            )}
          </span>
        ))}
      </div>

      {/* IR content */}
      <div
        style={{
          flex: 1,
          overflow: 'auto',
          background: '#1a1a2e',
          border: '1px solid #333',
          borderTop: 'none',
        }}
      >
        <pre
          style={{
            margin: 0,
            padding: '12px',
            fontSize: '13px',
            lineHeight: '1.5',
            fontFamily: '"JetBrains Mono", "Fira Code", "Cascadia Code", monospace',
            color: '#e0e0e0',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
          }}
        >
          {highlightIR(allStages[selectedStage]?.ir || '')}
        </pre>
      </div>
    </div>
  );
}

/** Simple syntax highlighting for MLIR-like IR */
function highlightIR(ir: string): JSX.Element[] {
  return ir.split('\n').map((line, i) => {
    let highlighted = line;
    // Color tinygpu ops
    highlighted = highlighted.replace(
      /\b(tinygpu\.\w+)\b/g,
      '\x01OP$1\x02'
    );
    // Color registers and SSA values
    highlighted = highlighted.replace(/%(\w+)/g, '\x01REG%$1\x02');
    // Color attributes
    highlighted = highlighted.replace(/\{([^}]+)\}/g, '\x01ATTR{$1}\x02');
    // Color types
    highlighted = highlighted.replace(/: (i\d+)/g, ': \x01TYPE$1\x02');
    // Color comments
    highlighted = highlighted.replace(/(\/\/.*)$/g, '\x01COMMENT$1\x02');
    // Color keywords
    highlighted = highlighted.replace(
      /\b(kernel|global|int|for|if|else)\b/g,
      '\x01KW$1\x02'
    );

    const parts: JSX.Element[] = [];
    let remaining = highlighted;
    let key = 0;

    while (remaining.length > 0) {
      const opIdx = remaining.indexOf('\x01OP');
      const regIdx = remaining.indexOf('\x01REG');
      const attrIdx = remaining.indexOf('\x01ATTR');
      const typeIdx = remaining.indexOf('\x01TYPE');
      const commentIdx = remaining.indexOf('\x01COMMENT');
      const kwIdx = remaining.indexOf('\x01KW');

      const indices = [
        { idx: opIdx, prefix: '\x01OP', color: '#c586c0', end: '\x02' },
        { idx: regIdx, prefix: '\x01REG', color: '#4ec9b0', end: '\x02' },
        { idx: attrIdx, prefix: '\x01ATTR', color: '#ce9178', end: '\x02' },
        { idx: typeIdx, prefix: '\x01TYPE', color: '#569cd6', end: '\x02' },
        { idx: commentIdx, prefix: '\x01COMMENT', color: '#6a9955', end: '\x02' },
        { idx: kwIdx, prefix: '\x01KW', color: '#c586c0', end: '\x02' },
      ].filter((x) => x.idx >= 0);

      if (indices.length === 0) {
        parts.push(<span key={key++}>{remaining}</span>);
        break;
      }

      indices.sort((a, b) => a.idx - b.idx);
      const first = indices[0];

      if (first.idx > 0) {
        parts.push(<span key={key++}>{remaining.slice(0, first.idx)}</span>);
      }

      const afterPrefix = remaining.slice(first.idx + first.prefix.length);
      const endIdx = afterPrefix.indexOf(first.end);
      const text = endIdx >= 0 ? afterPrefix.slice(0, endIdx) : afterPrefix;

      parts.push(
        <span key={key++} style={{ color: first.color }}>
          {text}
        </span>
      );

      remaining = endIdx >= 0 ? afterPrefix.slice(endIdx + 1) : '';
    }

    return (
      <div key={i} style={{ minHeight: '1.5em' }}>
        {parts}
      </div>
    );
  });
}
