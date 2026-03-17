import { useState, useCallback, useEffect, useRef } from 'react';
import { Editor } from './components/Editor';
import { PipelineView } from './components/PipelineView';
import { BinaryView } from './components/BinaryView';
import { GPUSimulator } from './components/GPUSimulator';
import { AnalysisPanel } from './components/AnalysisPanel';
import { compileTGC } from './compiler/TGCCompiler';
import { EXAMPLES } from './examples';
import { CompilationTrace, SimulationState } from './compiler/types';

type RightPanel = 'simulator' | 'analysis';

export default function App() {
  const [source, setSource] = useState(EXAMPLES[0].source);
  const [selectedExample, setSelectedExample] = useState(0);
  const [trace, setTrace] = useState<CompilationTrace>(EXAMPLES[0].trace);
  const [highlightAddr, setHighlightAddr] = useState<number | undefined>(undefined);
  const [rightPanel, setRightPanel] = useState<RightPanel>('simulator');
  const compileTimeout = useRef<number | null>(null);

  const handleSourceChange = useCallback((newSource: string) => {
    setSource(newSource);
    if (compileTimeout.current) clearTimeout(compileTimeout.current);
    compileTimeout.current = window.setTimeout(() => {
      const result = compileTGC(newSource);
      setTrace(result);
    }, 300);
  }, []);

  const handleExampleChange = useCallback((index: number) => {
    setSelectedExample(index);
    setSource(EXAMPLES[index].source);
    setTrace(EXAMPLES[index].trace);
  }, []);

  const handleCycleChange = useCallback((state: SimulationState) => {
    const activeThread = state.threads.find((t) => !t.done);
    setHighlightAddr(activeThread?.pc);
  }, []);

  useEffect(() => {
    setTrace(EXAMPLES[0].trace);
  }, []);

  const example = EXAMPLES[selectedExample];

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '8px 16px',
          background: '#111',
          borderBottom: '1px solid #222',
          gap: '12px',
          flexShrink: 0,
        }}
      >
        <h1
          style={{
            fontSize: '16px',
            fontWeight: 700,
            color: '#4ec9b0',
            fontFamily: 'monospace',
            letterSpacing: '-0.5px',
          }}
        >
          tiny-gpu-compiler
        </h1>
        <span style={{ color: '#555', fontSize: '12px' }}>|</span>
        <span style={{ color: '#888', fontSize: '12px' }}>
          See your code compile to GPU hardware
        </span>

        <div style={{ flex: 1 }} />

        {/* Example selector */}
        <select
          value={selectedExample}
          onChange={(e) => handleExampleChange(parseInt(e.target.value))}
          style={{
            padding: '4px 8px',
            fontSize: '12px',
            background: '#1a1a2e',
            color: '#e0e0e0',
            border: '1px solid #333',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          {EXAMPLES.map((ex, i) => (
            <option key={i} value={i}>
              {ex.name} {'\u2014'} {ex.description}
            </option>
          ))}
        </select>

        <a
          href="https://github.com/gautam1858/tiny-gpu-compiler"
          target="_blank"
          rel="noopener"
          style={{
            color: '#888',
            fontSize: '12px',
            textDecoration: 'none',
          }}
        >
          GitHub {'\u2197'}
        </a>
      </header>

      {/* Main layout */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left: Editor */}
        <div
          style={{
            width: '25%',
            minWidth: '240px',
            display: 'flex',
            flexDirection: 'column',
            borderRight: '1px solid #222',
          }}
        >
          <div
            style={{
              padding: '6px 12px',
              background: '#111',
              fontSize: '11px',
              color: '#888',
              borderBottom: '1px solid #222',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <span style={{ color: '#e06c75' }}>{'\u25CF'}</span> Source Code (.tgc)
          </div>
          <div style={{ flex: 1 }}>
            <Editor value={source} onChange={handleSourceChange} />
          </div>
        </div>

        {/* Center: Pipeline + Binary */}
        <div
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            borderRight: '1px solid #222',
          }}
        >
          {/* Pipeline View */}
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <div
              style={{
                padding: '6px 12px',
                background: '#111',
                fontSize: '11px',
                color: '#888',
                borderBottom: '1px solid #222',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              <span style={{ color: '#c586c0' }}>{'\u25CF'}</span> Compilation Pipeline
              {trace.analysis?.metrics && (
                <span style={{ marginLeft: 'auto', fontSize: '10px', color: '#4ec9b0' }}>
                  {trace.analysis.metrics.totalInstructions} inst | {trace.analysis.metrics.registersUsed} regs
                  {trace.analysis.metrics.sharedMemoryBytes > 0 && ` | ${trace.analysis.metrics.sharedMemoryBytes}B shared`}
                </span>
              )}
            </div>
            <div style={{ height: 'calc(100% - 28px)', overflow: 'hidden' }}>
              <PipelineView stages={trace.stages} source={source} />
            </div>
          </div>

          {/* Binary View */}
          <div
            style={{
              height: '40%',
              borderTop: '1px solid #222',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <div
              style={{
                padding: '6px 12px',
                background: '#111',
                fontSize: '11px',
                color: '#888',
                borderBottom: '1px solid #222',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                flexShrink: 0,
              }}
            >
              <span style={{ color: '#b5cea8' }}>{'\u25CF'}</span> 16-bit Binary
              <span style={{ marginLeft: 'auto', fontSize: '10px' }}>
                {trace.binary.instructions.length} instructions
              </span>
              <span style={{ fontSize: '10px', color: '#555' }}>
                <span style={{ color: '#e06c75' }}>{'\u25A0'}</span> opcode{' '}
                <span style={{ color: '#61afef' }}>{'\u25A0'}</span> rd{' '}
                <span style={{ color: '#98c379' }}>{'\u25A0'}</span> rs{' '}
                <span style={{ color: '#d19a66' }}>{'\u25A0'}</span> rt
              </span>
            </div>
            <div style={{ flex: 1, overflow: 'auto' }}>
              <BinaryView
                instructions={trace.binary.instructions}
                highlightAddr={highlightAddr}
              />
            </div>
          </div>
        </div>

        {/* Right: GPU Simulator / Analysis */}
        <div
          style={{
            width: '28%',
            minWidth: '250px',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Tab selector */}
          <div
            style={{
              padding: '6px 12px',
              background: '#111',
              fontSize: '11px',
              color: '#888',
              borderBottom: '1px solid #222',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <button
              onClick={() => setRightPanel('simulator')}
              style={{
                padding: '3px 10px',
                fontSize: '11px',
                background: rightPanel === 'simulator' ? '#2d5a3d' : 'transparent',
                color: rightPanel === 'simulator' ? '#7fff7f' : '#888',
                border: rightPanel === 'simulator' ? '1px solid #4a8a5a' : '1px solid transparent',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              <span style={{ color: '#61afef' }}>{'\u25CF'}</span> GPU Execution
            </button>
            <button
              onClick={() => setRightPanel('analysis')}
              style={{
                padding: '3px 10px',
                fontSize: '11px',
                background: rightPanel === 'analysis' ? '#2d5a3d' : 'transparent',
                color: rightPanel === 'analysis' ? '#7fff7f' : '#888',
                border: rightPanel === 'analysis' ? '1px solid #4a8a5a' : '1px solid transparent',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              <span style={{ color: '#e5c07b' }}>{'\u25CF'}</span> Analysis
            </button>
          </div>
          <div style={{ flex: 1, overflow: 'hidden' }}>
            {rightPanel === 'simulator' ? (
              <GPUSimulator
                instructions={trace.binary.instructions}
                initialMemory={example.initialMemory}
                numBlocks={example.numBlocks}
                threadsPerBlock={example.threadsPerBlock}
                onCycleChange={handleCycleChange}
              />
            ) : (
              <AnalysisPanel analysis={trace.analysis} />
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer
        style={{
          padding: '4px 16px',
          background: '#111',
          borderTop: '1px solid #222',
          fontSize: '10px',
          color: '#555',
          display: 'flex',
          justifyContent: 'space-between',
          flexShrink: 0,
        }}
      >
        <span>
          Built with MLIR + React | Targeting{' '}
          <a
            href="https://github.com/adam-maj/tiny-gpu"
            target="_blank"
            rel="noopener"
            style={{ color: '#4ec9b0', textDecoration: 'none' }}
          >
            tiny-gpu
          </a>{' '}
          hardware | Shared memory + barriers + divergence analysis
        </span>
        <span>
          8-bit data | 16-bit instructions | {trace.binary.instructions.length} ops compiled
        </span>
      </footer>
    </div>
  );
}
