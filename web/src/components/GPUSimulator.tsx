import { useState, useEffect, useCallback, useRef } from 'react';
import { TinyGPUSim } from '../simulator/TinyGPUSim';
import { Instruction, SimulationState, PipelineStage } from '../compiler/types';

interface GPUSimulatorProps {
  instructions: Instruction[];
  initialMemory: number[];
  numBlocks: number;
  threadsPerBlock: number;
  onCycleChange?: (state: SimulationState) => void;
}

const STAGE_COLORS: Record<string, string> = {
  [PipelineStage.FETCH]: '#e06c75',
  [PipelineStage.DECODE]: '#d19a66',
  [PipelineStage.REQUEST]: '#e5c07b',
  [PipelineStage.WAIT]: '#98c379',
  [PipelineStage.EXECUTE]: '#61afef',
  [PipelineStage.UPDATE]: '#c678dd',
  [PipelineStage.BARRIER]: '#4ec9b0',
  [PipelineStage.DONE]: '#555',
};

export function GPUSimulator({
  instructions,
  initialMemory,
  numBlocks,
  threadsPerBlock,
  onCycleChange,
}: GPUSimulatorProps) {
  const [history, setHistory] = useState<SimulationState[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [selectedThread, setSelectedThread] = useState<number | null>(null);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (instructions.length === 0) return;
    const sim = new TinyGPUSim(instructions, initialMemory, numBlocks, threadsPerBlock);
    const allStates = sim.runToEnd(5000);
    setHistory(allStates);
    setCurrentStep(0);
    setIsPlaying(false);
    setSelectedThread(null);
  }, [instructions, initialMemory, numBlocks, threadsPerBlock]);

  const state = history[currentStep];

  useEffect(() => {
    if (state && onCycleChange) onCycleChange(state);
  }, [currentStep, state, onCycleChange]);

  useEffect(() => {
    if (isPlaying && history.length > 0) {
      intervalRef.current = window.setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= history.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, speed, history.length]);

  const stepForward = useCallback(() => {
    setCurrentStep((prev) => Math.min(prev + 1, history.length - 1));
  }, [history.length]);

  const stepBackward = useCallback(() => {
    setCurrentStep((prev) => Math.max(prev - 1, 0));
  }, []);

  const reset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  const jumpToEnd = useCallback(() => {
    setCurrentStep(history.length - 1);
    setIsPlaying(false);
  }, [history.length]);

  if (!state || instructions.length === 0) {
    return (
      <div style={{ padding: '16px', color: '#666', fontSize: '13px' }}>
        Compile a kernel to start the GPU simulator.
      </div>
    );
  }

  const selectedThreadState = selectedThread !== null
    ? state.threads.find(t => t.threadId === selectedThread)
    : null;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {/* Controls */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '8px',
          background: '#111',
          borderRadius: '4px',
          flexShrink: 0,
          flexWrap: 'wrap',
        }}
      >
        <button onClick={reset} style={btnStyle} title="Reset">
          {'\u23EE'}
        </button>
        <button onClick={stepBackward} style={btnStyle} title="Step Back">
          {'\u23EA'}
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          style={{ ...btnStyle, background: isPlaying ? '#e06c75' : '#2d5a3d', width: '48px' }}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? '\u23F8' : '\u25B6'}
        </button>
        <button onClick={stepForward} style={btnStyle} title="Step Forward">
          {'\u23E9'}
        </button>
        <button onClick={jumpToEnd} style={btnStyle} title="Jump to End">
          {'\u23ED'}
        </button>

        <div style={{ flex: 1 }} />

        <label style={{ fontSize: '11px', color: '#888' }}>
          Speed:
          <input
            type="range"
            min={50}
            max={1000}
            step={50}
            value={1050 - speed}
            onChange={(e) => setSpeed(1050 - parseInt(e.target.value))}
            style={{ width: '60px', marginLeft: '4px', verticalAlign: 'middle' }}
          />
        </label>

        <span style={{ fontSize: '11px', color: '#4ec9b0', fontFamily: 'monospace' }}>
          Cycle {state.cycle} / {history.length - 1}
        </span>
      </div>

      {/* Scrubber */}
      <input
        type="range"
        min={0}
        max={history.length - 1}
        value={currentStep}
        onChange={(e) => {
          setCurrentStep(parseInt(e.target.value));
          setIsPlaying(false);
        }}
        style={{ width: '100%', flexShrink: 0 }}
      />

      {/* Block info + divergence indicator */}
      <div style={{ fontSize: '11px', color: '#888', padding: '0 4px', flexShrink: 0, display: 'flex', gap: '8px', alignItems: 'center' }}>
        <span>Block {state.currentBlock} / {state.totalBlocks}</span>
        {state.currentBlock >= state.totalBlocks && (
          <span style={{ color: '#4ec9b0' }}>DONE</span>
        )}
        {state.threads.some(t => t.divergent) && (
          <span style={{
            color: '#e06c75',
            padding: '1px 6px',
            background: '#3a2020',
            borderRadius: '3px',
            fontSize: '9px',
            fontWeight: 700,
          }}>
            DIVERGENT
          </span>
        )}
        {state.threads.some(t => t.stage === PipelineStage.BARRIER) && (
          <span style={{
            color: '#4ec9b0',
            padding: '1px 6px',
            background: '#1a3a3a',
            borderRadius: '3px',
            fontSize: '9px',
            fontWeight: 700,
          }}>
            BARRIER
          </span>
        )}
      </div>

      {/* Thread grid */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${Math.min(threadsPerBlock, 4)}, 1fr)`,
            gap: '6px',
            padding: '4px',
          }}
        >
          {state.threads.map((thread) => (
            <ThreadCard
              key={`${thread.blockId}-${thread.threadId}`}
              thread={thread}
              selected={selectedThread === thread.threadId}
              onClick={() => setSelectedThread(selectedThread === thread.threadId ? null : thread.threadId)}
            />
          ))}
        </div>

        {/* Selected thread detail (debugger view) */}
        {selectedThreadState && (
          <div style={{ margin: '8px 4px', padding: '8px', background: '#1a1a2e', border: '1px solid #4ec9b0', borderRadius: '6px' }}>
            <div style={{ fontSize: '11px', color: '#4ec9b0', fontWeight: 700, marginBottom: '6px' }}>
              Thread {selectedThreadState.threadId} Register File
            </div>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: '4px',
              fontSize: '10px',
              fontFamily: 'monospace',
            }}>
              {selectedThreadState.registers.map((val, r) => (
                <div key={r} style={{
                  padding: '3px 4px',
                  background: r === 13 || r === 14 || r === 15 ? '#2a2a4a' : val !== 0 ? '#1a2a1a' : '#111',
                  borderRadius: '3px',
                  color: r >= 13 ? '#c586c0' : val !== 0 ? '#98c379' : '#444',
                }}>
                  <span style={{ color: '#888' }}>
                    {r === 13 ? 'BID' : r === 14 ? 'BDM' : r === 15 ? 'TID' : `R${r}`}
                  </span>
                  {' '}{val}
                </div>
              ))}
            </div>
            <div style={{ marginTop: '4px', fontSize: '10px', color: '#888' }}>
              PC: <span style={{ color: '#b5cea8' }}>{selectedThreadState.pc}</span>
              {' | '}NZP: <span style={{ color: '#d19a66' }}>
                {((selectedThreadState.nzp >> 2) & 1) ? 'N' : '-'}
                {((selectedThreadState.nzp >> 1) & 1) ? 'Z' : '-'}
                {(selectedThreadState.nzp & 1) ? 'P' : '-'}
              </span>
              {selectedThreadState.currentInstruction && (
                <>
                  {' | '}<span style={{ color: '#c586c0' }}>{selectedThreadState.currentInstruction}</span>
                </>
              )}
            </div>
          </div>
        )}

        {/* Memory visualization */}
        <div style={{ marginTop: '12px', padding: '4px' }}>
          <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>
            Data Memory (256 bytes)
          </div>
          <MemoryHeatmap memory={state.memory} />
        </div>

        {/* Shared Memory visualization */}
        {state.sharedMemory.some(v => v !== 0) && (
          <div style={{ marginTop: '8px', padding: '4px' }}>
            <div style={{ fontSize: '11px', color: '#4ec9b0', marginBottom: '4px' }}>
              Shared Memory (64 bytes)
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1px' }}>
              {state.sharedMemory.slice(0, 32).map((val, i) => (
                <div
                  key={i}
                  style={{
                    width: '20px',
                    height: '18px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '9px',
                    fontFamily: 'monospace',
                    background: val > 0
                      ? `#4ec9b0${Math.min(Math.floor(val / 255 * 200) + 55, 255).toString(16).padStart(2, '0')}`
                      : '#1a1a2e',
                    color: val > 0 ? '#fff' : '#333',
                    borderRadius: '2px',
                    border: '1px solid #222',
                  }}
                  title={`[S${i}] = ${val}`}
                >
                  {val}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ThreadCard({
  thread,
  selected,
  onClick,
}: {
  thread: import('../compiler/types').ThreadState;
  selected: boolean;
  onClick: () => void;
}) {
  const stageColor = STAGE_COLORS[thread.stage] || '#555';

  return (
    <div
      onClick={onClick}
      style={{
        background: '#1a1a2e',
        border: `1px solid ${selected ? '#4ec9b0' : thread.divergent ? '#e06c75' : thread.done ? '#333' : stageColor}`,
        borderRadius: '6px',
        padding: '8px',
        fontSize: '11px',
        fontFamily: 'monospace',
        opacity: thread.done ? 0.5 : 1,
        transition: 'all 0.15s',
        cursor: 'pointer',
        boxShadow: selected ? '0 0 8px #4ec9b044' : thread.divergent ? '0 0 4px #e06c7533' : 'none',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ color: '#9cdcfe' }}>T{thread.threadId}</span>
        <div style={{ display: 'flex', gap: '2px' }}>
          {thread.divergent && (
            <span style={{
              background: '#3a2020',
              color: '#e06c75',
              padding: '1px 4px',
              borderRadius: '3px',
              fontSize: '8px',
              fontWeight: 700,
            }}>
              DIV
            </span>
          )}
          <span
            style={{
              background: stageColor,
              color: '#fff',
              padding: '1px 6px',
              borderRadius: '3px',
              fontSize: '9px',
              fontWeight: 700,
            }}
          >
            {thread.stage}
          </span>
        </div>
      </div>

      <div style={{ color: '#888', marginBottom: '4px' }}>
        PC: <span style={{ color: '#b5cea8' }}>{thread.pc}</span>
        {thread.currentInstruction && (
          <span style={{ color: '#c586c0', marginLeft: '8px' }}>{thread.currentInstruction}</span>
        )}
      </div>

      {/* Registers (compact) */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px' }}>
        {thread.registers.slice(0, 13).map((val, r) => (
          <span
            key={r}
            style={{
              padding: '1px 3px',
              background: val !== 0 ? '#2a2a4a' : 'transparent',
              borderRadius: '2px',
              color: val !== 0 ? '#e0e0e0' : '#444',
              fontSize: '9px',
            }}
            title={`R${r} = ${val}`}
          >
            {val !== 0 ? val : '\u00B7'}
          </span>
        ))}
      </div>
    </div>
  );
}

function MemoryHeatmap({ memory }: { memory: number[] }) {
  const regions = [
    { name: 'A (0-63)', start: 0, end: 64, color: '#e06c75' },
    { name: 'B (64-127)', start: 64, end: 128, color: '#61afef' },
    { name: 'C (128-191)', start: 128, end: 192, color: '#98c379' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      {regions.map((region) => (
        <div key={region.name}>
          <div style={{ fontSize: '10px', color: region.color, marginBottom: '2px' }}>
            {region.name}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1px' }}>
            {memory.slice(region.start, Math.min(region.end, region.start + 16)).map((val, i) => (
              <div
                key={i}
                style={{
                  width: '20px',
                  height: '18px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '9px',
                  fontFamily: 'monospace',
                  background: val > 0 ? `${region.color}${Math.min(Math.floor(val / 255 * 200) + 55, 255).toString(16).padStart(2, '0')}` : '#1a1a2e',
                  color: val > 0 ? '#fff' : '#333',
                  borderRadius: '2px',
                  border: '1px solid #222',
                }}
                title={`[${region.start + i}] = ${val}`}
              >
                {val}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: '4px 10px',
  fontSize: '14px',
  background: '#1a1a2e',
  color: '#e0e0e0',
  border: '1px solid #333',
  borderRadius: '4px',
  cursor: 'pointer',
  width: '36px',
  textAlign: 'center',
};
