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
  const intervalRef = useRef<number | null>(null);

  // Initialize simulation when instructions change
  useEffect(() => {
    if (instructions.length === 0) return;

    const sim = new TinyGPUSim(instructions, initialMemory, numBlocks, threadsPerBlock);
    const allStates = sim.runToEnd(5000);
    setHistory(allStates);
    setCurrentStep(0);
    setIsPlaying(false);
  }, [instructions, initialMemory, numBlocks, threadsPerBlock]);

  const state = history[currentStep];

  useEffect(() => {
    if (state && onCycleChange) onCycleChange(state);
  }, [currentStep, state, onCycleChange]);

  // Playback timer
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

  if (!state || instructions.length === 0) {
    return (
      <div style={{ padding: '16px', color: '#666', fontSize: '13px' }}>
        Compile a kernel to start the GPU simulator.
      </div>
    );
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {/* Controls */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '8px',
          background: '#111',
          borderRadius: '4px',
          flexShrink: 0,
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

      {/* Block info */}
      <div style={{ fontSize: '11px', color: '#888', padding: '0 4px', flexShrink: 0 }}>
        Block {state.currentBlock} / {state.totalBlocks}
        {state.currentBlock >= state.totalBlocks && (
          <span style={{ color: '#4ec9b0', marginLeft: '8px' }}>DONE</span>
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
            <ThreadCard key={`${thread.blockId}-${thread.threadId}`} thread={thread} />
          ))}
        </div>

        {/* Memory visualization */}
        <div style={{ marginTop: '12px', padding: '4px' }}>
          <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>
            Data Memory (256 bytes)
          </div>
          <MemoryHeatmap memory={state.memory} />
        </div>
      </div>
    </div>
  );
}

function ThreadCard({ thread }: { thread: import('../compiler/types').ThreadState }) {
  const stageColor = STAGE_COLORS[thread.stage] || '#555';

  return (
    <div
      style={{
        background: '#1a1a2e',
        border: `1px solid ${thread.done ? '#333' : stageColor}`,
        borderRadius: '6px',
        padding: '8px',
        fontSize: '11px',
        fontFamily: 'monospace',
        opacity: thread.done ? 0.5 : 1,
        transition: 'all 0.15s',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ color: '#9cdcfe' }}>T{thread.threadId}</span>
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
  // Show first 192 bytes (3 regions of 64: A, B, C)
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
