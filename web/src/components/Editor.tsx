import { useRef, useEffect } from 'react';
import MonacoEditor, { OnMount } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';

interface EditorProps {
  value: string;
  onChange: (value: string) => void;
}

export function Editor({ value, onChange }: EditorProps) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;

    // Register .tgc language
    monaco.languages.register({ id: 'tgc' });

    // Syntax highlighting rules
    monaco.languages.setMonarchTokensProvider('tgc', {
      keywords: ['kernel', 'global', 'int', 'for', 'if', 'else'],
      builtins: ['threadIdx', 'blockIdx', 'blockDim'],
      operators: ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>='],
      tokenizer: {
        root: [
          [/\/\/.*$/, 'comment'],
          [/\/\*/, 'comment', '@comment'],
          [/\b(kernel|global|int|for|if|else)\b/, 'keyword'],
          [/\b(threadIdx|blockIdx|blockDim)\b/, 'variable.predefined'],
          [/\b\d+\b/, 'number'],
          [/[a-zA-Z_]\w*/, 'identifier'],
          [/[{}()\[\]]/, 'delimiter.bracket'],
          [/[;,]/, 'delimiter'],
          [/[+\-*/=<>!]+/, 'operator'],
        ],
        comment: [
          [/[^/*]+/, 'comment'],
          [/\*\//, 'comment', '@pop'],
          [/[/*]/, 'comment'],
        ],
      },
    });

    // Theme
    monaco.editor.defineTheme('tgc-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'keyword', foreground: 'C586C0' },
        { token: 'variable.predefined', foreground: '4EC9B0', fontStyle: 'bold' },
        { token: 'number', foreground: 'B5CEA8' },
        { token: 'comment', foreground: '6A9955' },
        { token: 'operator', foreground: 'D4D4D4' },
        { token: 'identifier', foreground: '9CDCFE' },
      ],
      colors: {
        'editor.background': '#1a1a2e',
        'editor.foreground': '#e0e0e0',
      },
    });

    monaco.editor.setTheme('tgc-dark');
  };

  useEffect(() => {
    if (editorRef.current) {
      const model = editorRef.current.getModel();
      if (model && model.getValue() !== value) {
        model.setValue(value);
      }
    }
  }, [value]);

  return (
    <div style={{ height: '100%', border: '1px solid #333' }}>
      <MonacoEditor
        defaultLanguage="tgc"
        theme="tgc-dark"
        value={value}
        onChange={(v) => onChange(v ?? '')}
        onMount={handleMount}
        options={{
          fontSize: 14,
          lineNumbers: 'on',
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          padding: { top: 8 },
          automaticLayout: true,
          tabSize: 4,
        }}
      />
    </div>
  );
}
