# tiny-gpu-compiler Onboarding

This folder is a fast-start wiki for understanding the repository as code, not just as a project README.

## Suggested reading order

1. [00-overview.md](./00-overview.md)
2. [01-reading-order.md](./01-reading-order.md)
3. [02-native-compiler-architecture.md](./02-native-compiler-architecture.md)
4. [03-web-stack.md](./03-web-stack.md)
5. [04-build-test-debug.md](./04-build-test-debug.md)
6. [05-gotchas-and-mental-models.md](./05-gotchas-and-mental-models.md)
7. [06-where-to-change-what.md](./06-where-to-change-what.md)

## What this is for

Use these docs when you want to answer questions like:

- Where is the real compiler entry point?
- Which parts are native C++/MLIR versus browser-side TypeScript?
- What contract connects the TinyGPU dialect to the final ISA?
- If I change the DSL, optimizer, emitter, or simulator, which files move together?

## Core idea

The repo is easiest to understand as two coupled systems:

1. A native C++/MLIR compiler targeting the tiny-gpu ISA.
2. A web visualizer with its own in-browser compiler and simulator for teaching and exploration.

They share concepts, but they are not the same implementation.
