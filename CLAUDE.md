# 3D Rendering App

This is an application built to render 3D scenes. It uses modern techniques to render highly realistic scenes. Techniques include things like path tracing, global illumination, ray tracing, and so on.

## Stack
- **Frontend**: Entirely built as a web app, using TypeScript, not JavaScript. The Frontend uses the Rendering Engine via WASM.
- **Rendering Engine**: The Rendering Engine is built in Rust. It is compiled into WASM for use in the Frontend.

## Code style

### Comments
Always comment code thoroughly. Assume the reader is not an expert in rendering or graphics programming. Explain:
- What a type or function does and why it exists, not just what its fields/parameters are named.
- Any non-obvious algorithm, formula, or data layout decision.
- Rendering concepts (BVH, path tracing, PBR, etc.) the first time they appear in a file.

Use `///` doc comments on all public items. Use `//` inline comments for anything that would not be immediately obvious to a competent Rust developer unfamiliar with rendering.

### Commits
Write meaningful commit messages. The subject line should complete the sentence "This commit will…". Use the body to explain *why* the change was made, not just what files were touched. Keep subject lines under 72 characters.
