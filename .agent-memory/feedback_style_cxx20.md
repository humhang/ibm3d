---
name: Feedback — coding style and C++20 use
description: How the user wants code to look in this repo. Set explicitly during the 2026-05-14 project-setup conversation.
type: feedback
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
**Rule**: format all C/C++ source with `clang-format -style=LLVM` (a
`.clang-format` file in the repo root pins this).  Target C++20.

**Why**: stated preference; the LLVM style is the user's default and
clang-format with the explicit file makes it enforceable.  C++20 is
chosen because the project is greenfield — no legacy C++17 code to
keep compatible with.

**How to apply**:

- Run formatting on touched files before committing.  Zed's
  format-on-save with the language-server formatter (clangd) reads
  `.clang-format` and does the right thing.
- Use C++20 features **when they make code clearer**: concepts to
  constrain templates, ranges for trivial transformations, `std::span`
  for non-owning array views, `[[likely]] / [[unlikely]]` in hot loops,
  designated initialisers for option structs, `consteval` / `constexpr`
  where it conveys intent.
- Do **not** reach for C++20 just to demonstrate it.  A range-based
  `for` over a `std::vector` is fine; rewriting it as a ranges pipeline
  is noise.  The test: would removing the C++20-ism make the code
  harder to read?  If no, drop the C++20-ism.
- AMReX code in this project uses `.H` for headers and `.cpp` for
  sources — match that, don't switch to `.hpp`.
