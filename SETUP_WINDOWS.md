# Galaga training on Windows

This folder trains a PPO agent to play arcade Galaga. `train.py` runs the games in
`galaga.c` (a headless emulator of the original ROMs in `roms/galaga/`) and learns on the
GPU. `galaga_ppo.pt` is the model trained so far; training resumes from it.

## Setup

1. Install Python 3.12 (python.org) and a C compiler, one of:
   - Visual Studio Build Tools with "Desktop development with C++"; then run everything
     from a "Developer Command Prompt" so `cl` is on PATH
   - MinGW-w64, so `gcc` is on PATH
2. In this folder:
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\pip install numpy gymnasium
   .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```
   Use the PyTorch CUDA build that matches the GPU driver (see pytorch.org). Check it with
   `.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"`, which should print True.
3. Train (Ctrl-C saves; rerunning resumes):
   ```
   .venv\Scripts\python train.py
   ```
   The first run builds `galaga.dll` from `galaga.c`. The startup line should say
   `learner on cuda` and the number of CPU threads.

## Notes

- Hardware use is automatic: the games run on one thread per logical CPU, the number of games
  scales with the CPU (`N_ENVS`), and learning runs on the GPU at the same time as the games.
  Expect every CPU thread busy in Task Manager during training.
- `watch`, `eval` and `check` need MAME (`mame` on PATH, from mamedev.org); plain training does not.
- The Windows build (`cl`/`gcc` in `FastGalaga`, `__declspec(dllexport)` in `galaga.c`) was written
  on a Mac and has not run on Windows yet. If the build fails, fix it there.
- The ROMs are Namco's copyrighted code: keep them off public repositories.
