"""Train PPO to play arcade Galaga.

Training runs on galaga.c, a headless Galaga-only emulator running the original ROMs.
MAME is used to watch the result and to check it against the reference emulator.

  .venv/bin/python train.py          train (resumes from galaga_ppo.pt if present)
  .venv/bin/python train.py watch    watch the saved agent play in MAME at normal speed
  .venv/bin/python train.py eval     score the saved agent over 10 games in MAME and in galaga.c
  .venv/bin/python train.py check    self-check: random play in both emulators
"""
import collections
import concurrent.futures
import ctypes
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

HERE = pathlib.Path(__file__).parent
MODEL = HERE / "galaga_ppo.pt"
HIT_REWARD = 2.0     # per enemy hit
DEATH_PENALTY = 7.0  # per ship lost
CLEAR_REWARD = 20.0  # per stage cleared
STEP_PENALTY = 0.02  # per step (4 frames) of play, so slow stages cost more: about -0.3 per second
MISS_PENALTY = 0.5   # per shot that hits nothing: charged when fired, refunded on a hit

# PPO settings (Stable-Baselines3 defaults unless noted)
N_ENVS = max(256, 32 * os.cpu_count() // 64 * 64)  # games played in parallel: 256 here, more on bigger CPUs
N_STEPS = 128        # steps per game per update
BATCH = 4096
EPOCHS = 4
LR, GAMMA, LAMBDA, CLIP, ENT_COEF, VF_COEF, MAX_GRAD = 3e-4, 0.99, 0.95, 0.2, 0.01, 0.5, 0.5
CHECKPOINT_EVERY = 5_000_000


class Galaga(gym.Env):
    """One game in MAME.
    Obs: 384 RAM bytes (object states, sprite positions, sprite control), scaled to 0..1.
    Actions: 0 noop, 1 left, 2 right, 3 fire, 4 left+fire, 5 right+fire.
    Reward: +HIT_REWARD per enemy hit, +CLEAR_REWARD per stage cleared, -DEATH_PENALTY per ship lost,
    -MISS_PENALTY per missed shot, -STEP_PENALTY every step.
    Episode ends at game over."""

    observation_space = gym.spaces.Box(0.0, 1.0, (384,), np.float32)
    action_space = gym.spaces.Discrete(6)

    def __init__(self, watch=False):
        cmd = ["mame", "galaga", "-rompath", "roms", "-skip_gameinfo", "-sound", "none",
               "-autoboot_script", "bridge.lua"]
        cmd += ["-window"] if watch else ["-video", "none", "-nothrottle"]
        self.proc = subprocess.Popen(cmd, cwd=HERE, text=True, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._read()  # bridge sends one state line as soon as the machine is up

    def _read(self):
        for line in self.proc.stdout:
            if line.startswith("@"):
                _, state, ships, stage, hits, shots, ram = line.split()
                self.info = {"state": int(state), "ships": int(ships), "stage": int(stage), "hits": int(hits),
                             "shots": int(shots)}
                return np.frombuffer(bytes.fromhex(ram), np.uint8).astype(np.float32) / 255.0
        raise RuntimeError("MAME exited")

    def _send(self, msg):
        self.proc.stdin.write(f"{msg}\n")
        self.proc.stdin.flush()
        return self._read()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._send("r"), self.info

    def step(self, action):
        before = self.info
        obs = self._send(int(action))
        now = self.info
        # _b_nships is the reserve count: it drops when the first ship deploys (stage still 0,
        # not a death), then once per ship lost, wrapping 0 -> 255 when the last one goes.
        # It goes up when a bonus ship is awarded, which is not a death.
        died = (before["ships"] - now["ships"]) % 256 == 1 and now["stage"] > 0
        done = now["state"] != 3 or now["ships"] == 255
        cleared = before["stage"] > 0 and now["stage"] > before["stage"]  # 0 -> 1 is the game starting, not a clear
        hit, fired = max(now["hits"] - before["hits"], 0), max(now["shots"] - before["shots"], 0)
        # a miss is only known later, so every shot is charged when fired and refunded if it hits
        reward = ((HIT_REWARD + MISS_PENALTY) * hit - MISS_PENALTY * fired + CLEAR_REWARD * cleared
                  - DEATH_PENALTY * died - STEP_PENALTY)
        return obs, float(reward), done, False, now

    def close(self):
        self.proc.kill()


class FastGalaga:
    """n_envs games in galaga.c, stepped together across all cores. Same observations,
    actions and rewards as Galaga. Finished games restart on their own."""

    def __init__(self, n_envs, seed=0):
        src = HERE / "galaga.c"
        lib = HERE / {"win32": "galaga.dll", "darwin": "libgalaga.dylib"}.get(sys.platform, "libgalaga.so")
        if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
            if sys.platform == "win32" and shutil.which("cl"):  # MSVC, from a Developer Command Prompt
                cmd = ["cl", "/nologo", "/O2", "/LD", str(src), f"/Fe:{lib}", f"/Fo:{HERE}\\"]
            else:  # clang, Apple's cc, or MinGW gcc on Windows
                cmd = [next(c for c in ("cc", "gcc", "clang") if shutil.which(c)), "-O3", "-shared", "-fPIC", "-o", str(lib), str(src)]
            subprocess.run(cmd, check=True)
        self.lib = ctypes.CDLL(str(lib))
        self.lib.gx_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        self.lib.gx_set_rewards.argtypes = [ctypes.c_float] * 5
        code = self.lib.gx_init(str(HERE / "roms" / "galaga").encode(), n_envs, 64, seed)
        if code:
            raise RuntimeError(f"galaga.c failed to start (code {code}): are the ROMs in roms/galaga?")
        self.lib.gx_set_rewards(HIT_REWARD, DEATH_PENALTY, CLEAR_REWARD, STEP_PENALTY, MISS_PENALTY)
        self.obs = np.zeros((n_envs, 384), np.float32)
        self.rew = np.zeros(n_envs, np.float32)
        self.done = np.zeros(n_envs, np.uint8)
        self.info = np.zeros((n_envs, 2), np.int32)  # hits, stage (final values when a game just ended)
        self.act = np.zeros(n_envs, np.uint8)
        self.args = tuple(self._p(a) for a in (self.act, self.obs, self.rew, self.done, self.info))
        # One thread per logical CPU, each stepping a slice of the games (ctypes releases the GIL).
        self.pool = concurrent.futures.ThreadPoolExecutor(os.cpu_count())

    @staticmethod
    def _p(a):
        return a.ctypes.data_as(ctypes.c_void_p)

    def reset(self):
        self.lib.gx_reset(self._p(self.obs))
        return self.obs.copy()

    def step(self, actions, lo=0, hi=None):
        """Steps games lo..hi-1 (default: all) and returns their obs, reward, done, info, with
        info[k] = (hits, stage). Separate threads may step non-overlapping ranges at the same time."""
        hi = len(self.act) if hi is None else hi
        self.act[lo:hi] = actions
        cuts = np.linspace(lo, hi, min(hi - lo, os.cpu_count()) + 1).astype(int)
        list(self.pool.map(lambda a, b: self.lib.gx_step_range(int(a), int(b), *self.args), cuts[:-1], cuts[1:]))
        return self.obs[lo:hi].copy(), self.rew[lo:hi].copy(), self.done[lo:hi].astype(bool), self.info[lo:hi].copy()


class Policy(nn.Module):
    """Separate policy and value MLPs, 384 -> 256 -> 256, tanh, orthogonal init (as SB3's MlpPolicy)."""

    def __init__(self):
        super().__init__()

        def mlp(out, gain):
            layers = [nn.Linear(384, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, out)]
            for layer, g in zip(layers[::2], (2 ** 0.5, 2 ** 0.5, gain)):
                nn.init.orthogonal_(layer.weight, g)
                nn.init.zeros_(layer.bias)
            return nn.Sequential(*layers)

        self.pi, self.v = mlp(6, 0.01), mlp(1, 1.0)

    def act(self, obs):
        """Sampled action(s) for one observation or a batch, as numpy."""
        with torch.no_grad():
            return Categorical(logits=self.pi(torch.as_tensor(obs))).sample().numpy()


def load_policy():
    if not MODEL.exists():
        sys.exit(f"no {MODEL.name} yet: train first with  .venv/bin/python train.py")
    net = Policy()
    net.load_state_dict(torch.load(MODEL)["net"])
    return net


def update(net, opt, b, dev, stats):
    """One PPO update on rollout b. Runs on the learner thread while the next rollout is collected,
    so b was played by the weights from one update earlier; the clipped ratio absorbs that lag."""
    T, N = b["rew"].shape
    obs = torch.from_numpy(b["obs"]).to(dev).reshape(T * N, 384)
    with torch.no_grad():  # GAE with the learner's current value estimates
        values = net.v(obs).reshape(T, N).cpu().numpy()
        next_v = net.v(torch.from_numpy(b["last_obs"]).to(dev)).squeeze(-1).cpu().numpy()
    adv = np.zeros((T, N), np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        alive = 1.0 - b["done"][t]
        delta = b["rew"][t] + GAMMA * next_v * alive - values[t]
        gae = delta + GAMMA * LAMBDA * alive * gae
        adv[t], next_v = gae, values[t]
    ret = torch.from_numpy(adv + values).to(dev).flatten()
    adv = torch.from_numpy(adv).to(dev).flatten()
    act = torch.from_numpy(b["act"]).to(dev).flatten()
    old_logp = torch.from_numpy(b["logp"]).to(dev).flatten()
    for _ in range(EPOCHS):
        for idx in torch.randperm(T * N, device=dev).split(BATCH):
            dist = Categorical(logits=net.pi(obs[idx]))
            logp = dist.log_prob(act[idx])
            ratio = torch.exp(logp - old_logp[idx])
            a = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)
            pg_loss = -torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a).mean()
            v_loss = ((net.v(obs[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
            entropy = dist.entropy().mean()
            loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD)
            opt.step()
    stats.update(pg_loss=pg_loss.item(), v_loss=v_loss.item(), entropy=entropy.item(),
                 kl=(old_logp[idx] - logp).mean().item())


def train():
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"learner on {dev}, games on {os.cpu_count()} CPU threads")
    net = Policy().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)
    steps = 0
    if MODEL.exists():
        ck = torch.load(MODEL, map_location=dev)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        steps = ck["steps"]
        print(f"resuming from {MODEL.name} at {steps:,} steps")
    actor = Policy()  # CPU copy that plays while the learner updates net
    actor.load_state_dict(net.state_dict())
    env = FastGalaga(N_ENVS)
    obs = env.reset()
    ep_ret, ep_len = np.zeros(N_ENVS), np.zeros(N_ENVS, int)
    finished = collections.deque(maxlen=200)  # (return, length, hits, stage) of recent games
    stats, learner = {}, None
    t0, steps0, last_log, next_ckpt = time.time(), steps, 0.0, (steps // CHECKPOINT_EVERY + 1) * CHECKPOINT_EVERY
    (HERE / "checkpoints").mkdir(exist_ok=True)

    def save(path):
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "steps": steps}, path)

    try:
        while True:
            b = {"obs": np.empty((N_STEPS, N_ENVS, 384), np.float32), "act": np.empty((N_STEPS, N_ENVS), np.int64),
                 "logp": np.empty((N_STEPS, N_ENVS), np.float32), "rew": np.empty((N_STEPS, N_ENVS), np.float32),
                 "done": np.empty((N_STEPS, N_ENVS), np.float32)}

            def collect(lo, hi):  # plays games lo..hi-1 for N_STEPS steps into b
                with torch.no_grad():
                    for t in range(N_STEPS):
                        b["obs"][t, lo:hi] = obs[lo:hi]
                        dist = Categorical(logits=actor.pi(torch.from_numpy(obs[lo:hi])))
                        a = dist.sample()
                        b["act"][t, lo:hi], b["logp"][t, lo:hi] = a.numpy(), dist.log_prob(a).numpy()
                        obs[lo:hi], b["rew"][t, lo:hi], done, info = env.step(b["act"][t, lo:hi], lo, hi)
                        b["done"][t, lo:hi] = done
                        ep_ret[lo:hi] += b["rew"][t, lo:hi]
                        ep_len[lo:hi] += 1
                        for k in np.flatnonzero(done):
                            finished.append((ep_ret[lo + k], ep_len[lo + k], *info[k]))
                            ep_ret[lo + k], ep_len[lo + k] = 0, 0

            # Two halves of the games, each on its own thread: while one half is in the emulator,
            # the other is choosing its next actions, so the CPU cores never wait on Python.
            halves = [threading.Thread(target=collect, args=(lo, hi)) for lo, hi in ((0, N_ENVS // 2), (N_ENVS // 2, N_ENVS))]
            for h in halves:
                h.start()
            for h in halves:
                h.join()
            b["last_obs"] = obs.copy()
            if learner:
                learner.join()
                actor.load_state_dict(net.state_dict())
            steps += N_STEPS * N_ENVS
            learner = threading.Thread(target=update, args=(net, opt, b, dev, stats))
            learner.start()

            if time.time() - last_log > 10 and finished and stats:
                last_log = time.time()
                r, l, hits, stage = np.mean(finished, axis=0)
                rate = (steps - steps0) / (time.time() - t0)
                print(f"{steps:>12,} steps  {rate:6,.0f}/s ({rate * 4 / 60.6:,.0f}x real time)  "
                      f"last {len(finished)} games: return {r:6.1f}  steps {l:5.0f}  hits {hits:5.1f}  stage {stage:4.2f}  "
                      f"| entropy {stats['entropy']:.2f}  kl {stats['kl']:.4f}  v_loss {stats['v_loss']:.2f}", flush=True)
            if steps >= next_ckpt:
                learner.join()
                save(HERE / "checkpoints" / f"galaga_{steps}.pt")
                next_ckpt += CHECKPOINT_EVERY
    finally:  # Ctrl-C saves too
        if learner:
            learner.join()
        save(MODEL)
        print(f"saved {MODEL.name} at {steps:,} steps")


def play_mame(policy, games):
    """Returns (hits, stage, steps) per game played in MAME."""
    env, out = Galaga(), []
    for _ in range(games):
        obs, _ = env.reset()
        done, steps = False, 0
        while not done:
            obs, _, done, _, info = env.step(policy(obs))
            steps += 1
        out.append((info["hits"], info["stage"], steps))
    env.close()
    return np.array(out)


def play_fast(policy, games, n_envs=16):
    """Returns (hits, stage, steps) per game played in galaga.c. policy takes a batch of observations."""
    env, out = FastGalaga(n_envs, seed=1), []
    obs, steps = env.reset(), np.zeros(n_envs, int)
    while len(out) < games:
        obs, _, done, info = env.step(policy(obs))
        steps += 1
        for k in np.flatnonzero(done):
            out.append((*info[k], steps[k]))
            steps[k] = 0
    return np.array(out[:games])


def report(name, results):
    hits, stage, steps = results.mean(0)
    print(f"{name:9s} {len(results)} games: mean hits {hits:.1f}, mean stage {stage:.2f}, mean steps {steps:.0f}")
    return results.mean(0)


def check():
    rng = np.random.default_rng(0)
    mame = report("MAME", play_mame(lambda obs: rng.integers(6), 6))
    fast = report("galaga.c", play_fast(lambda obs: rng.integers(6, size=len(obs)), 64))
    assert mame[0] > 0, "random play never hit anything in MAME: inputs are not reaching the game"
    assert fast[0] > 0, "random play never hit anything in galaga.c"
    for name, a, b in zip(("hits", "stage", "steps"), mame, fast):
        assert abs(a - b) < 0.35 * max(a, b), f"random-play {name} differ between emulators: MAME {a:.1f}, galaga.c {b:.1f}"
    print("ok: both emulators run and random play scores alike in each")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "check":
        check()
    elif mode == "watch":
        net, env = load_policy(), Galaga(watch=True)
        obs, _ = env.reset()
        while True:
            obs, _, done, _, _ = env.step(net.act(obs))
            if done:
                obs, _ = env.reset()
    elif mode == "eval":
        net = load_policy()
        report("MAME", play_mame(net.act, 10))
        report("galaga.c", play_fast(net.act, 64))
    else:
        try:
            train()
        except KeyboardInterrupt:  # the save already happened in train()
            pass
