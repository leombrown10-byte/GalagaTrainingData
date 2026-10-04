-- MAME <-> Python bridge. Each step: one "@ ..." state line out on stdout,
-- one command line in on stdin ("r" = start a new game, "0".."5" = action).
-- RAM addresses come from rom0/mrw.s and rom0/structs.inc in hackbar/galaga.
local m = manager.machine
local mem = m.devices[":maincpu"].spaces["program"]
local SKIP = 4 -- emulated frames per agent step

local f = {}
for _, port in pairs(m.ioport.ports) do
  for name, field in pairs(port.fields) do f[name] = field end
end
local function press(name, on) f[name]:set_value(on and 1 or 0) end

local function hex(from, len)
  local t = {}
  for a = from, from + len - 1 do t[#t + 1] = string.format("%02x", mem:read_u8(a)) end
  return table.concat(t)
end

local function playing() return mem:read_u8(0x9201) == 3 end -- b8_9201_game_state: 3 = PLAY_MODE

local mode, n, act, idle = "step", SKIP, 0, false

local function exchange()
  io.stdout:write(string.format("@ %d %d %d %d %d %s%s%s\n",
    mem:read_u8(0x9201),                             -- game state
    mem:read_u8(0x9820),                             -- ds_plyr_actv._b_nships (reserve ships)
    mem:read_u8(0x9821),                             -- ds_plyr_actv._b_stgctr
    mem:read_u8(0x9844) + 256 * mem:read_u8(0x9845), -- ds_plyr_actv._w_hit_ct
    mem:read_u8(0x9846) + 256 * mem:read_u8(0x9847), -- ds_plyr_actv._w_shot_ct
    hex(0x8800, 0x80),                               -- object state, 2 bytes x 64 objects
    hex(0x9300, 0x80),                               -- sprite positions (sx, sy low)
    hex(0x9B00, 0x80)))                              -- sprite control (flip, sy bit 8, enable)
  io.stdout:flush()
  local line = io.read("l")
  if not line then m:exit() return end -- Python went away
  n = 0
  if line == "r" then mode, idle = "reset", false else mode, act = "step", tonumber(line) end
end

-- Runs at each vblank. The inputs set here are what the game sees during the
-- frame that follows, so a new action takes effect in the very next frame.
local function on_frame()
  n = n + 1
  if mode == "reset" then
    -- ponytail: waits out the old game (results screen, or the ship dying untouched)
    -- before starting a new one; use a save-state reload if resets get slow.
    idle = idle or not playing()
    local t = n % 120
    press("P1 Left", false) press("P1 Right", false) press("P1 Button 1", false)
    press("Coin 1", idle and t < 6)
    press("1 Player Start", idle and t >= 60 and t < 66)
    if not (idle and playing()) then return end
    press("Coin 1", false) press("1 Player Start", false)
    exchange()
  elseif n >= SKIP then
    exchange()
  end
  if mode ~= "step" then return end
  press("P1 Left", act == 1 or act == 4)
  press("P1 Right", act == 2 or act == 5)
  press("P1 Button 1", act >= 3 and n < 2) -- 2 frames down, 2 up: the 51xx chip needs both to see a fresh press every step
end

-- keep the subscription referenced or MAME drops the callback
bridge_sub = emu.add_machine_frame_notifier(on_frame)
