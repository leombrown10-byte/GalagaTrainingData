// Headless Galaga-only machine: runs the original ROMs with three Z80s and the
// Namco 06xx/51xx custom chips modelled at the behaviour level. No video, no
// sound output. CPUs fast-forward through wait loops (see backjump()).
//
// Hardware facts (memory map, latches, interrupt timing, 51xx protocol) follow
// MAME's galaga driver; RAM addresses follow rom0/mrw.s in hackbar/galaga.
//
// Build: cc -O3 -shared -fPIC -o libgalaga.dylib galaga.c   (train.py does this itself;
//        on Windows it builds galaga.dll with cl or gcc)
// Test:  cc -O3 -DGX_MAIN -o galaga_test galaga.c && ./galaga_test roms/galaga
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;

typedef union { u16 w; struct { u8 l, h; }; } pair; // little-endian host

typedef struct {
    pair bc, de, hl, af, ix, iy, bc2, de2, hl2, af2;
    u16 pc, sp;
    u8 i, r, r7, iff1, iff2, im, halted;
    u8 irq, nmi, ei, held; // irq: line level, nmi: pending edge, ei: block irq for one instruction, held: in reset
    u8 idle;               // parked in a wait loop since m->writes == idle_w
    u16 loop_pc;           // last backward-jump target, with the registers and write count seen there
    pair loop_reg[7];
    u32 loop_w, idle_w;
    int cyc, target; // cycles within the current frame
    const u8 *rom;
} Z80;

typedef struct {
    Z80 cpu[3];
    u8 ram[0x4000]; // 0x8000-0xBFFF, shared by all three CPUs
    u32 writes;     // bumped on every memory write and I/O access, by any CPU
    u8 irq_mask[2], nmi_mask;
    u8 ctrl06;
    int nmi06_at; // main-CPU cycle of the next 06xx NMI
    int lastcoins, lastbuttons, credits, coins[2], coins_per_cred[2], creds_per_coin[2];
    int in_count, mode, coincred_mode, remap_joy;
    u8 in[4];  // 51xx ports as the chip reports them, active low nibbles: IN0L buttons/start, IN0H coins, IN1L P1 stick, IN1H P2 stick
    u8 raw[4], raw_prev[4], dly_prev[4]; // the switches themselves, this frame and earlier (see latch_inputs)
} Machine;

#define FRAME 50688 // 3.072 MHz CPU clock / 60.606 Hz
#define SLICE 512   // 99 slices per frame; MAME uses 100
#define NEVER 0x3fffffff
#define DSWA 0xF7
#define DSWB 0x97

enum { CF = 1, NF = 2, PF = 4, XF = 8, HF = 16, YF = 32, ZF = 64, SF = 128 };

static u8 g_rom[3][0x4000];
static u8 sz[256], szp[256];
static long g_unimpl; // unimplemented opcodes executed (should stay 0)

// ---------------------------------------------------------------- custom chips

static void n51_write(Machine *m, u8 d) {
    d &= 7;
    if (m->coincred_mode) {
        switch (m->coincred_mode--) {
        case 4: m->coins_per_cred[0] = d; break;
        case 3: m->creds_per_coin[0] = d; break;
        case 2: m->coins_per_cred[1] = d; break;
        case 1: m->creds_per_coin[1] = d; break;
        }
        return;
    }
    switch (d) {
    case 1: m->coincred_mode = 4; m->credits = 0; break; // set coinage
    case 2: m->mode = 1; m->in_count = 0; break;         // credits mode, start buttons live
    case 3: m->remap_joy = 0; break;
    case 4: m->remap_joy = 1; break;
    case 5: m->mode = 0; m->in_count = 0; break;         // switch mode
    }
}

static u8 n51_read(Machine *m) {
    static const u8 joy_map[16] = {0xf, 0xe, 0xd, 0x5, 0xc, 0x9, 0x7, 0x6, 0xb, 0x3, 0xa, 0x4, 0x1, 0x2, 0x0, 0x8};
    int p0 = m->in[0], p1 = m->in[1], phase = m->in_count++ % 3;
    if (m->mode == 0) return phase == 0 ? p0 | (p1 << 4) : phase == 1 ? m->in[2] | (m->in[3] << 4) : 0;
    if (phase == 0) { // credit count in BCD; also where coins and start buttons are noticed
        int in = ~(p0 | (p1 << 4)), toggle = in ^ m->lastcoins;
        m->lastcoins = in;
        if (m->coins_per_cred[0] > 0) {
            if (m->credits < 99) {
                for (int k = 0; k < 2; k++)
                    if (toggle & in & (0x10 << k)) {
                        if (++m->coins[k] >= m->coins_per_cred[k]) {
                            m->credits += m->creds_per_coin[k];
                            m->coins[k] -= m->coins_per_cred[k];
                        }
                    }
                if (toggle & in & 0x40) m->credits++; // service coin
            }
        } else m->credits = 100; // free play
        if (m->mode == 1) {
            if (toggle & in & 0x04) { if (m->credits >= 1) { m->credits--; m->mode = 2; } }
            else if (toggle & in & 0x08) { if (m->credits >= 2) { m->credits -= 2; m->mode = 2; } }
        }
        if (~p1 & 0x08) return 0xbb; // test switch
        return (m->credits / 10) * 16 + m->credits % 10;
    }
    int in = ~p0, toggle = in ^ m->lastbuttons, bit = phase == 1 ? 1 : 2;
    int joy = m->in[phase == 1 ? 2 : 3] & 0x0f;
    m->lastbuttons = (m->lastbuttons & (3 ^ bit)) | (in & bit);
    if (m->remap_joy) joy = joy_map[joy];
    joy |= (toggle & in & bit) ? 0 : 0x10; // fire, edge
    joy |= (in & bit) ? 0 : 0x20;          // fire, level
    return joy;
}

static void z80_reset(Z80 *c) {
    c->pc = 0; c->i = c->r = c->r7 = 0; c->iff1 = c->iff2 = c->im = 0;
    c->halted = c->irq = c->nmi = c->ei = c->idle = 0;
    c->loop_pc = 0xffff;
}

static u8 io_rd(Machine *m, u16 a) {
    if ((a & 0xfff8) == 0x6800) return ((DSWB >> (a & 7)) & 1) | (((DSWA >> (a & 7)) & 1) << 1);
    if ((a & 0xff00) == 0x7000) {
        if (!(m->ctrl06 & 0x10)) return 0;
        return (m->ctrl06 & 1) ? n51_read(m) : 0xff;
    }
    if (a == 0x7100) return m->ctrl06;
    return 0;
}

static void io_wr(Machine *m, u16 a, u8 v) {
    if ((a & 0xfff8) == 0x6820) {
        switch (a & 7) {
        case 0: case 1:
            m->irq_mask[a & 1] = v & 1;
            if (!(v & 1)) m->cpu[a & 1].irq = 0;
            break;
        case 2: m->nmi_mask = !(v & 1); break;
        case 3:
            for (int k = 1; k < 3; k++) {
                if (!(v & 1)) { z80_reset(&m->cpu[k]); m->cpu[k].held = 1; }
                else m->cpu[k].held = 0;
            }
            break;
        }
    } else if ((a & 0xff00) == 0x7000) {
        if (!(m->ctrl06 & 0x10) && (m->ctrl06 & 1)) n51_write(m, v); // 54xx (noise) writes are dropped
    } else if (a == 0x7100) {
        m->ctrl06 = v;
        m->nmi06_at = (v & 0x0f) ? m->cpu[0].cyc + 614 : NEVER; // 200 us
        m->cpu[0].target = m->cpu[0].cyc;                       // make run_main() reschedule
    }
}

// ------------------------------------------------------------------------- Z80

static inline u8 rd(Machine *m, Z80 *c, u16 a) {
    if (a & 0x8000) return m->ram[a & 0x3fff];
    if (a < 0x4000) return c->rom[a];
    m->writes++;
    return io_rd(m, a);
}
static inline void wr(Machine *m, Z80 *c, u16 a, u8 v) {
    m->writes++;
    if (a & 0x8000) m->ram[a & 0x3fff] = v;
    else if (a >= 0x4000) io_wr(m, a, v);
}
static inline u16 rd16(Machine *m, Z80 *c, u16 a) { return rd(m, c, a) | (rd(m, c, a + 1) << 8); }
static inline void wr16(Machine *m, Z80 *c, u16 a, u16 v) { wr(m, c, a, v); wr(m, c, a + 1, v >> 8); }
static inline u8 imm8(Machine *m, Z80 *c) { u16 a = c->pc++; return a < 0x4000 ? c->rom[a] : rd(m, c, a); }
static inline u16 imm16(Machine *m, Z80 *c) { u16 v = rd16(m, c, c->pc); c->pc += 2; return v; }
static inline void push(Machine *m, Z80 *c, u16 v) { c->sp -= 2; wr16(m, c, c->sp, v); }
static inline u16 pop(Machine *m, Z80 *c) { u16 v = rd16(m, c, c->sp); c->sp += 2; return v; }

#define A c->af.h
#define F c->af.l
#define B c->bc.h
#define BC c->bc.w
#define DE c->de.w
#define HL c->hl.w

static const u8 roff[8] = {offsetof(Z80, bc) + 1, offsetof(Z80, bc), offsetof(Z80, de) + 1, offsetof(Z80, de),
                           offsetof(Z80, hl) + 1, offsetof(Z80, hl), 0, offsetof(Z80, af) + 1};
#define R8(n) (*((u8 *)c + roff[n]))

// A CPU that jumps backwards to the same place with identical registers, with no
// memory write or I/O access by anyone in between, is in a wait loop that only an
// interrupt or another CPU's write can end: park it (see the top of run()).
static inline void backjump(Machine *m, Z80 *c) {
    if (c->loop_w == m->writes && c->loop_pc == c->pc && !memcmp(c->loop_reg, &c->bc, 12) && c->loop_reg[6].w == c->sp) {
        if (c->cyc < c->target) c->cyc = c->target;
        c->idle = 1;
        c->idle_w = m->writes;
        return;
    }
    memcpy(c->loop_reg, &c->bc, 12); // bc, de, hl, af, ix, iy
    c->loop_reg[6].w = c->sp;
    c->loop_pc = c->pc;
    c->loop_w = m->writes;
}
static inline void jump(Machine *m, Z80 *c, u16 to) {
    int back = to < c->pc;
    c->pc = to;
    if (back) backjump(m, c);
}

static inline int cond(Z80 *c, int n) {
    static const u8 mask[4] = {ZF, CF, PF, SF};
    return ((F & mask[n >> 1]) != 0) == (n & 1);
}

static inline void alu(Z80 *c, int kind, u8 v) {
    int a = A, r;
    switch (kind) {
    case 0: case 1: // add, adc
        r = a + v + (kind & F & CF);
        F = sz[r & 0xff] | ((r >> 8) & CF) | ((a ^ v ^ r) & HF) | (((a ^ r) & (v ^ r) & 0x80) >> 5);
        A = r;
        break;
    case 2: case 3: case 7: // sub, sbc, cp
        r = a - v - (kind == 3 ? F & CF : 0);
        F = sz[r & 0xff] | ((r >> 8) & CF) | NF | ((a ^ v ^ r) & HF) | (((a ^ v) & (a ^ r) & 0x80) >> 5);
        if (kind == 7) F = (F & ~(XF | YF)) | (v & (XF | YF));
        else A = r;
        break;
    case 4: A &= v; F = szp[A] | HF; break;
    case 5: A ^= v; F = szp[A]; break;
    case 6: A |= v; F = szp[A]; break;
    }
}
static inline u8 inc8(Z80 *c, u8 v) {
    v++;
    F = (F & CF) | sz[v] | ((v & 0x0f) ? 0 : HF) | (v == 0x80 ? PF : 0);
    return v;
}
static inline u8 dec8(Z80 *c, u8 v) {
    v--;
    F = (F & CF) | NF | sz[v] | ((v & 0x0f) == 0x0f ? HF : 0) | (v == 0x7f ? PF : 0);
    return v;
}
static inline u16 add16(Z80 *c, u16 a, u16 b) {
    u32 r = a + b;
    F = (F & (SF | ZF | PF)) | (((a ^ b ^ r) >> 8) & HF) | ((r >> 16) & CF) | ((r >> 8) & (XF | YF));
    return r;
}
static inline u8 rot(Z80 *c, int kind, u8 v) {
    u8 r, cy;
    switch (kind) {
    case 0: r = (v << 1) | (v >> 7); cy = v >> 7; break;         // rlc
    case 1: r = (v >> 1) | (v << 7); cy = v & 1; break;          // rrc
    case 2: r = (v << 1) | (F & CF); cy = v >> 7; break;         // rl
    case 3: r = (v >> 1) | ((F & CF) << 7); cy = v & 1; break;   // rr
    case 4: r = v << 1; cy = v >> 7; break;                      // sla
    case 5: r = (v >> 1) | (v & 0x80); cy = v & 1; break;        // sra
    case 6: r = (v << 1) | 1; cy = v >> 7; break;                // sll
    default: r = v >> 1; cy = v & 1; break;                      // srl
    }
    F = szp[r] | cy;
    return r;
}
static inline void bit(Z80 *c, int n, u8 v) {
    F = (F & CF) | HF | (v & (XF | YF)) | ((v & (1 << n)) ? (n == 7 ? SF : 0) : (ZF | PF));
}

static void op_cb(Machine *m, Z80 *c) {
    u8 op = imm8(m, c);
    int reg = op & 7, n = (op >> 3) & 7;
    c->r++;
    u8 v = reg == 6 ? rd(m, c, HL) : R8(reg);
    c->cyc += reg == 6 ? ((op & 0xc0) == 0x40 ? 12 : 15) : 8;
    switch (op >> 6) {
    case 0: v = rot(c, n, v); break;
    case 1: bit(c, n, v); return;
    case 2: v &= ~(1 << n); break;
    case 3: v |= 1 << n; break;
    }
    if (reg == 6) wr(m, c, HL, v); else R8(reg) = v;
}

static void op_ed(Machine *m, Z80 *c) {
    u8 op = imm8(m, c);
    pair *rr = &c->bc; // bc, de, hl
    int p = (op >> 4) & 3;
    c->r++;
    c->cyc += 8;
    if (op >= 0x40 && op < 0x80) {
        switch (op & 0x0f) {
        case 0x0: case 0x8: { // in r,(c): no ports on this board
            if (((op >> 3) & 7) != 6) R8((op >> 3) & 7) = 0xff;
            F = (F & CF) | szp[0xff];
            c->cyc += 4;
            return;
        }
        case 0x1: case 0x9: c->cyc += 4; return; // out (c),r
        case 0x2: { // sbc hl,rr
            u32 a = HL, b = p == 3 ? c->sp : rr[p].w, r = a - b - (F & CF);
            F = (((a ^ b ^ r) >> 8) & HF) | NF | ((r >> 16) & CF) | ((r >> 8) & (SF | XF | YF)) |
                ((r & 0xffff) ? 0 : ZF) | (((b ^ a) & (a ^ r) & 0x8000) >> 13);
            HL = r;
            c->cyc += 7;
            return;
        }
        case 0xa: { // adc hl,rr
            u32 a = HL, b = p == 3 ? c->sp : rr[p].w, r = a + b + (F & CF);
            F = (((a ^ b ^ r) >> 8) & HF) | ((r >> 16) & CF) | ((r >> 8) & (SF | XF | YF)) |
                ((r & 0xffff) ? 0 : ZF) | (((b ^ a ^ 0x8000) & (b ^ r) & 0x8000) >> 13);
            HL = r;
            c->cyc += 7;
            return;
        }
        case 0x3: wr16(m, c, imm16(m, c), p == 3 ? c->sp : rr[p].w); c->cyc += 12; return;
        case 0xb: {
            u16 v = rd16(m, c, imm16(m, c));
            if (p == 3) c->sp = v; else rr[p].w = v;
            c->cyc += 12;
            return;
        }
        case 0x4: case 0xc: { u8 v = A; A = 0; alu(c, 2, v); return; } // neg
        case 0x5: case 0xd: c->pc = pop(m, c); c->iff1 = c->iff2; c->cyc += 6; return; // retn / reti
        case 0x6: case 0xe: c->im = (op & 0x10) ? ((op & 0x08) ? 2 : 1) : 0; return;
        case 0x7:
            switch (op) {
            case 0x47: c->i = A; c->cyc += 1; return;
            case 0x57: A = c->i; F = (F & CF) | sz[A] | (c->iff2 ? PF : 0); c->cyc += 1; return;
            case 0x67: { // rrd
                u8 t = rd(m, c, HL);
                wr(m, c, HL, (t >> 4) | (A << 4));
                A = (A & 0xf0) | (t & 0x0f);
                F = (F & CF) | szp[A];
                c->cyc += 10;
                return;
            }
            }
            return;
        case 0xf:
            switch (op) {
            case 0x4f: c->r = A; c->r7 = A & 0x80; c->cyc += 1; return;
            case 0x5f: A = (c->r & 0x7f) | c->r7; F = (F & CF) | sz[A] | (c->iff2 ? PF : 0); c->cyc += 1; return;
            case 0x6f: { // rld
                u8 t = rd(m, c, HL);
                wr(m, c, HL, (t << 4) | (A & 0x0f));
                A = (A & 0xf0) | (t >> 4);
                F = (F & CF) | szp[A];
                c->cyc += 10;
                return;
            }
            }
            return;
        }
    }
    int d = (op & 0x08) ? -1 : 1, rep = op & 0x10;
    switch (op & 0xe7) {
    case 0xa0: { // ldi ldd ldir lddr
        u8 v = rd(m, c, HL);
        wr(m, c, DE, v);
        HL += d; DE += d; BC--;
        v += A;
        F = (F & (SF | ZF | CF)) | (BC ? PF : 0) | (v & XF) | ((v & 2) ? YF : 0);
        c->cyc += 8;
        if (rep && BC) { c->pc -= 2; c->cyc += 5; }
        return;
    }
    case 0xa1: { // cpi cpd cpir cpdr
        u8 v = rd(m, c, HL), r = A - v, h = (A ^ v ^ r) & HF;
        HL += d; BC--;
        F = (F & CF) | NF | (sz[r] & ~(XF | YF)) | h | (BC ? PF : 0);
        r -= h ? 1 : 0;
        F |= (r & XF) | ((r & 2) ? YF : 0);
        c->cyc += 8;
        if (rep && BC && !(F & ZF)) { c->pc -= 2; c->cyc += 5; }
        return;
    }
    }
    g_unimpl++;
}

static void step(Machine *m, Z80 *c);

static void op_xy(Machine *m, Z80 *c, pair *xy) {
    u8 op = imm8(m, c);
    int src = op & 7, dst = (op >> 3) & 7;
    c->r++;
#define EA ((u16)(xy->w + (int8_t)imm8(m, c)))
#define XR(n) (*((n) == 4 ? &xy->h : (n) == 5 ? &xy->l : &R8(n)))
    switch (op) {
    case 0x09: xy->w = add16(c, xy->w, BC); c->cyc += 15; return;
    case 0x19: xy->w = add16(c, xy->w, DE); c->cyc += 15; return;
    case 0x29: xy->w = add16(c, xy->w, xy->w); c->cyc += 15; return;
    case 0x39: xy->w = add16(c, xy->w, c->sp); c->cyc += 15; return;
    case 0x21: xy->w = imm16(m, c); c->cyc += 14; return;
    case 0x22: wr16(m, c, imm16(m, c), xy->w); c->cyc += 20; return;
    case 0x2a: xy->w = rd16(m, c, imm16(m, c)); c->cyc += 20; return;
    case 0x23: xy->w++; c->cyc += 10; return;
    case 0x2b: xy->w--; c->cyc += 10; return;
    case 0x24: xy->h = inc8(c, xy->h); c->cyc += 8; return;
    case 0x25: xy->h = dec8(c, xy->h); c->cyc += 8; return;
    case 0x2c: xy->l = inc8(c, xy->l); c->cyc += 8; return;
    case 0x2d: xy->l = dec8(c, xy->l); c->cyc += 8; return;
    case 0x26: xy->h = imm8(m, c); c->cyc += 11; return;
    case 0x2e: xy->l = imm8(m, c); c->cyc += 11; return;
    case 0x34: { u16 a = EA; wr(m, c, a, inc8(c, rd(m, c, a))); c->cyc += 23; return; }
    case 0x35: { u16 a = EA; wr(m, c, a, dec8(c, rd(m, c, a))); c->cyc += 23; return; }
    case 0x36: { u16 a = EA; wr(m, c, a, imm8(m, c)); c->cyc += 19; return; }
    case 0xcb: {
        u16 a = EA;
        u8 op2 = imm8(m, c), v = rd(m, c, a);
        int n = (op2 >> 3) & 7;
        c->cyc += 23;
        switch (op2 >> 6) {
        case 0: v = rot(c, n, v); break;
        case 1: bit(c, n, v); c->cyc -= 3; return;
        case 2: v &= ~(1 << n); break;
        case 3: v |= 1 << n; break;
        }
        wr(m, c, a, v);
        if ((op2 & 7) != 6) R8(op2 & 7) = v;
        return;
    }
    case 0xe1: xy->w = pop(m, c); c->cyc += 14; return;
    case 0xe5: push(m, c, xy->w); c->cyc += 15; return;
    case 0xe3: { u16 t = rd16(m, c, c->sp); wr16(m, c, c->sp, xy->w); xy->w = t; c->cyc += 23; return; }
    case 0xe9: jump(m, c, xy->w); c->cyc += 8; return;
    case 0xf9: c->sp = xy->w; c->cyc += 10; return;
    }
    if (op >= 0x40 && op < 0xc0 && op != 0x76 && (src == 6 || (src & 6) == 4 || (op < 0x80 && (dst == 6 || (dst & 6) == 4)))) {
        if (op < 0x80) {
            if (src == 6) { R8(dst) = rd(m, c, EA); c->cyc += 19; }
            else if (dst == 6) { wr(m, c, EA, R8(src)); c->cyc += 19; }
            else { XR(dst) = XR(src); c->cyc += 8; }
        } else if (src == 6) { alu(c, dst, rd(m, c, EA)); c->cyc += 19; }
        else { alu(c, dst, XR(src)); c->cyc += 8; }
        return;
    }
    // Not an index instruction: the prefix is a 4-cycle no-op in front of a normal one.
    c->pc--;
    c->cyc += 4;
    step(m, c);
#undef EA
#undef XR
}

static const u8 cc_op[256] = {
    4, 10, 7, 6, 4, 4, 7, 4, 4, 11, 7, 6, 4, 4, 7, 4,  8, 10, 7, 6, 4, 4, 7, 4, 12, 11, 7, 6, 4, 4, 7, 4,
    7, 10, 16, 6, 4, 4, 7, 4, 7, 11, 16, 6, 4, 4, 7, 4, 7, 10, 13, 6, 11, 11, 10, 4, 7, 11, 13, 6, 4, 4, 7, 4,
    4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 4, 7, 4,     4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 4, 7, 4,
    4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 4, 7, 4,     7, 7, 7, 7, 7, 7, 4, 7, 4, 4, 4, 4, 4, 4, 7, 4,
    4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 4, 7, 4,     4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 4, 7, 4,
    4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 4, 7, 4,     4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 4, 7, 4,
    5, 10, 10, 10, 10, 11, 7, 11, 5, 10, 10, 0, 10, 17, 7, 11, 5, 10, 10, 11, 10, 11, 7, 11, 5, 4, 10, 11, 10, 0, 7, 11,
    5, 10, 10, 19, 10, 11, 7, 11, 5, 4, 10, 4, 10, 0, 7, 11,  5, 10, 10, 4, 10, 11, 7, 11, 5, 6, 10, 4, 10, 0, 7, 11};

static void step(Machine *m, Z80 *c) {
    u8 op = imm8(m, c);
    int y = (op >> 3) & 7, z = op & 7;
    pair *rr = &c->bc; // bc, de, hl, af are consecutive
    c->r++;
    c->cyc += cc_op[op];
    if (op >= 0x40 && op < 0xc0) {
        if (op >= 0x80) { alu(c, y, z == 6 ? rd(m, c, HL) : R8(z)); return; }
        if (op == 0x76) { c->halted = 1; c->pc--; return; }
        if (z == 6) R8(y) = rd(m, c, HL);
        else if (y == 6) wr(m, c, HL, R8(z));
        else R8(y) = R8(z);
        return;
    }
    switch (op) {
    case 0x00: return;
    case 0x01: case 0x11: case 0x21: rr[op >> 4].w = imm16(m, c); return;
    case 0x31: c->sp = imm16(m, c); return;
    case 0x02: wr(m, c, BC, A); return;
    case 0x12: wr(m, c, DE, A); return;
    case 0x0a: A = rd(m, c, BC); return;
    case 0x1a: A = rd(m, c, DE); return;
    case 0x03: case 0x13: case 0x23: rr[op >> 4].w++; return;
    case 0x33: c->sp++; return;
    case 0x0b: case 0x1b: case 0x2b: rr[op >> 4].w--; return;
    case 0x3b: c->sp--; return;
    case 0x04: case 0x0c: case 0x14: case 0x1c: case 0x24: case 0x2c: case 0x3c: R8(y) = inc8(c, R8(y)); return;
    case 0x05: case 0x0d: case 0x15: case 0x1d: case 0x25: case 0x2d: case 0x3d: R8(y) = dec8(c, R8(y)); return;
    case 0x34: wr(m, c, HL, inc8(c, rd(m, c, HL))); return;
    case 0x35: wr(m, c, HL, dec8(c, rd(m, c, HL))); return;
    case 0x06: case 0x0e: case 0x16: case 0x1e: case 0x26: case 0x2e: case 0x3e: R8(y) = imm8(m, c); return;
    case 0x36: wr(m, c, HL, imm8(m, c)); return;
    case 0x07: A = (A << 1) | (A >> 7); F = (F & (SF | ZF | PF)) | (A & (XF | YF | CF)); return;
    case 0x0f: F = (F & (SF | ZF | PF)) | (A & CF); A = (A >> 1) | (A << 7); F |= A & (XF | YF); return;
    case 0x17: { u8 cy = A >> 7; A = (A << 1) | (F & CF); F = (F & (SF | ZF | PF)) | cy | (A & (XF | YF)); return; }
    case 0x1f: { u8 cy = A & 1; A = (A >> 1) | ((F & CF) << 7); F = (F & (SF | ZF | PF)) | cy | (A & (XF | YF)); return; }
    case 0x08: { pair t = c->af; c->af = c->af2; c->af2 = t; return; }
    case 0x09: case 0x19: case 0x29: HL = add16(c, HL, rr[op >> 4].w); return;
    case 0x39: HL = add16(c, HL, c->sp); return;
    case 0x10: { int8_t d = imm8(m, c); if (--B) { c->pc += d; c->cyc += 5; } return; }
    case 0x18: { int8_t d = imm8(m, c); jump(m, c, c->pc + d); return; }
    case 0x20: case 0x28: case 0x30: case 0x38: {
        int8_t d = imm8(m, c);
        if (cond(c, y - 4)) { c->cyc += 5; jump(m, c, c->pc + d); }
        return;
    }
    case 0x22: wr16(m, c, imm16(m, c), HL); return;
    case 0x2a: HL = rd16(m, c, imm16(m, c)); return;
    case 0x32: wr(m, c, imm16(m, c), A); return;
    case 0x3a: A = rd(m, c, imm16(m, c)); return;
    case 0x27: { // daa
        u8 a = A, adj = 0, cy = F & CF, h;
        if ((F & HF) || (a & 0x0f) > 9) adj = 6;
        if (cy || a > 0x99) { adj |= 0x60; cy = 1; }
        if (F & NF) { h = (F & HF) && (a & 0x0f) < 6; a -= adj; }
        else { h = (a & 0x0f) > 9; a += adj; }
        A = a;
        F = (F & NF) | cy | szp[a] | (h ? HF : 0);
        return;
    }
    case 0x2f: A = ~A; F = (F & (SF | ZF | PF | CF)) | HF | NF | (A & (XF | YF)); return;
    case 0x37: F = (F & (SF | ZF | PF)) | CF | (A & (XF | YF)); return;
    case 0x3f: F = ((F & (SF | ZF | PF | CF)) | ((F & CF) << 4) | (A & (XF | YF))) ^ CF; return;
    case 0xc0: case 0xc8: case 0xd0: case 0xd8: case 0xe0: case 0xe8: case 0xf0: case 0xf8:
        if (cond(c, y)) { c->pc = pop(m, c); c->cyc += 6; }
        return;
    case 0xc9: c->pc = pop(m, c); return;
    case 0xc1: case 0xd1: case 0xe1: case 0xf1: rr[(op >> 4) & 3].w = pop(m, c); return;
    case 0xc5: case 0xd5: case 0xe5: case 0xf5: push(m, c, rr[(op >> 4) & 3].w); return;
    case 0xc2: case 0xca: case 0xd2: case 0xda: case 0xe2: case 0xea: case 0xf2: case 0xfa: {
        u16 a = imm16(m, c);
        if (cond(c, y)) jump(m, c, a);
        return;
    }
    case 0xc3: jump(m, c, imm16(m, c)); return;
    case 0xc4: case 0xcc: case 0xd4: case 0xdc: case 0xe4: case 0xec: case 0xf4: case 0xfc: {
        u16 a = imm16(m, c);
        if (cond(c, y)) { push(m, c, c->pc); c->pc = a; c->cyc += 7; }
        return;
    }
    case 0xcd: { u16 a = imm16(m, c); push(m, c, c->pc); c->pc = a; return; }
    case 0xc6: case 0xce: case 0xd6: case 0xde: case 0xe6: case 0xee: case 0xf6: case 0xfe: alu(c, y, imm8(m, c)); return;
    case 0xc7: case 0xcf: case 0xd7: case 0xdf: case 0xe7: case 0xef: case 0xf7: case 0xff:
        push(m, c, c->pc); c->pc = op & 0x38; return;
    case 0xcb: op_cb(m, c); return;
    case 0xed: op_ed(m, c); return;
    case 0xdd: op_xy(m, c, &c->ix); return;
    case 0xfd: op_xy(m, c, &c->iy); return;
    case 0xd3: c->pc++; return;            // out (n),a
    case 0xdb: c->pc++; A = 0xff; return;  // in a,(n)
    case 0xd9: {
        pair t;
        t = c->bc; c->bc = c->bc2; c->bc2 = t;
        t = c->de; c->de = c->de2; c->de2 = t;
        t = c->hl; c->hl = c->hl2; c->hl2 = t;
        return;
    }
    case 0xe3: { u16 t = rd16(m, c, c->sp); wr16(m, c, c->sp, HL); HL = t; return; }
    case 0xe9: jump(m, c, HL); return;
    case 0xeb: { u16 t = DE; DE = HL; HL = t; return; }
    case 0xf3: c->iff1 = c->iff2 = 0; return;
    case 0xfb: c->iff1 = c->iff2 = 1; c->ei = 1; return;
    case 0xf9: c->sp = HL; return;
    }
    g_unimpl++;
}

#ifdef GX_TRACE
static long g_trace; // instructions of main-CPU trace still to print
#endif

static void run(Machine *m, Z80 *c, int target) {
    c->target = target;
    if (c->held) { if (c->cyc < target) c->cyc = target; return; }
    if (c->idle) {
        if (c->idle_w == m->writes && !c->nmi && !(c->irq && c->iff1)) { if (c->cyc < target) c->cyc = target; return; }
        c->idle = 0;
    }
    while (c->cyc < c->target) {
        if (c->nmi) {
            c->nmi = 0; c->halted = 0; c->iff1 = 0;
            push(m, c, c->pc);
            c->pc = 0x66;
            c->cyc += 11;
        } else if (c->irq && c->iff1 && !c->ei) {
            c->halted = 0; c->iff1 = c->iff2 = 0;
            push(m, c, c->pc);
            c->pc = c->im == 2 ? rd16(m, c, (c->i << 8) | 0xff) : 0x38;
            c->cyc += 13;
        }
        c->ei = 0;
        if (c->halted) { c->cyc = c->target; break; }
#ifdef GX_TRACE
        if (g_trace > 0 && c == &m->cpu[0]) {
            g_trace--;
            printf("%04X %04X %04X %04X %04X %04X\n", c->pc, c->af.w, c->bc.w, c->de.w, c->hl.w, c->sp);
        }
#endif
        step(m, c);
    }
}

// The real 51xx (as emulated by MAME from its MCU ROM) reports every input one
// frame late, and buttons and coins only once two consecutive frames agree. The
// stick is not debounced. Measured against MAME with press/release patterns.
static void latch_inputs(Machine *m) {
    for (int k = 0; k < 4; k++) {
        u8 late = m->raw_prev[k];
        if (k < 2) {
            u8 bouncing = late ^ m->dly_prev[k];
            m->in[k] = (m->in[k] & bouncing) | (late & ~bouncing);
            m->dly_prev[k] = late;
        } else m->in[k] = late;
        m->raw_prev[k] = m->raw[k];
    }
}

// One video frame, from vblank to vblank, so the state seen between frames is
// what MAME's frame callback sees.
static void run_frame(Machine *m) {
    Z80 *c0 = &m->cpu[0];
    latch_inputs(m);
    if (m->irq_mask[0]) m->cpu[0].irq = 1; // vblank, scanline 224
    if (m->irq_mask[1]) m->cpu[1].irq = 1;
    for (int s = 1; s <= FRAME / SLICE; s++) {
        int t = s * SLICE;
        while (c0->cyc < t) { // main CPU, stopping at each 06xx NMI
            run(m, c0, t < m->nmi06_at ? t : m->nmi06_at);
            if (c0->cyc >= m->nmi06_at) { c0->nmi = 1; m->nmi06_at += 614; }
        }
        run(m, &m->cpu[1], t);
        run(m, &m->cpu[2], t);
        if ((s == 39 || s == 87) && m->nmi_mask) m->cpu[2].nmi = 1; // scanlines 64 and 192
    }
    for (int k = 0; k < 3; k++) m->cpu[k].cyc -= FRAME;
    if (m->nmi06_at != NEVER) m->nmi06_at -= FRAME;
}

static void machine_init(Machine *m) {
    memset(m, 0, sizeof *m);
    for (int k = 0; k < 3; k++) {
        Z80 *c = &m->cpu[k];
        c->rom = g_rom[k];
        c->af.w = 0x0040;
        c->ix.w = c->iy.w = 0xffff;
        z80_reset(c);
        c->held = k > 0; // latch 3 powers up at 0: sub CPUs held in reset
    }
    m->nmi06_at = NEVER;
    memset(m->in, 0x0f, 16); // in, raw, raw_prev, dly_prev: nothing pressed
}

static int load_roms(const char *dir) {
    static const struct { const char *name; int cpu, at; } files[] = {
        {"gg1_1b.3p", 0, 0x0000}, {"gg1_2b.3m", 0, 0x1000}, {"gg1_3.2m", 0, 0x2000},
        {"gg1_4b.2l", 0, 0x3000}, {"gg1_5b.3f", 1, 0x0000}, {"gg1_7b.2c", 2, 0x0000}};
    memset(g_rom, 0xff, sizeof g_rom);
    for (int k = 0; k < 6; k++) {
        char path[1024];
        snprintf(path, sizeof path, "%s/%s", dir, files[k].name);
        FILE *f = fopen(path, "rb");
        if (!f || fread(g_rom[files[k].cpu] + files[k].at, 1, 0x1000, f) != 0x1000) return -1;
        fclose(f);
    }
    for (int v = 0; v < 256; v++) {
        int p = 0;
        for (int b = 0; b < 8; b++) p ^= v >> b;
        sz[v] = (v & (SF | XF | YF)) | (v ? 0 : ZF);
        szp[v] = sz[v] | ((p & 1) ? 0 : PF);
    }
    return 0;
}

// ------------------------------------------------------ training environments

#define RAM(m, a) ((m)->ram[(a)-0x8000])
#define PLAYING(m) (RAM(m, 0x9201) == 3) // b8_9201_game_state: 3 = PLAY_MODE
#define SHIPS(m) RAM(m, 0x9820)          // ds_plyr_actv._b_nships
#define STAGE(m) RAM(m, 0x9821)          // ds_plyr_actv._b_stgctr
#define HITS(m) (RAM(m, 0x9844) | (RAM(m, 0x9845) << 8))  // ds_plyr_actv._w_hit_ct
#define SHOTS(m) (RAM(m, 0x9846) | (RAM(m, 0x9847) << 8)) // ds_plyr_actv._w_shot_ct
#define OBS 384
#define SKIP 4 // frames per agent step

static Machine *g_env, *g_pool;
static int g_nenv, g_npool;
static u32 *g_rng;
static float g_hit_reward = 2.0f, g_death_penalty = 7.0f, g_clear_reward = 20.0f, g_step_penalty = 0.02f, g_miss_penalty = 0.5f;

static void set_inputs(Machine *m, int left, int right, int fire, int coin, int start) {
    m->raw[0] = 0x0f & ~(fire ? 1 : 0) & ~(start ? 4 : 0);
    m->raw[1] = 0x0f & ~(coin ? 1 : 0);
    m->raw[2] = 0x0f & ~(right ? 2 : 0) & ~(left ? 8 : 0);
}

// Coin + start until the game reports play mode; same sequence as bridge.lua.
static int start_game(Machine *m, int max_frames) {
    for (int n = 0; n < max_frames; n++) {
        int t = n % 120;
        set_inputs(m, 0, 0, 0, t < 6, t >= 60 && t < 66);
        run_frame(m);
        if (PLAYING(m)) return n;
    }
    return -1;
}

static void observe(const Machine *m, float *obs) {
    static const u16 base[3] = {0x8800, 0x9300, 0x9B00}; // object state, sprite position, sprite control
    for (int k = 0; k < 3; k++)
        for (int j = 0; j < 0x80; j++) *obs++ = m->ram[base[k] - 0x8000 + j] * (1.0f / 255.0f);
}

// Boots once, then builds `npool` distinct game-start snapshots (each starts the
// game a different number of frames into attract mode). Returns 0 on success.
EXPORT int gx_init(const char *romdir, int nenv, int npool, unsigned seed) {
    if (load_roms(romdir)) return -1;
    Machine boot;
    machine_init(&boot);
    for (int n = 0, attract = 0; attract < 60; n++) { // attract mode, held (the RAM test scribbles on this byte)
        if (n > 3000) return -2;
        run_frame(&boot);
        attract = RAM(&boot, 0x9201) == 1 ? attract + 1 : 0;
    }
    free(g_env); free(g_pool); free(g_rng);
    g_env = malloc(sizeof(Machine) * nenv);
    g_pool = malloc(sizeof(Machine) * npool);
    g_rng = malloc(sizeof(u32) * nenv);
    g_nenv = nenv; g_npool = npool;
    for (int k = 0; k < npool; k++) {
        g_pool[k] = boot;
        for (int n = 0; n < k * 7; n++) run_frame(&g_pool[k]);
        if (start_game(&g_pool[k], 3000) < 0) return -3;
    }
    for (int k = 0; k < nenv; k++) {
        g_rng[k] = seed * 2654435761u + k * 40503u + 1;
        g_env[k] = g_pool[k % npool];
    }
    return 0;
}

EXPORT void gx_set_rewards(float hit, float death, float clear, float step, float miss) {
    g_hit_reward = hit; g_death_penalty = death; g_clear_reward = clear; g_step_penalty = step; g_miss_penalty = miss;
}

static void step_env(int k, const u8 *act, float *obs, float *rew, u8 *done, int *info) {
    Machine *m = &g_env[k];
    int a = act[k], hits = HITS(m), shots = SHOTS(m), ships = SHIPS(m), stage = STAGE(m);
    for (int f = 0; f < SKIP; f++) { // fire: 2 frames down, 2 up, so the 51xx sees a fresh press every step
        set_inputs(m, a == 1 || a == 4, a == 2 || a == 5, a >= 3 && f < 2, 0, 0);
        run_frame(m);
    }
    // Same rules as train.py: _b_nships drops when the first ship deploys (stage
    // still 0, not a death), then once per ship lost, wrapping 0 -> 255 at the end.
    // It goes up when a bonus ship is awarded, which is not a death.
    int died = ((ships - SHIPS(m)) & 0xff) == 1 && STAGE(m) > 0;
    int gained = HITS(m) - hits, fired = SHOTS(m) - shots;
    int cleared = stage > 0 && STAGE(m) > stage; // 0 -> 1 is the game starting, not a clear
    // Misses: every shot is charged when fired and refunded if it hits, since a miss is only known later.
    gained = gained > 0 ? gained : 0;
    fired = fired > 0 ? fired : 0;
    rew[k] = (g_hit_reward + g_miss_penalty) * gained - g_miss_penalty * fired + (cleared ? g_clear_reward : 0) -
             (died ? g_death_penalty : 0) - g_step_penalty;
    done[k] = !PLAYING(m) || SHIPS(m) == 255;
    info[2 * k] = HITS(m);
    info[2 * k + 1] = STAGE(m);
    if (done[k]) {
        g_rng[k] = g_rng[k] * 1664525u + 1013904223u;
        *m = g_pool[(g_rng[k] >> 16) % g_npool];
    }
    observe(m, obs + k * OBS);
}

// One agent step for environments lo..hi-1. Finished games restart automatically
// (obs is then the first observation of the new game). info: hits, stage per env.
// train.py calls this from one thread per core, each with its own range.
EXPORT void gx_step_range(int lo, int hi, const u8 *act, float *obs, float *rew, u8 *done, int *info) {
    for (int k = lo; k < hi; k++) step_env(k, act, obs, rew, done, info);
}

EXPORT void gx_reset(float *obs) {
    for (int k = 0; k < g_nenv; k++) {
        g_rng[k] = g_rng[k] * 1664525u + 1013904223u;
        g_env[k] = g_pool[(g_rng[k] >> 16) % g_npool];
        observe(&g_env[k], obs + k * OBS);
    }
}

#ifdef GX_MAIN
#include <time.h>
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "roms/galaga";
    if (load_roms(dir)) { fprintf(stderr, "cannot read ROMs in %s\n", dir); return 1; }
    Machine m;
    machine_init(&m);
#ifdef GX_TRACE
    g_trace = argc > 2 ? atol(argv[2]) : 100000;
    for (int n = 0; n < 2000 && g_trace > 0; n++) run_frame(&m);
    return 0;
#endif
    int last = -1;
    for (int n = 0; n < 1500; n++) {
        run_frame(&m);
        if (RAM(&m, 0x9201) != last) {
            last = RAM(&m, 0x9201);
            printf("frame %d: game_state=%d\n", n, last);
        }
    }
    int started = start_game(&m, 3000);
    printf("start_game: playing after %d frames, ships=%d stage=%d, unimplemented ops=%ld\n", started, SHIPS(&m),
           STAGE(&m), g_unimpl);
    if (started < 0) return 1;

    // random play, single machine, with the same rules the environments use
    u32 rng = 12345;
    int steps = 0, episodes = 0;
    double t0 = now();
    Machine s = m;
    while (episodes < 20) {
        int hits = HITS(&m), ships = SHIPS(&m);
        rng = rng * 1664525u + 1013904223u;
        int a = (rng >> 16) % 6;
        for (int f = 0; f < SKIP; f++) {
            set_inputs(&m, a == 1 || a == 4, a == 2 || a == 5, a >= 3 && f < 2, 0, 0);
            run_frame(&m);
        }
        steps++;
        if (((ships - SHIPS(&m)) & 0xff) == 1 && STAGE(&m) > 0) printf("  step %d: ships %d -> %d, hits %d\n", steps, ships, SHIPS(&m), hits);
        if (!PLAYING(&m) || SHIPS(&m) == 255) {
            printf("episode %d: %d steps so far, hits=%d stage=%d\n", ++episodes, steps, HITS(&m), STAGE(&m));
            m = s;
        }
    }
    double dt = now() - t0;
    printf("%d steps (%d frames) in %.3fs = %.0f frames/s = %.0fx real time, single core; unimplemented ops=%ld\n",
           steps, steps * SKIP, dt, steps * SKIP / dt, steps * SKIP / dt / 60.606, g_unimpl);
    return 0;
}
#endif
