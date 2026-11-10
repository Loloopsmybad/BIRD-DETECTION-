/*
 * birdcpp.c - the whole live bird pipeline for the TI SK-AM62, in C99.
 *
 * A C port of board_camera_server.py (the Python birdboard) with the
 * same dashboard HTML, the same routes, the same /camera-set form
 * semantics (checkbox absent = off; scalars absent = unchanged; fps
 * preset overrides vblank; mode/flip changes rebuild the pipeline) and
 * the same user-tuned defaults. Architecture:
 *
 *   capture thread   fork/exec frame_pumper (same proven C helper used
 *                    by camkit/python), reads the raw RG10 frames from
 *                    the pipe here - no Python per frame;
 *                    media-ctl / v4l2-ctl are spawned only per session
 *                    or per settings apply
 *   debayer          in-house: 2x2 Bayer bin for the "half" view, full
 *                    bilinear 4-phase debayer (pthread row split) for
 *                    "3/4"/"full", bilinear resize in between
 *   color            gray-world AWB (EMA) + gamma-0.55 LUT, exactly the
 *                    python formulas, byte tables only in the hot loop
 *   JPEG             libturbojpeg (stable API declared here: the board
 *                    ships /usr/lib/libturbojpeg.so.0 but no
 *                    turbojpeg.h); `--tjtest` verifies the TJPF_BGR
 *                    channel-order assumption empirically
 *   HTTP             minimal own server (thread per connection):
 *                    / /stream /meta.json /phone /camera-set (POST and
 *                    GET, 303 to /) /ping - snapshot JPEG capture is
 *                    GONE (/frame.jpg, /det.jpg respond 410): live
 *                    media only, so no capture happens while nobody
 *                    watches (the pumper is SIGSTOP-suspended when the
 *                    last /stream viewer closes, resumed instantly when
 *                    one opens)
 *   inference        native onnxruntime C API, loaded with dlopen at
 *                    runtime (official aarch64 package at
 *                    /home/root/ort): the board libstdc++ has no
 *                    symbol-version records so a link-time bind fails;
 *                    dlsym(OrtGetApiBase)() works exactly like cv2's
 *                    runtime bind. v1.31.1 4 intra threads: ~1.1 s
 *                    forward vs ~2.3 s cv2.dnn. Letterbox, YOLOv8 NMS,
 *                    box drawing (builtin 5x7 font) and JSON are all C.
 *                    Detection boxes are drawn live into the annotated
 *                    MPJPEG stream (/stream; plain view on /stream?raw).
 *
 * NOTHING is written to disk while the live feed runs; detection
 * results and the annotated frame live in RAM like the python server.
 *
 * build: gcc -O2 -std=gnu99 -Wall -pthread birdcpp.c -o birdcpp \
 *            -l:libturbojpeg.so.0 -lm   (see Makefile)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <poll.h>
#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sched.h>                    /* cpu affinity pin (opt-B core partition) */
#include <sys/resource.h>             /* setpriority(2): low-prio copier         */

/* ------------------------------------------------------------------ *
 * onnxruntime C API (v31 = 1.31.x): the struct OrtApi declarations come
 * from the package header at build time; the SHARED LIBRARY is loaded
 * at runtime with dlopen (link-time bind fails on this board - see
 * README-CPP.md). Build: gcc ... -I/home/root/ort/include
 * ------------------------------------------------------------------ */
#include "onnxruntime_c_api.h"

/* ------------------------------------------------------------------ *
 * libturbojpeg - the stable public API declared from memory: the board
 * ships /usr/lib/libturbojpeg.so.0 but no turbojpeg.h. Enum numbers are
 * libjpeg-turbo 2.x; `--tjtest` verifies the BGR layout empirically.
 * ------------------------------------------------------------------ */
typedef void *tjhandle;
enum { TJSAMP_444J = 0, TJSAMP_422J, TJSAMP_420J, TJSAMP_GRAYJ,
       TJSAMP_440J, TJSAMP_411J };
enum { TJPF_RGB = 0, TJPF_BGR, TJPF_GRAY, TJPF_RGBA, TJPF_BGRA,
       TJPF_ARGB, TJPF_ABGR };

extern tjhandle tjInitCompress(void);
extern int tjCompress2(tjhandle handle, const unsigned char *srcBuf,
                       int width, int pitch, int height, int pixelFormat,
                       unsigned char **jpegBuf, unsigned long *jpegSize,
                       int jpegSubsamp, int jpegQual, int flags);
extern void tjFree(unsigned char *jpegBuf);
extern char *tjGetErrorStr2(tjhandle handle);

/* ------------------------------------------------------------------ *
 * tiny helpers
 * ------------------------------------------------------------------ */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void logf_(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("birdcpp: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputs("\n", stderr);
    fflush(stderr);
    va_end(ap);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { logf_("out of memory (%zu)", n); exit(1); }
    return p;
}

static char *xstrdup(const char *s)
{
    char *d = strdup(s ? s : "");
    if (!d) { logf_("out of memory"); exit(1); }
    return d;
}

static char *read_file_full(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0 || n > 256L * 1024 * 1024) { fclose(f); return NULL; }
    char *b = xmalloc((size_t)n + 1);
    size_t got = fread(b, 1, (size_t)n, f);
    fclose(f);
    b[got] = 0;
    if (out_len) *out_len = got;
    return b;
}

static int write_all(int fd, const void *buf, size_t len)
{
    const unsigned char *p = (const unsigned char *)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

/* dynamic string buffer */
typedef struct { char *b; size_t len, cap; } sbuf;

static void sb_init(sbuf *s)
{
    s->cap = 4096;
    s->len = 0;
    s->b = xmalloc(s->cap);
    s->b[0] = 0;
}

static void sb_ensure(sbuf *s, size_t want)
{
    if (s->len + want + 1 > s->cap) {
        size_t nc = s->cap;
        while (nc < s->len + want + 1) nc *= 2;
        char *nb = realloc(s->b, nc);
        if (!nb) { logf_("out of memory"); exit(1); }
        s->b = nb;
        s->cap = nc;
    }
}

static void sb_appn(sbuf *s, const char *t, size_t n)
{
    if (n) {
        sb_ensure(s, n);
        memcpy(s->b + s->len, t, n);
        s->len += n;
        s->b[s->len] = 0;
    }
}

static void sb_app(sbuf *s, const char *t)
{
    sb_appn(s, t, t ? strlen(t) : 0);
}

static void sb_appc(sbuf *s, char c)
{
    sb_ensure(s, 1);
    s->b[s->len++] = c;
    s->b[s->len] = 0;
}

static void sb_appf(sbuf *s, const char *fmt, ...)
{
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n >= (int)sizeof(tmp)) {            /* rare long case */
        char *big = xmalloc((size_t)n + 1);
        va_start(ap, fmt);
        vsnprintf(big, (size_t)n + 1, fmt, ap);
        va_end(ap);
        sb_app(s, big);
        free(big);
        return;
    }
    if (n > 0) sb_appn(s, tmp, (size_t)n);
}

static void sb_free(sbuf *s)
{
    free(s->b);
    s->b = NULL;
    s->len = s->cap = 0;
}

/* ------------------------------------------------------------------ *
 * configuration constants (mirrors of the python birdboard)
 * ------------------------------------------------------------------ */
#define HOME_ROOTBIRD   "/home/root/bird"
#define HOME_BIRDCPP    "/home/root/birdcpp"
#define URLFILE         HOME_BIRDCPP "/camera_url.txt"
#define MODEL_PATH      HOME_BIRDCPP "/best.onnx"
#define ORT_LIB_DEFAULT "/home/root/ort/lib/libonnxruntime.so"
#define SENSOR_DEV      "/dev/v4l-subdev2"
#define V4L_BIN         "/usr/bin/v4l2-ctl"
#define MEDIA_BIN       "/usr/bin/media-ctl"
#define MEDIA_DEV       "/dev/media0"
#define RETRY_EVERY     2.0            /* s between reconnect attempts  */
#define FRAME_STALL     15.0           /* s with no frames -> resync    */
#define PIPE_WIDEN      (1 << 20)      /* F_SETPIPE_SZ for 4MB frames   */
#define F_SETPIPE_SZ_L  1031

/* native YOLOv8 inference (python birdboard parity) */
/* model letterbox size is a runtime option (--img N, multiple of 32,
 * 320..640; default 640). The systemd unit pairs it with the deployed
 * onnx: production runs best_448.onnx + --img 448 (promoted 2026-11-10
 * from the trial-a measurements: cycle ~1.09 s live at -3.1% mAP50 vs
 * 640; 640 artifact stays deployed as best.onnx fallback mode).        */

static int ORT_IMG = 640;
#define ORT_CONF        0.25f          /* score threshold               */
#define ORT_IOU         0.45f          /* NMS IoU threshold             */
#define MAX_DETS        64
#define DET_IDLE_S      15.0           /* s a det frame stays "fresh"   */

#define NOTE_MAX 220

#define NMODES 3
static const char *const MODE_NM[NMODES] = { "1640x1232", "1920x1080",
                                             "3280x2464" };
static const int MODE_WH[NMODES][2] = { {1640,1232}, {1920,1080},
                                        {3280,2464} };

enum { SCALE_HALF = 0, SCALE_34, SCALE_FULL, NSCALES };
static const char *const SCALE_FIELD[NSCALES] = { "half", "34", "full" };
static const char *const SCALE_LABEL[NSCALES] = { "half", "3/4", "full" };

#define NSCALARS 9
static const char *const SCALARS[NSCALARS] = {
    "exposure", "analogue_gain", "digital_gain", "vertical_blanking",
    "test_pattern", "red_pixel_value", "green_red_pixel_value",
    "green_blue_pixel_value", "blue_pixel_value"
};
static const char *const SCALAR_FIELD[NSCALARS] = {
    "exposure", "gain", "dig", "vblank", "pattern", "pr", "pgr", "pgb", "pb"
};
static const long RANGEB_LO[NSCALARS] = { 4, 0, 256, 4, 0, 0, 0, 0, 0 };
static const long RANGEB_HI[NSCALARS] = { 1911, 232, 4095, 64303, 4,
                                          1023, 1023, 1023, 1023 };

enum { WEXP = 0, WGAIN, WDIG, WVB, WPATT, WPR, WPGR, WPGB, WPB };

/* ------------------------------------------------------------------ *
 * state (python S[] equivalent)
 * ------------------------------------------------------------------ */
#define CTLS_MAX 48

struct ctlent {
    char name[48];
    long val, lo, hi;
    int has_range;
};

typedef struct {
    long scal[NSCALARS];             /* exposure gain dig vb patt pr    */
    int hflip, vflip;                /* pgr pgb pb                      */
} want_t;

typedef struct {
    float x1, y1, x2, y2;            /* image (view) coords             */
    float sc;
} detbox_t;

typedef struct {
    pthread_mutex_t mu;

    /* camera session */
    char url[128];
    int connected, attempts;
    double last_try, last_ok, fps;
    char backend[16];
    char note[NOTE_MAX];

    /* frames */
    unsigned char *jpg;              /* view JPEG (owned)               */
    size_t jpg_len;
    double jpeg_t;
    unsigned long frames;

    /* detection results (native ORT; boxes in view coords)            */
    detbox_t dets[MAX_DETS];
    int ndets;
    double det_t;                    /* wall time of the last forward   */
    char meta_txt[4096];             /* JSON string                     */
    double last_infer_t;
    unsigned char *det_jpg;          /* annotated stream frame (owned)  */
    size_t det_len;
    double det_jpg_t;                /* annotated publish wall time     */

    /* view settings */
    int mode_idx, scale;
    int quality;
    int vw, vh;                      /* current view pixel dims         */
    int wb, gamma;
    double line_us;
    int reconf;

    /* sensor state */
    want_t want;
    struct ctlent ctl[CTLS_MAX];
    int nctl;

    /* live view pixels (capture writes, infer reads; under mu)        */
    unsigned char *view_bgr;
    int vw_alloc, vh_alloc;
    unsigned char *infer_snap;
    int snap_alloc;
    unsigned long view_seq;
    unsigned long infer_handled;
    unsigned char *det_bgr;          /* annotated stream copy buffer    */

    /* capture gating: capture only while someone watches live media   */
    volatile int pp_pid;             /* pumper pid or -1                */
    volatile int pp_suspended;       /* SIGSTOP'd when idle             */
    int stream_live;                 /* open consumers of either stream */
    int stream_raw;                  /* consumers of the raw stream     */
    unsigned long det_pubs;          /* diagnostic: annotated publishes */
    unsigned long raw_pubs;          /* diagnostic: raw publishes       */
    double activity_t;               /* last view-ish heartbeat         */

    /* AWB EMA (capture thread writes only)                            */
    double wb_ema[3];                /* B,G,R                           */

    /* turbojpeg handle                                                */
    tjhandle tj;
} state_t;

static state_t G;
static const char *opt_model = NULL;      /* -m override, else static */
static const char *opt_ortlib = NULL;     /* --ortlib override        */
static int opt_ortthreads = 4;            /* --ort-threads            */

static void g_lock(void)   { pthread_mutex_lock(&G.mu); }
static void g_unlock(void) { pthread_mutex_unlock(&G.mu); }

static void set_note(const char *fmt, ...)
{
    char t[NOTE_MAX];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(t, sizeof(t), fmt, ap);
    va_end(ap);
    g_lock();
    snprintf(G.note, sizeof(G.note), "%s", t);
    G.last_try = now_sec();
    g_unlock();
    logf_("%s", t);
}

static void set_meta_json(const char *txt)
{
    g_lock();
    snprintf(G.meta_txt, sizeof(G.meta_txt), "%s", txt);
    G.last_infer_t = now_sec();
    g_unlock();
}

static void set_meta_error(const char *msg)
{
    char t[512];
    char esc[420];
    const char *p = msg;
    char *o = esc;
    size_t room = sizeof(esc) - 2;
    while (*p && room > 4) {
        char c = *p++;
        if (c == '"' || c == '\\') {
            *o++ = '\\';
            *o++ = c;
            room -= 2;
        } else if ((unsigned char)c >= 0x20) {
            *o++ = c;
            room--;
        }
    }
    *o = 0;
    snprintf(t, sizeof(t), "{\"n\":null,\"infer_ms\":null,\"err\":\"%s\"}",
             esc);
    set_meta_json(t);
}

static void publish_raw_jpg_locked(const unsigned char *data, size_t len)
{
    unsigned char *cp = xmalloc(len ? len : 1);
    memcpy(cp, data, len);
    free(G.jpg);
    G.jpg = cp;
    G.jpg_len = len;
    G.jpeg_t = now_sec();
    G.frames++;
    G.raw_pubs++;
}

static void publish_det_jpg_locked(const unsigned char *data, size_t len)
{
    unsigned char *cp = xmalloc(len ? len : 1);
    memcpy(cp, data, len);
    free(G.det_jpg);
    G.det_jpg = cp;
    G.det_len = len;
    G.det_jpg_t = now_sec();
    G.det_pubs++;
}

static int ctl_index(const char *name)
{
    for (int i = 0; i < G.nctl; i++)
        if (!strcmp(G.ctl[i].name, name)) return i;
    return -1;
}

static long clamp_name(const char *name, long v)
{
    int i = ctl_index(name);
    if (i >= 0 && G.ctl[i].has_range) {
        long lo = G.ctl[i].lo, hi = G.ctl[i].hi;
        if (lo > hi) { long t = lo; lo = hi; hi = t; }
        return v < lo ? lo : (v > hi ? hi : v);
    }
    for (int k = 0; k < NSCALARS; k++)
        if (!strcmp(SCALARS[k], name))
            return v < RANGEB_LO[k] ? RANGEB_LO[k] :
                   (v > RANGEB_HI[k] ? RANGEB_HI[k] : v);
    return v;
}

/* ------------------------------------------------------------------ *
 * v4l2 / media tool plumbing (spawned per session or per apply)
 * ------------------------------------------------------------------ */
static int run_tool(char **argv, char **out, int timeout_s)
{
    int pfd[2];
    if (pipe(pfd) != 0) return 99;
    pid_t p = fork();
    if (p < 0) { close(pfd[0]); close(pfd[1]); return 99; }
    if (p == 0) {
        dup2(pfd[1], 1);
        dup2(pfd[1], 2);
        close(pfd[0]);
        close(pfd[1]);
        execv(argv[0], argv);
        _exit(127);
    }
    close(pfd[1]);
    sbuf sb;
    sb_init(&sb);
    struct pollfd pf = { pfd[0], POLLIN, 0 };
    time_t deadline = time(NULL) + timeout_s;
    char chunk[4096];
    for (;;) {
        int d = (int)(deadline - time(NULL));
        if (d <= 0) { kill(p, SIGKILL); break; }
        if (poll(&pf, 1, d * 1000) <= 0) break;
        ssize_t n = read(pfd[0], chunk, sizeof(chunk));
        if (n <= 0) break;
        sb_appn(&sb, chunk, (size_t)n);
        if (sb.len > 1024u * 1024) break;
    }
    close(pfd[0]);
    int st = 0;
    waitpid(p, &st, 0);
    int rc = WIFEXITED(st) ? WEXITSTATUS(st) : 99;
    if (out) *out = sb.b;
    else sb_free(&sb);
    return rc;
}

/* parse `v4l2-ctl -l`:  name 0x.. (type): min=.. max=.. value=.. */
static void parse_ctl_list(char *txt)
{
    g_lock();
    G.nctl = 0;
    char *save = NULL;
    for (char *line = strtok_r(txt, "\n", &save); line && G.nctl < CTLS_MAX;
         line = strtok_r(NULL, "\n", &save)) {
        char name[48] = "";
        if (sscanf(line, " %47[a-zA-Z0-9_]", name) == 1 && name[0] &&
            strchr(line, ':') && strstr(line, "0x") &&
            strstr(line, "value=")) {
            long val = 0;
            if (sscanf(strstr(line, "value="), " value=%ld", &val) == 1) {
                struct ctlent *e = &G.ctl[G.nctl];
                snprintf(e->name, sizeof(e->name), "%s", name);
                e->val = val;
                e->has_range = 0;
                if (strstr(line, "min=") && strstr(line, "max=")) {
                    long lo = 0, hi = 0;
                    int a = sscanf(strstr(line, "min="), " min=%ld", &lo);
                    int b = sscanf(strstr(line, "max="), " max=%ld", &hi);
                    if (a == 1 && b == 1) {
                        e->lo = lo;
                        e->hi = hi;
                        e->has_range = 1;
                    }
                }
                G.nctl++;
            }
        }
    }
    g_unlock();
}

static void get_ctl(void)
{
    char *argv[] = { (char *)V4L_BIN, (char *)"-d",
                     (char *)SENSOR_DEV, (char *)"-l", NULL };
    char *out = NULL;
    if (run_tool(argv, &out, 10) == 0 && out) parse_ctl_list(out);
    free(out);
}

/* one combined sensor apply: every name=value gets its own -c arg */
static int sensor_set(const char *const *names, const long *vals, int n)
{
    if (n <= 0) return 0;
    char *argv[10 + NSCALARS * 2];
    char cstr[NSCALARS][96];
    int idx = 0;
    argv[idx++] = (char *)V4L_BIN;
    argv[idx++] = (char *)"-d";
    argv[idx++] = (char *)SENSOR_DEV;
    for (int i = 0; i < n; i++) {
        snprintf(cstr[i], sizeof(cstr[i]), "%s=%ld", names[i], vals[i]);
        argv[idx++] = (char *)"-c";
        argv[idx++] = cstr[i];
    }
    argv[idx] = NULL;
    char *out = NULL;
    int rc = run_tool(argv, &out, 20);
    if (rc != 0)
        set_note("sensor apply rejected: %.140s", out ? out : "(no out)");
    free(out);
    return rc;
}

static void csi_configure(const char *dev, int W, int H)
{
    char wh[16];
    snprintf(wh, sizeof(wh), "%dx%d", W, H);
    char f[4][160];
    snprintf(f[0], sizeof(f[0]),
             "\"imx219 4-0010\":0 [fmt:SRGGB10_1X10/%s field:none]", wh);
    snprintf(f[1], sizeof(f[1]),
             "\"cdns_csi2rx.30101000.csi-bridge\":0 [fmt:SRGGB10_1X10/%s "
             "field:none]", wh);
    snprintf(f[2], sizeof(f[2]),
             "\"30102000.ticsi2rx\":0 [fmt:SRGGB10_1X10/%s field:none]", wh);
    snprintf(f[3], sizeof(f[3]),
             "--set-fmt-video=width=%d,height=%d,pixelformat=RG10", W, H);

    char *argv_imx[]  = { (char *)MEDIA_BIN, (char *)"-d", (char *)MEDIA_DEV,
                          (char *)"-V", f[0], NULL };
    char *argv_csi[]  = { (char *)MEDIA_BIN, (char *)"-d", (char *)MEDIA_DEV,
                          (char *)"-V", f[1], NULL };
    char *argv_tics[] = { (char *)MEDIA_BIN, (char *)"-d", (char *)MEDIA_DEV,
                          (char *)"-V", f[2], NULL };
    char *argv_fmt[]  = { (char *)V4L_BIN, (char *)"-d", (char *)dev,
                          f[3], NULL };
    char **cmds[4] = { argv_imx, argv_csi, argv_tics, argv_fmt };
    const char *lbl[4] = { "media imx219", "media csi2rx", "media ticsi2rx",
                           "set-fmt" };
    for (int i = 0; i < 4; i++) {
        char *out = NULL;
        int rc = run_tool(cmds[i], &out, 15);
        if (rc != 0) {
            set_note("csi config failed (%s rc %d): %.90s", lbl[i], rc,
                     out ? out : "");
            free(out);
            return;          /* caller surfaces the failure via session */
        }
        free(out);
    }
}

/* apply every want scalar + flips in ONE pre-stream sensor call, then
 * read back kernel state (python apply_want) */
static int apply_want(int W)
{
    get_ctl();
    const char *names[NSCALARS + 2];
    long vals[NSCALARS + 2];
    int n = 0;
    for (int i = 0; i < NSCALARS; i++) {
        names[n] = SCALARS[i];
        vals[n] = clamp_name(SCALARS[i], G.want.scal[i]);
        n++;
    }
    names[n] = "horizontal_flip";
    vals[n] = G.want.hflip;
    n++;
    names[n] = "vertical_flip";
    vals[n] = G.want.vflip;
    n++;
    int rc = sensor_set(names, vals, n);
    if (rc != 0) return rc;

    get_ctl();
    long hb = 0, pr = 0;
    int ihb = ctl_index("horizontal_blanking");
    int ipr = ctl_index("pixel_rate");
    if (ihb >= 0) hb = G.ctl[ihb].val;
    if (ipr >= 0) pr = G.ctl[ipr].val;
    g_lock();
    for (int i = 0; i < NSCALARS; i++) {
        int ic = ctl_index(SCALARS[i]);
        G.want.scal[i] = ic >= 0 ? G.ctl[ic].val :
                         clamp_name(SCALARS[i], G.want.scal[i]);
    }
    if (pr > 0)
        G.line_us = (double)(W + hb) / (double)pr * 1e6;
    g_unlock();
    return 0;
}

/* ------------------------------------------------------------------ *
 * Bayer debayering (BG/GB/RG/GR): 10-bit u16 -> 8-bit BGR
 *
 * Quad mapping (p00=(even,even) p01=(odd,even) p10=(even,odd)
 * p11=(odd,odd)):
 *   BG: B=p00 G=p01,p10 R=p11      RG: R=p00 G=p01,p10 B=p11
 *   GB: G=p00,p11 R=p01 B=p10      GR: G=p00,p11 B=p01 R=p10
 * ------------------------------------------------------------------ */
enum { PH_BG = 0, PH_GB, PH_RG, PH_GR, NPHASE };
static const char *const PH_NAME[NPHASE] = { "BG", "GB", "RG", "GR" };

/* R site parity (x,y) per phase; B is always the mirrored (1-rx,1-ry) */
static const int PH_RX[NPHASE] = { 1, 1, 0, 0 };
static const int PH_RY[NPHASE] = { 1, 0, 0, 1 };

typedef struct {
    const unsigned short *src;
    unsigned char *dst;              /* half-view BGR (hw*hb*3)         */
    int W, H, phase;
    int hw, hb;
} binctx_t;

static void bin_quad_rows(void *ap, int y0, int y1)
{
    binctx_t *c = (binctx_t *)ap;
    const unsigned short *src = c->src;
    unsigned char *dst = c->dst;
    const int W = c->W, hw = c->hw, ph = c->phase;
    for (int y = y0; y < y1; y++) {
        const unsigned short *r0 = src + (size_t)(2 * y) * W;
        const unsigned short *r1 = r0 + W;
        unsigned char *d = dst + (size_t)y * hw * 3;
        for (int x = 0; x < hw; x++) {
            int x2 = 2 * x;
            unsigned v00 = r0[x2], v01 = r0[x2 + 1];
            unsigned v10 = r1[x2], v11 = r1[x2 + 1];
            unsigned char b, g, r;
            switch (ph) {
            case PH_BG: b = (unsigned char)(v00 >> 2);
                        r = (unsigned char)(v11 >> 2);
                        g = (unsigned char)((v01 + v10) >> 3);
                        break;
            case PH_RG: r = (unsigned char)(v00 >> 2);
                        b = (unsigned char)(v11 >> 2);
                        g = (unsigned char)((v01 + v10) >> 3);
                        break;
            case PH_GB: g = (unsigned char)((v00 + v11) >> 3);
                        r = (unsigned char)(v01 >> 2);
                        b = (unsigned char)(v10 >> 2);
                        break;
            default:    g = (unsigned char)((v00 + v11) >> 3);   /* GR */
                        b = (unsigned char)(v01 >> 2);
                        r = (unsigned char)(v10 >> 2);
                        break;
            }
            d[0] = b; d[1] = g; d[2] = r;
            d += 3;
        }
    }
}

typedef struct {
    const unsigned short *src;
    unsigned char *dst;              /* full-res BGR (W*H*3)            */
    int W, H, phase;
} dbctx_t;

static void dbay_bilinear_rows(void *ap, int y0, int y1)
{
    dbctx_t *c = (dbctx_t *)ap;
    const unsigned short *src = c->src;
    unsigned char *dst = c->dst;
    const int W = c->W, H = c->H;
    const int rx = PH_RX[c->phase], ry = PH_RY[c->phase];
    const int bx = 1 - rx, by = 1 - ry;
    for (int y = y0; y < y1; y++) {
        int ym = y > 0 ? y - 1 : 1;
        int yq = y < H - 1 ? y + 1 : H - 2;
        const unsigned short *r_up = src + (size_t)ym * W;
        const unsigned short *r_dn = src + (size_t)yq * W;
        const unsigned short *r_ct = src + (size_t)y * W;
        unsigned char *d = dst + (size_t)y * W * 3;
        for (int x = 0; x < W; x++) {
            int xm = x > 0 ? x - 1 : 1;
            int xq = x < W - 1 ? x + 1 : W - 2;
            unsigned v_ct = r_ct[x];
            unsigned v_up = r_up[x], v_dn = r_dn[x];
            unsigned v_lf = r_ct[xm], v_rt = r_ct[xq];
            unsigned ul = r_up[xm], ur = r_up[xq];
            unsigned dl = r_dn[xm], dr = r_dn[xq];
            unsigned char b, g, r;
            if (((x & 1) == rx) && ((y & 1) == ry)) {         /* R site  */
                r = (unsigned char)(v_ct >> 2);
                g = (unsigned char)((v_up + v_dn + v_lf + v_rt) >> 4);
                b = (unsigned char)((ul + ur + dl + dr) >> 4);
            } else if (((x & 1) == bx) && ((y & 1) == by)) {  /* B site  */
                b = (unsigned char)(v_ct >> 2);
                g = (unsigned char)((v_up + v_dn + v_lf + v_rt) >> 4);
                r = (unsigned char)((ul + ur + dl + dr) >> 4);
            } else if (((x & 1) != rx) && ((y & 1) == ry)) {  /* G: R-hz */
                g = (unsigned char)(v_ct >> 2);
                r = (unsigned char)((v_lf + v_rt) >> 3);
                b = (unsigned char)((v_up + v_dn) >> 3);
            } else {                                          /* G: R-vt */
                g = (unsigned char)(v_ct >> 2);
                r = (unsigned char)((v_up + v_dn) >> 3);
                b = (unsigned char)((v_lf + v_rt) >> 3);
            }
            d[0] = b; d[1] = g; d[2] = r;
            d += 3;
        }
    }
}

/* bilinear upsample half->view (used for the "3/4" view)              */
typedef struct {
    const unsigned char *src;        /* hw*hb*3                         */
    unsigned char *dst;              /* vw*vh*3                         */
    int hw, hb, vw, vh;
} rsctx_t;

static void resize_rows(void *ap, int y0, int y1)
{
    rsctx_t *c = (rsctx_t *)ap;
    int vwm = c->vw > 1 ? c->vw - 1 : 1;
    int vhm = c->vh > 1 ? c->vh - 1 : 1;
    for (int y = y0; y < y1; y++) {
        int sy = (y * (c->hb - 1)) / vhm;
        int wy = ((y * (c->hb - 1)) % vhm) * 256 / vhm;
        int sy2 = sy < c->hb - 1 ? sy + 1 : sy;
        const unsigned char *r0 = c->src + (size_t)sy * c->hw * 3;
        const unsigned char *r1 = c->src + (size_t)sy2 * c->hw * 3;
        unsigned char *d = c->dst + (size_t)y * c->vw * 3;
        for (int x = 0; x < c->vw; x++) {
            int sx = (x * (c->hw - 1)) / vwm;
            int wx = ((x * (c->hw - 1)) % vwm) * 256 / vwm;
            int sx2 = sx < c->hw - 1 ? sx + 1 : sx;
            const unsigned char *p00 = r0 + (size_t)sx * 3;
            const unsigned char *p01 = r0 + (size_t)sx2 * 3;
            const unsigned char *p10 = r1 + (size_t)sx * 3;
            const unsigned char *p11 = r1 + (size_t)sx2 * 3;
            for (int k = 0; k < 3; k++) {
                int top = p00[k] * (256 - wx) + p01[k] * wx;
                int bot = p10[k] * (256 - wx) + p11[k] * wx;
                d[k] = (unsigned char)((top * (256 - wy) + bot * wy) >> 16);
            }
            d += 3;
        }
    }
}

/* row-split runner (parallel userspace work across the A53 cores)     */
typedef struct {
    void (*fn)(void *, int, int);
    void *ctx;
    int y0, y1;
} rjob_t;

static void *rjob_main(void *arg)
{
    rjob_t *j = (rjob_t *)arg;
    j->fn(j->ctx, j->y0, j->y1);
    return NULL;
}

static void rows_split(int H, void (*fn)(void *, int, int), void *ctx, int nt)
{
    if (nt < 2 || H < 128) { fn(ctx, 0, H); return; }
    rjob_t jobs[4];
    pthread_t th[4];
    int njob = 0, y = 0, nthreaded = 0;
    int chunk = (H + nt - 1) / nt;
    for (int i = 0; i < nt; i++) {
        if (y >= H) break;
        int e = y + chunk;
        if (e > H) e = H;
        jobs[njob].fn = fn;
        jobs[njob].ctx = ctx;
        jobs[njob].y0 = y;
        jobs[njob].y1 = e;
        njob++;
        y = e;
    }
    for (int i = 1; i < njob; i++) {
        if (pthread_create(&th[nthreaded], NULL, rjob_main, &jobs[i]) == 0)
            nthreaded++;
        else
            fn(ctx, jobs[i].y0, jobs[i].y1);     /* fallback: inline   */
    }
    fn(ctx, jobs[0].y0, jobs[0].y1);
    for (int i = 0; i < nthreaded; i++) pthread_join(th[i], NULL);
}

/* ------------------------------------------------------------------ *
 * color: gray-world AWB EMA + gamma-0.55 LUT (python formulas)
 * ------------------------------------------------------------------ */
static void awb_update(unsigned char *bgr, int vw, int vh)
{
    long sum[3] = { 0, 0, 0 };
    long cnt = 0;
    for (int y = 0; y < vh; y += 16) {
        const unsigned char *row = bgr + (size_t)y * vw * 3;
        for (int x = 0; x < vw; x += 16) {
            sum[0] += row[(size_t)x * 3 + 0];
            sum[1] += row[(size_t)x * 3 + 1];
            sum[2] += row[(size_t)x * 3 + 2];
            cnt++;
        }
    }
    if (!cnt) return;
    double m[3] = { (double)sum[0] / cnt, (double)sum[1] / cnt,
                    (double)sum[2] / cnt };
    if (m[1] <= 8.0 || m[0] <= 4.0 || m[2] <= 4.0) return;
    for (int c = 0; c < 3; c++) {
        double t = m[1] / (m[c] > 1.0 ? m[c] : 1.0);
        if (t < 0.6) t = 0.6;
        if (t > 2.2) t = 2.2;
        G.wb_ema[c] = G.wb_ema[c] * 0.95 + t * 0.05;
    }
}

static void color_lut_apply(unsigned char *bgr, int vw, int vh,
                            int wb_on, int gam_on)
{
    if (!wb_on && !gam_on) return;
    unsigned char lut[3][256];
    for (int c = 0; c < 3; c++) {
        double g = wb_on ? G.wb_ema[c] : 1.0;
        for (int i = 0; i < 256; i++) {
            double v = i;
            if (gam_on)
                v = pow((double)i / 255.0, 0.55) * 255.0;
            double o = v * g + 0.5;
            lut[c][i] = (unsigned char)(o < 255.0 ? o : 255.0);
        }
    }
    size_t n = (size_t)vw * vh * 3;
    for (size_t i = 0; i + 2 < n; i += 3) {
        bgr[i + 0] = lut[0][bgr[i + 0]];
        bgr[i + 1] = lut[1][bgr[i + 1]];
        bgr[i + 2] = lut[2][bgr[i + 2]];
    }
}

/* JPEG encode via turbojpeg; returns malloc'd buffer (caller frees)  */
static unsigned char *encode_jpg(const unsigned char *bgr, int vw, int vh,
                                 int q, size_t *out_len)
{
    unsigned char *tb = NULL;
    unsigned long tsz = 0;
    if (q < 30) q = 30;
    if (q > 95) q = 95;
    int rc = tjCompress2(G.tj, bgr, vw, vw * 3, vh, TJPF_BGR, &tb, &tsz,
                         TJSAMP_420J, q, 0);
    if (rc != 0 || !tb || !tsz) {
        char *es = tjGetErrorStr2(G.tj);
        logf_("jpeg encode failed: %s", es ? es : "?");
        if (tb) tjFree(tb);
        return NULL;
    }
    unsigned char *cp = xmalloc(tsz);
    memcpy(cp, tb, tsz);
    tjFree(tb);
    *out_len = tsz;
    return cp;
}

/* ------------------------------------------------------------------ *
 * frame pump (fork/exec frame_pumper -> raw pipe)
 * ------------------------------------------------------------------ */
static void pumper_kill(pid_t *p, int *rfd)
{
    if (*rfd >= 0) { close(*rfd); *rfd = -1; }
    if (*p > 0) {
        kill(*p, SIGTERM);
        for (int i = 0; i < 30; i++) {
            int st;
            pid_t r = waitpid(*p, &st, WNOHANG);
            if (r == *p || (r < 0 && errno == ECHILD)) break;
            usleep(100 * 1000);
            if (i == 20) kill(*p, SIGKILL);
        }
        *p = 0;
    }
}

/* opt-B core partition: the copier gets exactly one core of its own so the
 * ORT forward pass owns the others (measured: 1 copier thread still
 * delivers ~14 fps >= the ~12 fps live contended rate).                  */
#define PUMPER_CORE 3

static int pumper_spawn(pid_t *p, int *rfd, const char *dev, int W, int H)
{
    const char *cands[] = { HOME_BIRDCPP "/frame_pumper",
                            HOME_ROOTBIRD "/frame_pumper",
                            "/home/root/camkit/frame_pumper" };
    const char *exe = NULL;
    for (int i = 0; i < 3; i++)
        if (access(cands[i], X_OK) == 0) { exe = cands[i]; break; }
    if (!exe) {
        set_note("frame_pumper binary not deployed");
        return -1;
    }
    int pp[2];
    if (pipe(pp) != 0) return -1;
    pid_t pid = fork();
    if (pid < 0) { close(pp[0]); close(pp[1]); return -1; }
    if (pid == 0) {
        prctl(PR_SET_PDEATHSIG, 9);      /* belt: pumper also sets it   */
        if (getppid() == 1) _exit(0);
        /* opt-B: pin the copier to its own core + low priority so the
         * inference forward pass keeps the remaining cores.             */
        {
            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(PUMPER_CORE, &cs);
            if (sched_setaffinity(getpid(), sizeof(cs), &cs) != 0)
                logf_("pumper: affinity pin failed (err %d)", errno);
            setpriority(PRIO_PROCESS, 0, 10);
        }
        dup2(pp[1], 1);
        close(pp[0]);
        close(pp[1]);
        char ws[16], hs[16], ts[16];
        snprintf(ws, sizeof(ws), "%d", W);
        snprintf(hs, sizeof(hs), "%d", H);
        snprintf(ts, sizeof(ts), "1");   /* 1 copier thread on PUMPER_CORE */
        char *argv[] = { (char *)exe, (char *)dev, ws, hs, ts, NULL };
        execv(exe, argv);
        _exit(127);
    }
    close(pp[1]);
    int r = pp[0];
    int fl = fcntl(r, F_GETFL, 0);
    fcntl(r, F_SETFL, fl | O_NONBLOCK);
    fcntl(r, F_SETPIPE_SZ_L, PIPE_WIDEN);
    *p = pid;
    *rfd = r;
    return 0;
}

/* one complete raw frame from the pipe: 1 ok / 0 timeout / -1 closed */
static int read_frame_pipe(int rfd, unsigned short *buf, size_t fb,
                           int ms)
{
    size_t done = 0;
    unsigned char *p = (unsigned char *)buf;
    struct pollfd pf;
    while (done < fb) {
        pf.fd = rfd;
        pf.events = POLLIN;
        int pr = poll(&pf, 1, ms);
        if (pr <= 0) return 0;
        ssize_t n = read(rfd, p + done, fb - done);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            return -1;
        }
        if (n == 0) return -1;
        done += (size_t)n;
    }
    return 1;
}

/* ------------------------------------------------------------------ *
 * phase verification (solid-red pattern; R/B/G quad-site scoring)
 * ------------------------------------------------------------------ */
static void sensor_set_pattern_red(void)
{
    const char *names[5] = { "test_pattern", "red_pixel_value",
        "green_red_pixel_value", "green_blue_pixel_value",
        "blue_pixel_value" };
    long vals[5] = { 2, 1023, 0, 0, 0 };
    sensor_set(names, vals, 5);
}

static void sensor_restore_pattern(void)
{
    const char *names[5] = { "test_pattern", "red_pixel_value",
        "green_red_pixel_value", "green_blue_pixel_value",
        "blue_pixel_value" };
    long vals[5];
    vals[0] = clamp_name("test_pattern", G.want.scal[WPATT]);
    vals[1] = clamp_name("red_pixel_value", G.want.scal[WPR]);
    vals[2] = clamp_name("green_red_pixel_value", G.want.scal[WPGR]);
    vals[3] = clamp_name("green_blue_pixel_value", G.want.scal[WPGB]);
    vals[4] = clamp_name("blue_pixel_value", G.want.scal[WPB]);
    sensor_set(names, vals, 5);
}

/* R / B site indices per phase; the other two lattice classes are G */
static const int PH_RSI[NPHASE] = { 3, 1, 0, 2 };   /* p11 p01 p00 p10  */
static const int PH_BSI[NPHASE] = { 0, 2, 3, 1 };   /* p00 p10 p11 p01  */

static int verify_phase(int W, int H, int rfd, unsigned short *buf)
{
    size_t fb = (size_t)W * (size_t)H * 2;
    double best = 0.0;
    int bestph = PH_BG;
    sensor_set_pattern_red();
    for (int f = 0; f < 2; f++) {
        if (read_frame_pipe(rfd, buf, fb, 2500) != 1) break;
        double acc[4] = { 0, 0, 0, 0 };
        long cnt[4] = { 0, 0, 0, 0 };
        for (int y = 0; y < H; y++) {
            const unsigned short *row = buf + (size_t)y * W;
            for (int x = 0; x < W; x++) {
                int site = ((y & 1) << 1) | (x & 1);
                acc[site] += (double)(row[x] >> 2);
                cnt[site]++;
            }
        }
        double m[4];
        for (int s = 0; s < 4; s++)
            m[s] = cnt[s] ? acc[s] / (double)cnt[s] : 0.0;
        for (int ph = 0; ph < NPHASE; ph++) {
            int g1i = -1, g2i = -1;
            for (int s = 0; s < 4; s++) {
                if (s == PH_RSI[ph] || s == PH_BSI[ph]) continue;
                g2i = g1i;
                g1i = s;
                if (g2i < 0) g2i = s;    /* until both G sites seen    */
            }
            double rm = m[PH_RSI[ph]];
            double bm = m[PH_BSI[ph]];
            double gm = 0.5 * (m[g1i] + m[g2i]);
            double den = 0.5 * (bm + gm);
            if (den < 1.0) den = 1.0;
            double sc = rm / den;
            if (sc > best) { best = sc; bestph = ph; }
        }
    }
    sensor_restore_pattern();
    return bestph;
}

/* ------------------------------------------------------------------ *
 * det drawing (builtin 5x7 font: same visual layout as cv2.putText
 * without HERXHEY - black label plate, green box + glyph pixels)
 * ------------------------------------------------------------------ */
#define FONTW 5
#define FONTH 7
static const unsigned char FONT57[][FONTH] = {
    /* ' ' */ {0, 0, 0, 0, 0, 0, 0},
    /* '.' */ {0, 0, 0, 0, 0, 0b00100, 0b00100},
    /* '0' */ {0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110},
    /* '1' */ {0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110},
    /* '2' */ {0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111},
    /* '3' */ {0b11110, 0b00010, 0b00010, 0b01110, 0b00010, 0b00010, 0b11110},
    /* '4' */ {0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010},
    /* '5' */ {0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110},
    /* '6' */ {0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110},
    /* '7' */ {0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000},
    /* '8' */ {0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110},
    /* '9' */ {0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100},
    /* 'b' */ {0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110},
    /* 'd' */ {0b00011, 0b00010, 0b00010, 0b01110, 0b10001, 0b10001, 0b01111},
    /* 'i' */ {0b01110, 0b00000, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110},
    /* 'r' */ {0b11100, 0b10010, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000},
};

static int font_index(char c)
{
    switch (c) {
    case ' ': return 0; case '.': return 1;
    case '0': return 2; case '1': return 3; case '2': return 4;
    case '3': return 5; case '4': return 6; case '5': return 7;
    case '6': return 8; case '7': return 9; case '8': return 10;
    case '9': return 11; case 'b': return 12; case 'd': return 13;
    case 'i': return 14; case 'r': return 15;
    default: return 0;
    }
}

static void set_px(unsigned char *bgr, int w, int h, int x, int y,
                   unsigned char B, unsigned char G, unsigned char R)
{
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    size_t o = ((size_t)y * w + x) * 3;
    bgr[o + 0] = B;
    bgr[o + 1] = G;
    bgr[o + 2] = R;
}

static void draw_hline(unsigned char *bgr, int w, int h,
                       int x0, int x1, int y, int t, unsigned char B,
                       unsigned char G, unsigned char R)
{
    for (int j = 0; j < t; j++)
        for (int x = x0; x <= x1; x++)
            set_px(bgr, w, h, x, y + j, B, G, R);
}

static void draw_vline(unsigned char *bgr, int w, int h,
                       int y0, int y1, int x, int t, unsigned char B,
                       unsigned char G, unsigned char R)
{
    for (int j = 0; j < t; j++)
        for (int y = y0; y <= y1; y++)
            set_px(bgr, w, h, x + j, y, B, G, R);
}

static void draw_glyph(unsigned char *bgr, int w, int h,
                       char c, int x, int baseline, int s)
{
    const unsigned char *g = FONT57[font_index(c)];
    for (int r = 0; r < FONTH; r++)
        for (int q = 0; q < FONTW; q++)
            if (g[r] & (1 << (FONTW - 1 - q)))
                for (int py = 0; py < s; py++)
                    for (int px = 0; px < s; px++)
                        set_px(bgr, w, h, x + q * s + px,
                               baseline - (FONTH - r) * s + py,
                               0, 255, 0);
}

static void draw_text_label(unsigned char *bgr, int w, int h,
                            const char *lit, int x, int baseline, int s)
{
    int cx = x;
    for (const char *p = lit; *p; p++) {
        draw_glyph(bgr, w, h, *p, cx, baseline, s);
        cx += (FONTW + 1) * s;
    }
}

static void text_size_label(const char *lit, int s, int *tw, int *tth)
{
    int n = (int)strlen(lit);
    *tw = n > 0 ? (n * (FONTW + 1) - 1) * s : 0;
    *tth = FONTH * s;
}

/* one det: black plate + green box + green label (python draw style) */
static void draw_one_det(unsigned char *bgr, int vw, int vh,
                         const detbox_t *b, int th)
{
    int xi = (int)b->x1 < 0 ? 0 : (int)b->x1;
    int yi = (int)b->y1 < 0 ? 0 : (int)b->y1;
    int xa = (int)b->x2;
    int ya = (int)b->y2;
    if (xa >= vw) xa = vw - 1;
    if (ya >= vh) ya = vh - 1;
    if (xa <= xi || ya <= yi) return;

    int s = vh >= 1000 ? 3 : 2;
    char label[24];
    snprintf(label, sizeof(label), "bird %.2f", (double)b->sc);
    int tw, tth;
    text_size_label(label, s, &tw, &tth);
    int ly = yi - tth - 8;
    if (ly < 0) ly = yi + tth + 10;

    /* filled black plate, inclusive corners (cv2 -1 thickness) */
    for (int y = ly - tth - 6; y <= ly + 6; y++)
        if (y >= 0 && y < vh)
            draw_hline(bgr, vw, vh, xi, xi + tw + 8, y, 1, 0, 0, 0);
    /* green box: four side strips thickness th */
    draw_hline(bgr, vw, vh, xi, xa, yi, th, 0, 255, 0);
    draw_hline(bgr, vw, vh, xi, xa, ya - th + 1, th, 0, 255, 0);
    draw_vline(bgr, vw, vh, yi, ya, xi, th, 0, 255, 0);
    draw_vline(bgr, vw, vh, yi, ya, xa - th + 1, th, 0, 255, 0);
    /* green label bottom at ly (putText lower-left origin) */
    draw_text_label(bgr, vw, vh, label, xi + 4, ly, s);
}

static void draw_dets(unsigned char *bgr, int vw, int vh,
                      const detbox_t *boxes, int n)
{
    int th = vh / 240;
    if (th < 3) th = 3;
    for (int i = 0; i < n; i++)
        draw_one_det(bgr, vw, vh, &boxes[i], th);
}

static unsigned char *encode_jpg_det(const unsigned char *bgr, int vw, int vh,
                                     int q, size_t *out_len)
{
    static double last_null_log;
    g_lock();
    int ndets = G.ndets;
    double dt = G.det_t;
    detbox_t boxes[MAX_DETS];
    if (ndets > MAX_DETS) ndets = MAX_DETS;
    for (int i = 0; i < ndets; i++) boxes[i] = G.dets[i];
    unsigned char *db = G.det_bgr;
    int aw = G.vw_alloc, ah = G.vh_alloc;
    g_unlock();
    if (ndets > 0 && (now_sec() - dt) > DET_IDLE_S) ndets = 0;
    if (!db || vw != aw || vh != ah) {
        /* self-heal: (re)allocate the annotated canvas for these dims  */
        g_lock();
        if (!G.det_bgr || G.vw_alloc != vw || G.vh_alloc != vh) {
            free(G.det_bgr);
            G.det_bgr = (unsigned char *)xmalloc((size_t)vw * vh * 3);
            logf_("det canvas allocated %dx%d (was %dx%d, db=%p)", vw, vh,
                  aw, ah, (void *)db);
        }
        db = G.det_bgr;
        g_unlock();
        if (!db) {
            double nw = now_sec();
            if (nw - last_null_log > 2.0) {
                last_null_log = nw;
                logf_("det-encode NULL: canvas alloc failed");
            }
            return NULL;
        }
    }
    memcpy(db, bgr, (size_t)vw * (size_t)vh * 3);
    if (ndets > 0) draw_dets(db, vw, vh, boxes, ndets);
    unsigned char *jpg = encode_jpg(db, vw, vh, q, out_len);
    if (!jpg) {
        double nw = now_sec();
        if (nw - last_null_log > 2.0) {
            last_null_log = nw;
            logf_("det-encode: turbojpeg returned NULL");
        }
    }
    return jpg;
}

/* ------------------------------------------------------------------ *
 * view geometry + per-frame processing pipeline
 * ------------------------------------------------------------------ */
static void view_dims(int mode_idx, int scale, int *ow, int *oh)
{
    int W = MODE_WH[mode_idx][0], H = MODE_WH[mode_idx][1];
    double f = scale == SCALE_HALF ? 0.5 :
               (scale == SCALE_34 ? 0.75 : 1.0);
    *ow = (int)((double)W * f) / 2 * 2;
    *oh = (int)((double)H * f) / 2 * 2;
    if (*ow < 2) *ow = 2;
    if (*oh < 2) *oh = 2;
}

static void ensure_view_bufs_locked(int mode_idx, int scale)
{
    static double last_entry_log;
    int vw, vh;
    view_dims(mode_idx, scale, &vw, &vh);
    if (G.view_bgr && G.vw_alloc == vw && G.vh_alloc == vh) {
        double nw = now_sec();
        if (nw - last_entry_log > 5.0) {
            last_entry_log = nw;
            logf_("bufs-ok: view=%p det=%p snap=%p vw/vh=%d/%d",
                  (void *)G.view_bgr, (void *)G.det_bgr,
                  (void *)G.infer_snap, vw, vh);
        }
        return;
    }
    free(G.view_bgr);
    G.view_bgr = (unsigned char *)xmalloc((size_t)vw * vh * 3);
    G.vw_alloc = vw;
    G.vh_alloc = vh;
    G.ndets = 0;                     /* boxes were in old-view coords   */
    G.det_t = 0.0;
    free(G.det_bgr);
    G.det_bgr = (unsigned char *)xmalloc((size_t)vw * vh * 3);
    logf_("bufs-alloc: view=%p det=%p %dx%d", (void *)G.view_bgr,
          (void *)G.det_bgr, vw, vh);
}

static int process_frame(int mode_idx, int scale, int W, int H, int phase,
                         unsigned short *raw, unsigned char *mid)
{
    int vw, vh;
    view_dims(mode_idx, scale, &vw, &vh);
    int hw = W / 2, hb = H / 2;
    ensure_view_bufs_locked(mode_idx, scale);

    g_lock();
    unsigned char *view = G.view_bgr;
    g_unlock();
    if (!view) return -1;

    if (scale == SCALE_HALF) {
        binctx_t bc = { raw, view, W, H, phase, hw, hb };
        bin_quad_rows(&bc, 0, hb);
    } else if (scale == SCALE_34) {
        binctx_t bc = { raw, mid, W, H, phase, hw, hb };
        bin_quad_rows(&bc, 0, hb);
        rsctx_t rc2 = { mid, view, hw, hb, vw, vh };
        rows_split(vh, resize_rows, &rc2, 3);
    } else {
        dbctx_t dc = { raw, view, W, H, phase };
        rows_split(H, dbay_bilinear_rows, &dc, 4);
    }

    /* capture gating: encode only while someone watches live media.   */
    static double last_gate_log;
    g_lock();
    int wb_on = G.wb != 0, gam_on = G.gamma != 0, q = G.quality;
    int want_raw = G.stream_raw > 0, want_det = G.stream_live > 0;
    int sl = G.stream_live, sr = G.stream_raw;
    g_unlock();
    if ((want_raw || want_det) && (now_sec() - last_gate_log) > 3.0) {
        last_gate_log = now_sec();
        logf_("gate: live=%d raw=%d want_raw=%d want_det=%d",
              sl, sr, want_raw, want_det);
    }
    if (wb_on) awb_update(view, vw, vh);
    color_lut_apply(view, vw, vh, wb_on, gam_on);

    /* drawing+uploads: one render per consume kind max. JPEG-encode is
     * skipped entirely when nobody watches (idle = near-zero CPU).     */
    if (want_raw || want_det) {
        g_lock();
        G.view_seq++;                /* processed frame (detect sync)   */
        g_unlock();
    }
    if (want_raw) {
        size_t jl = 0;
        unsigned char *jpg = encode_jpg(view, vw, vh, q, &jl);
        if (jpg) {
            g_lock();
            G.vw = vw;
            G.vh = vh;
            publish_raw_jpg_locked(jpg, jl);
            g_unlock();
            free(jpg);
        }
    }
    if (want_det) {
        size_t jl = 0;
        unsigned char *jpg = encode_jpg_det(view, vw, vh, q, &jl);
        if (jpg) {
            g_lock();
            G.vw = vw;
            G.vh = vh;
            publish_det_jpg_locked(jpg, jl);
            g_unlock();
            free(jpg);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ *
 * capture thread + CSI session
 * ------------------------------------------------------------------ */
static void cam_reconnect_note(const char *why)
{
    set_note("camera not connected - retrying every 2 s (%s)", why);
}

static void csi_session(const char *dev)
{
    g_lock();
    if (G.reconf) G.reconf = 0;          /* fresh session reflects want */
    int mode_idx = G.mode_idx;
    G.connected = 0;
    snprintf(G.url, sizeof(G.url), "%s", dev);
    g_unlock();
    int scale = G.scale;
    int W = MODE_WH[mode_idx][0], H = MODE_WH[mode_idx][1];
    size_t fb = (size_t)W * (size_t)H * 2;
    int vw0, vh0;
    view_dims(mode_idx, scale, &vw0, &vh0);

    set_note("CSI camera open, wiring pipeline pads...");
    csi_configure(dev, W, H);

    int fd = open(dev, O_RDWR | O_NONBLOCK);      /* format owner fd    */
    if (fd < 0) {
        cam_reconnect_note("open video node failed");
        return;
    }

    set_note("CSI camera (%s): applying sensor settings...", dev);
    if (apply_want(W) != 0) {
        close(fd);
        return;
    }
    g_lock();
    snprintf(G.backend, sizeof(G.backend), "pipe");
    g_unlock();

    pid_t pp = 0;
    int rfd = -1;
    if (pumper_spawn(&pp, &rfd, dev, W, H) != 0) {
        close(fd);
        return;
    }
    G.pp_pid = (int)pp;
    G.pp_suspended = 0;               /* fresh session runs until idle  */

    unsigned short *raw = (unsigned short *)xmalloc(fb);
    unsigned char *mid = (unsigned char *)xmalloc((size_t)(W/2) * (H/2) * 3);

    int phase = verify_phase(W, H, rfd, raw);
    g_lock();
    G.view_seq = 0;                      /* verify frames are not shown */
    g_unlock();
    logf_("phase verified per session: %s", PH_NAME[phase]);

    double last_ok = 0.0;
    double fps = 0.0;
    long fidx = 0;
    int first_live = 0;
    size_t fpos = 0;             /* per-frame pipe progress (survives a
                                  * poll timeout mid-frame, keeps the
                                  * 4MB frame alignment). */

    for (;;) {
        g_lock();
        if (G.reconf) {
            G.reconf = 0;
            g_unlock();
            set_note("settings changed - rebuilding camera session...");
            break;
        }
        int curscale = G.scale;
        int suspended = G.pp_suspended;
        g_unlock();
        if (suspended) {
            /* idle capture: only watch for a dead child                 */
            if (G.pp_pid > 1) {
                int st2 = 0;
                pid_t r = waitpid((pid_t)G.pp_pid, &st2, WNOHANG);
                if (r == (pid_t)G.pp_pid || (r < 0 && errno == ECHILD))
                    break;
            }
            usleep(250 * 1000);
            continue;
        }

        struct pollfd pf = { rfd, POLLIN, 0 };
        int pr = poll(&pf, 1, 3000);
        if (pr <= 0) {
            if ((now_sec() - last_ok) > FRAME_STALL && last_ok > 0.0) {
                set_note("no camera frames for %d s - reconnecting",
                         (int)FRAME_STALL);
                break;
            }
            continue;
        }
        if (read_frame_pipe(rfd, (unsigned short *)(void *)
                            ((unsigned char *)raw + fpos),
                            fb - fpos, 3000) != 1) {
            cam_reconnect_note("pumper stream ended");
            break;
        }
        fpos = 0;
        double tnow = now_sec();
        if (last_ok > 0.0 && fidx >= 3) {
            double d = tnow - last_ok;
            if (d > 0) {
                double iv = 1.0 / d;
                fps = fps <= 0 ? iv : fps * 0.9 + iv * 0.1;
            }
        }
        last_ok = tnow;

        process_frame(mode_idx, curscale, W, H, phase, raw, mid);
        fidx++;

        if (!first_live || fidx % 150 == 0) {
            first_live = 1;
            g_lock();
            G.connected = 1;
            G.last_ok = tnow;
            G.fps = fps;
            snprintf(G.note, sizeof(G.note),
                     "CSI camera %s: LIVE (%.1f fps, pipe backend, "
                     "phase %s)", dev, fps, PH_NAME[phase]);
            g_unlock();
        }
    }

    pumper_kill(&pp, &rfd);
    G.pp_pid = -1;
    G.pp_suspended = 0;
    close(fd);
    free(raw);
    free(mid);
    (void)vw0; (void)vh0;
    cam_reconnect_note("session ended");
}

static void *capture_thread(void *unused)
{
    (void)unused;
    for (;;) {
        char url[128] = "";
        size_t n = 0;
        char *txt = read_file_full(URLFILE, &n);
        if (txt) {
            char *t = txt;
            while (*t == ' ' || *t == '\t') t++;
            size_t L = strcspn(t, "\r\n");
            if (L >= sizeof(url)) L = sizeof(url) - 1;
            memcpy(url, t, L);
            url[L] = 0;
            free(txt);
        }
        g_lock();
        G.attempts++;
        g_unlock();

        if (!url[0]) {
            set_note("no camera source configured - retrying every 2 s");
            usleep((int)(RETRY_EVERY * 1e6));
            continue;
        }
        g_lock();
        int someone = G.stream_live > 0 ||
                      (now_sec() - G.activity_t) < 12.0;
        g_unlock();
        if (!someone) {
            /* idle: nobody watching live media; don't even open a     */
            /* session right now (camera stays completely quiet)       */
            usleep((int)(RETRY_EVERY * 1e6));
            continue;
        }
        if (strncmp(url, "/dev/", 5) == 0) {
            csi_session(url);
            g_lock();
            int was_reconf = G.reconf;
            g_unlock();
            if (was_reconf) continue;        /* rebuild instantly       */
            usleep((int)(RETRY_EVERY * 1e6));
            continue;
        }
        set_note("birdcpp supports /dev/videoN CSI sources only "
                 "(python birdboard still handles http cameras)");
        usleep((int)(RETRY_EVERY * 1e6));
    }
    return NULL;
}

/* ------------------------------------------------------------------ *
 * native inference - onnxruntime C-API, loaded with dlopen at runtime.
 * Letterbox, forward, YOLOv8 NMS, box drawing and the JSON all in C;
 * the python sidecar is gone. v1.31.1, 4 intra threads: ~1.1 s.
 * ------------------------------------------------------------------ */
typedef struct {
    const OrtApi *(*GetApi)(uint32_t);
    const char *(*GetVersionString)(void);
} OrtApiBaseMirror;

static void *ort_lib;
static OrtApiBaseMirror *(*ort_get_base)(void);
static const OrtApi *OAPI;
static OrtEnv *oenv;
static OrtSessionOptions *oopts;
static OrtSession *osess;
static OrtMemoryInfo *omi;
static OrtValue *oinval;
static char *oin_name, *oout_name;
static float *oin_buf;                   /* 1x3x640x640 float32 CHW      */
static int oithreads = 4;                /* resolved intra thread count  */

#define ORTCHK(c)                                                     \
    do {                                                              \
        OrtStatus *st_ = (c);                                         \
        if (st_) {                                                    \
            const char *m_ = OAPI->GetErrorMessage(st_);              \
            logf_("ort fatal (%d): %s", (int)OAPI->GetErrorCode(st_), \
                  m_ ? m_ : "?");                                     \
            OAPI->ReleaseStatus(st_);                                 \
            return -1;                                                \
        }                                                             \
    } while (0)

static int ort_release_partial(int keep_lib)
{
    if (osess) { OAPI->ReleaseSession(osess); osess = NULL; }
    if (oopts) { OAPI->ReleaseSessionOptions(oopts); oopts = NULL; }
    if (omi) { OAPI->ReleaseMemoryInfo(omi); omi = NULL; }
    if (!keep_lib && oenv) { OAPI->ReleaseEnv(oenv); oenv = NULL; }
    return -1;
}

static int ort_init(const char *model_path, char *err, size_t errcap)
{
    /* retryable: clean any partial state first */
    ort_release_partial(0);
    if (ort_lib) { dlclose(ort_lib); ort_lib = NULL; }

    const char *path = opt_ortlib ? opt_ortlib : ORT_LIB_DEFAULT;
    ort_lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!ort_lib) {
        const char *e = dlerror();
        snprintf(err, errcap, "dlopen %s: %s", path, e ? e : "?");
        return -1;
    }
    dlerror();
    ort_get_base = (OrtApiBaseMirror *(*)(void))dlsym(ort_lib, "OrtGetApiBase");
    if (!ort_get_base) {
        snprintf(err, errcap, "dlsym OrtGetApiBase: %s", dlerror());
        return -1;
    }
    OrtApiBaseMirror *base = ort_get_base();
    if (!base || !base->GetApi) {
        snprintf(err, errcap, "OrtGetApiBase() NULL");
        return -1;
    }
    const OrtApi *api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        snprintf(err, errcap, "GetApi(%d) NULL", ORT_API_VERSION);
        return -1;
    }
    OAPI = api;

    double t0 = now_sec();
    ORTCHK(OAPI->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "birdcpp", &oenv));
    ORTCHK(OAPI->CreateSessionOptions(&oopts));
    ORTCHK(OAPI->SetIntraOpNumThreads(oopts, oithreads));
    ORTCHK(OAPI->SetInterOpNumThreads(oopts, 1));
    ORTCHK(OAPI->CreateSession(oenv, model_path, oopts, &osess));
    double create_ms = (now_sec() - t0) * 1000.0;

    OrtAllocator *alloc = NULL;
    ORTCHK(OAPI->GetAllocatorWithDefaultOptions(&alloc));
    ORTCHK(OAPI->SessionGetInputName(osess, 0, alloc, &oin_name));
    ORTCHK(OAPI->SessionGetOutputName(osess, 0, alloc, &oout_name));

    ORTCHK(OAPI->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                     &omi));
    int64_t indims[4] = { 1, 3, ORT_IMG, ORT_IMG };
    size_t nb = (size_t)3 * ORT_IMG * ORT_IMG * sizeof(float);
    free(oin_buf);
    oin_buf = (float *)xmalloc(nb);
    for (long long i = 0; i < (long long)(nb / sizeof(float)); i++)
        oin_buf[i] = 0.45f;
    ORTCHK(OAPI->CreateTensorWithDataAsOrtValue(
        omi, (void *)oin_buf, nb, indims, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &oinval));

    logf_("ort: %s  session %.0f ms (threads=%d)", model_path,
          create_ms, oithreads);
    return 0;
}

/* ------------------------------------------------------------------ *
 * ort state for letterbox
 * ------------------------------------------------------------------ */
static unsigned char *ocanvas;           /* ORT_IMG*ORT_IMG*3  u8        */
static float olr;
static int oltop, olleft, olw, olh;      /* src (view) geometry          */

static float ort_min_r(int h, int w)
{
    float rh = (float)ORT_IMG / (float)h;
    float rw = (float)ORT_IMG / (float)w;
    return rh < rw ? rh : rw;
}

static void ort_letterbox(const unsigned char *bgr, int w, int h)
{
    if (!ocanvas)
        ocanvas = (unsigned char *)xmalloc((size_t)ORT_IMG * ORT_IMG * 3);
    unsigned char *cv = ocanvas;
    olr = ort_min_r(h, w);
    int nh = (int)rintf((float)h * olr);
    int nw = (int)rintf((float)w * olr);
    int top = (ORT_IMG - nh) / 2;
    int left = (ORT_IMG - nw) / 2;
    oltop = top;
    olleft = left;
    olw = w;
    olh = h;

    memset(cv, 114, (size_t)ORT_IMG * ORT_IMG * 3);

    /* bilinear rows: y from src [0..h-1] -> lambda precomputed per row */
    for (int y = 0; y < nh; y++) {
        float sy = ((float)y + 0.5f) * (float)h / (float)nh - 0.5f;
        int y0 = (int)sy;                /* floorf for >=0 values here? */
        if (y0 < 0) y0 = 0;
        if (y0 > h - 1) y0 = h - 1;
        int y1 = y0 + 1 < h ? y0 + 1 : h - 1;
        float fy = sy - (float)y0;
        if (fy < 0) fy = 0;
        float gy = 1.0f - fy;
        const unsigned char *r0 = bgr + (size_t)y0 * w * 3;
        const unsigned char *r1 = bgr + (size_t)y1 * w * 3;
        unsigned char *dst = cv + (size_t)(top + y) * ORT_IMG * 3 +
                             (size_t)left * 3;
        for (int x = 0; x < nw; x++) {
            float sx = ((float)x + 0.5f) * (float)w / (float)nw - 0.5f;
            int x0 = (int)sx;
            if (x0 < 0) x0 = 0;
            if (x0 > w - 1) x0 = w - 1;
            int x1 = x0 + 1 < w ? x0 + 1 : w - 1;
            float fx = sx - (float)x0;
            if (fx < 0) fx = 0;
            float gx = 1.0f - fx;
            float w00 = gy * gx, w01 = gy * fx;
            float w10 = fy * gx, w11 = fy * fx;
            size_t i0 = (size_t)x0 * 3, i1 = (size_t)x1 * 3;
            for (int c = 0; c < 3; c++) {
                float v = r0[i0 + c] * w00 + r0[i1 + c] * w01 +
                          r1[i0 + c] * w10 + r1[i1 + c] * w11;
                unsigned char u = v < 0 ? 0 : (v > 255 ? 255 :
                                (unsigned char)(v + 0.5f));
                dst[(size_t)x * 3 + c] = u;
            }
        }
    }

    /* CHW float, RGB (/255; BGR frame -> R,G,B channels)                */
    float *chR = oin_buf, *chG = chR + ORT_IMG * ORT_IMG,
          *chB = chG + ORT_IMG * ORT_IMG;
    const float pad = 114.0f / 255.0f;
    for (size_t i = 0; i < (size_t)ORT_IMG * ORT_IMG; i++) {
        chR[i] = pad;
        chG[i] = pad;
        chB[i] = pad;
    }
    for (int y = 0; y < nh; y++) {
        const unsigned char *row = cv + (size_t)(top + y) * ORT_IMG * 3 +
                                   (size_t)left * 3;
        float *oR = chR + (size_t)(top + y) * ORT_IMG + left;
        float *oG = chG + (size_t)(top + y) * ORT_IMG + left;
        float *oB = chB + (size_t)(top + y) * ORT_IMG + left;
        for (int x = 0; x < nw; x++) {
            size_t o = (size_t)x * 3;
            oR[x] = row[o + 2] / 255.0f;
            oG[x] = row[o + 1] / 255.0f;
            oB[x] = row[o + 0] / 255.0f;
        }
    }
}

/* forward: returns a caller-owned OrtValue (NULL on error). The value
 * holds the output storage; parse BEFORE releasing.                    */
static OrtValue *ort_run(void)
{
    const char *in_names[1] = { oin_name };
    const char *out_names[1] = { oout_name };
    OrtValue *outs[1] = { NULL };
    OrtStatus *st = OAPI->Run(osess, NULL, in_names,
                              (const OrtValue *const *)&oinval, 1,
                              out_names, 1, outs);
    if (st) {
        const char *m = OAPI->GetErrorMessage(st);
        logf_("ort run failed (%d): %s", (int)OAPI->GetErrorCode(st),
              m ? m : "?");
        OAPI->ReleaseStatus(st);
        return NULL;
    }
    return outs[0];
}

static void meta_csv_dets(sbuf *b, const detbox_t *d, int n)
{
    sb_app(b, "[");
    for (int i = 0; i < n; i++) {
        if (i) sb_app(b, ", ");
        sb_appf(b, "\"bird %.2f (%d,%d,%d,%d)\"", (double)d[i].sc,
                (int)d[i].x1, (int)d[i].y1, (int)d[i].x2, (int)d[i].y2);
    }
    sb_app(b, "]");
}

static void ort_meta_publish(const detbox_t *d, int n, int ms, int vw, int vh)
{
    sbuf b;
    sb_init(&b);
    sb_appf(&b, "{\"file\":\"frame.jpg\",\"n\":%d,\"infer_ms\":%.1f,"
                "\"size\":[%d,%d],\"dets\":", n, (double)ms, vw, vh);
    meta_csv_dets(&b, d, n);
    sb_app(&b, "}");
    g_lock();
    int lim = n < MAX_DETS ? n : MAX_DETS;
    for (int i = 0; i < lim; i++) G.dets[i] = d[i];
    G.ndets = lim;
    G.det_t = now_sec();
    G.last_infer_t = G.det_t;
    snprintf(G.meta_txt, sizeof(G.meta_txt), "%s", b.b ? b.b : "");
    g_unlock();
    sb_free(&b);
}

/* greedy NMS in letterbox-space xywh; keep order = score desc.         */
static int nms_pick(const float *bx, const float *by, const float *bw,
                    const float *bh, const float *sc, int n, int *keep)
{
    int idx[8400];
    int alive[8400];
    for (int i = 0; i < n; i++) { idx[i] = i; alive[i] = 1; }
    for (int i = 1; i < n; i++) {
        int v = idx[i], j = i - 1;
        while (j >= 0 && sc[idx[j]] < sc[v]) { idx[j + 1] = idx[j]; j--; }
        idx[j + 1] = v;
    }
    int k = 0;
    for (int a = 0; a < n; a++) {
        if (!alive[idx[a]]) continue;
        int i = idx[a];
        keep[k++] = i;
        float xi1 = bx[i], yi1 = by[i];
        float xi2 = bx[i] + bw[i], yi2 = by[i] + bh[i];
        float ai = bw[i] * bh[i];
        if (ai < 1e-9f) ai = 1e-9f;
        for (int b = a + 1; b < n; b++) {
            if (!alive[idx[b]]) continue;
            int j = idx[b];
            float xj1 = bx[j], yj1 = by[j];
            float xj2 = bx[j] + bw[j], yj2 = by[j] + bh[j];
            float xx1 = xi1 > xj1 ? xi1 : xj1;
            float yy1 = yi1 > yj1 ? yi1 : yj1;
            float xx2 = xi2 < xj2 ? xi2 : xj2;
            float yy2 = yi2 < yj2 ? yi2 : yj2;
            float iw = xx2 > xx1 ? xx2 - xx1 : 0.0f;
            float ih = yy2 > yy1 ? yy2 - yy1 : 0.0f;
            float aj = bw[j] * bh[j];
            if (aj < 1e-9f) aj = 1e-9f;
            float iou = (iw * ih) / (ai + aj - iw * ih);
            if (iou > ORT_IOU) alive[idx[b]] = 0;
        }
    }
    return k;
}

/* per-stage detection timings (last completed cycle, ms); written by the
 * infer thread, published in /phone and throttled to the journal        */
typedef struct { double snap, letterbox, fwd, parse, nms, cycle; } detstage_t;
static detstage_t dett;

/* full detect cycle: oin_buf must be letterboxed for (vw,vh) first     */
static int ort_detect_cycle(int vw, int vh)
{
    double t0 = now_sec();
    OrtValue *ov = ort_run();
    if (!ov) {
        set_meta_error("inference error: forward failed");
        return -1;
    }
    int ms = (int)((now_sec() - t0) * 1000.0);
    g_lock();
    dett.fwd = (double)ms;
    dett.parse = 0.0;
    dett.nms = 0.0;
    g_unlock();

    float *op = NULL;
    if (OAPI->GetTensorMutableData(ov, (void **)&op) != NULL || !op) {
        OAPI->ReleaseValue(ov);
        set_meta_error("inference error: no output tensor");
        return -1;
    }

    /* output layout: (1,4+nc,N) or the transposed (1,N,4+nc).          */
    OrtTensorTypeAndShapeInfo *tsi = NULL;
    OrtStatus *st = OAPI->GetTensorTypeAndShape(ov, &tsi);
    long long N = 0, layout = 0;
    if (st == NULL && tsi) {
        size_t nd = 0;
        if (OAPI->GetDimensionsCount(tsi, &nd) == NULL && nd == 3) {
            int64_t dims[4] = { 0, 0, 0, 0 };
            if (OAPI->GetDimensions(tsi, dims, nd) == NULL) {
                if (dims[1] == 4 + 1) { layout = 1; N = dims[2]; }
                else if (dims[2] == 4 + 1) { layout = 2; N = dims[1]; }
            }
        }
        OAPI->ReleaseTensorTypeAndShapeInfo(tsi);
    }
    if (!layout || N < 1 || N > 8400) {
        OAPI->ReleaseValue(ov);
        set_meta_error("inference error: unexpected output shape");
        return -1;
    }

    static float bx[8400], by[8400], bwv[8400], bhv[8400], scv[8400];
    static int keep[8400];
    static detbox_t res[MAX_DETS];

    int cnt = 0;
    double tf = now_sec();
    for (long long a = 0; a < N; a++) {
        size_t ai = (size_t)a;
        float s = layout == 1 ? op[4 * N + ai] : op[ai * 5 + 4];
        if (s > ORT_CONF && cnt < 8400) {
            float cx = layout == 1 ? op[0 * N + ai] : op[ai * 5 + 0];
            float cy = layout == 1 ? op[1 * N + ai] : op[ai * 5 + 1];
            float w_ = layout == 1 ? op[2 * N + ai] : op[ai * 5 + 2];
            float h_ = layout == 1 ? op[3 * N + ai] : op[ai * 5 + 3];
            bx[cnt] = cx - w_ * 0.5f;
            by[cnt] = cy - h_ * 0.5f;
            bwv[cnt] = w_;
            bhv[cnt] = h_;
            scv[cnt] = s;
            cnt++;
        }
    }
    dett.parse = (now_sec() - tf) * 1000.0;

    int nr = 0;
    double tn = now_sec();
    if (cnt > 0) {
        int k = nms_pick(bx, by, bwv, bhv, scv, cnt, keep);
        for (int i = 0; i < k && nr < MAX_DETS; i++) {
            int j = keep[i];
            res[nr].x1 = (bx[j] - (float)olleft) / olr;
            res[nr].y1 = (by[j] - (float)oltop) / olr;
            res[nr].x2 = (bx[j] + bwv[j] - (float)olleft) / olr;
            res[nr].y2 = (by[j] + bhv[j] - (float)oltop) / olr;
            res[nr].sc = scv[j];
            nr++;
        }
    }
    dett.nms = (now_sec() - tn) * 1000.0;
    OAPI->ReleaseValue(ov);

    ort_meta_publish(res, nr, ms, vw, vh);
    return 0;
}

/* ------------------------------------------------------------------ *
 * capture gating: viewers bring the pipeline up, ~12 s idle sleeps it
 * ------------------------------------------------------------------ */
static void gate_mark_activity(void)
{
    g_lock();
    G.activity_t = now_sec();
    g_unlock();
}

static void gate_stream_open(int raw)
{
    g_lock();
    G.stream_live++;
    if (raw) G.stream_raw++;
    G.activity_t = now_sec();
    if (G.pp_pid > 0 && G.pp_suspended) {
        kill((pid_t)G.pp_pid, SIGCONT);
        G.pp_suspended = 0;
        logf_("pumper resumed (stream opened)");
    }
    g_unlock();
}

static void gate_stream_close(int raw)
{
    g_lock();
    if (G.stream_live > 0) G.stream_live--;
    if (raw && G.stream_raw > 0) G.stream_raw--;
    G.activity_t = now_sec();        /* last viewer event: uniform grace */
    g_unlock();
}

static void *gate_thread(void *unused)
{
    (void)unused;
    for (;;) {
        usleep(1000 * 1000);
        g_lock();
        int live = G.stream_live;
        pid_t pid = (pid_t)G.pp_pid;
        int susp = G.pp_suspended;
        double age = now_sec() - G.activity_t;
        if (!live && pid > 0 && !susp && age > 12.0) {
            kill(pid, SIGSTOP);
            G.pp_suspended = 1;
            logf_("pumper suspended (idle %.0f s)", age);
        }
        g_unlock();
    }
    return NULL;
}

/* ------------------------------------------------------------------ *
 * inference thread (native ORT)
 * ------------------------------------------------------------------ */
static void *infer_thread(void *unused)
{
    (void)unused;
    const char *model = opt_model ? opt_model : MODEL_PATH;
    if (access(model, R_OK) != 0)
        model = HOME_ROOTBIRD "/best.onnx";

    char err[160];
    int ready = 0;

    for (;;) {
        g_lock();
        int act = G.stream_live;
        g_unlock();
        if (!act) {
            usleep(1500 * 1000);
            continue;
        }
        if (!ready) {
            if (ort_init(model, err, sizeof(err)) != 0) {
                set_meta_error(err);
                usleep(3000 * 1000);
                continue;
            }
            ready = 1;
            logf_("ort ready (threads=%d)", oithreads);
        }

        double t0 = now_sec();
        int vw, vh, ok = 0;
        g_lock();
        if (G.view_bgr && G.view_seq != G.infer_handled &&
            G.vw_alloc > 0) {
            vw = G.vw_alloc;
            vh = G.vh_alloc;
            size_t need = (size_t)vw * vh * 3;
            if (!G.infer_snap || G.snap_alloc != vw * vh) {
                free(G.infer_snap);
                G.infer_snap = (unsigned char *)xmalloc(need);
                G.snap_alloc = vw * vh;
            }
            memcpy(G.infer_snap, G.view_bgr, need);
            G.infer_handled = G.view_seq;
            ok = 1;
        }
        dett.snap = ok ? (now_sec() - t0) * 1000.0 : 0.0;
        g_unlock();
        if (!ok) {
            usleep(250 * 1000);
            continue;
        }

        double t1 = now_sec();
        ort_letterbox(G.infer_snap, vw, vh);
        double tt = now_sec();
        g_lock();
        dett.letterbox = (tt - t1) * 1000.0;
        g_unlock();

        if (ort_detect_cycle(vw, vh) != 0) {
            /* one-off errors: retried next cycle */
            usleep(500 * 1000);
            continue;
        }
        double t2 = now_sec();
        g_lock();
        dett.cycle = (t2 - t0) * 1000.0;
        g_unlock();

        /* throttled stage report: one journal line every ~2 s */
        static double last_rep = 0.0;
        if (t2 - last_rep >= 2.0) {
            last_rep = t2;
            logf_("det-stage: snap %.0f letterbox %.0f fwd %.0f parse %.0f "
                  "nms %.0f | cycle %.0f ms",
                  dett.snap, dett.letterbox, dett.fwd, dett.parse,
                  dett.nms, dett.cycle);
        }
        usleep(120 * 1000);
    }
    return NULL;
}

/* self-test: synthetic frame -> letterbox -> one forward -> parse      */
static int run_orttest(void)
{
    const char *model = opt_model ? opt_model : MODEL_PATH;
    if (access(model, R_OK) != 0)
        model = HOME_ROOTBIRD "/best.onnx";
    if (access(model, R_OK) != 0) {
        logf_("orttest: no model found at %s", model);
        return 1;
    }
    char err[160];
    if (ort_init(model, err, sizeof(err)) != 0) {
        logf_("orttest: init failed: %s", err);
        return 1;
    }
    const int W = 820, H = 616;
    unsigned char *fr = (unsigned char *)xmalloc((size_t)W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            size_t o = ((size_t)y * W + x) * 3;
            fr[o + 0] = (unsigned char)(x * 255 / W);       /* B        */
            fr[o + 1] = (unsigned char)(y * 255 / H);       /* G        */
            fr[o + 2] = (unsigned char)(128);               /* R        */
        }
    double tl = now_sec();
    ort_letterbox(fr, W, H);
    logf_("orttest: letterbox %.1f ms (canvas %dx%d paste %dx%d @%d,%d)",
          (now_sec() - tl) * 1000.0, ORT_IMG, ORT_IMG,
          (int)rintf((float)H * olr), (int)rintf((float)W * olr),
          oltop, olleft);
    free(fr);
    int rc1 = ort_detect_cycle(W, H);
    logf_("orttest: run1 rc=%d meta=%s", rc1, rc1 == 0 ? G.meta_txt : "-");
    int rc2 = ort_detect_cycle(W, H);
    logf_("orttest: run2 rc=%d meta=%s", rc2, rc2 == 0 ? G.meta_txt : "-");
    return rc2 != 0;
}

/* ------------------------------------------------------------------ *
 * camera-set apply (python do_POST semantics, verbatim):
 * checkbox absent = off; scalar field absent = unchanged;
 * fpreset > 0 overrides vblank; mode/flip change -> reconf
 * ------------------------------------------------------------------ */
struct field { char name[16]; char val[24]; };

static void url_decode(char *s)
{
    char *o = s;
    while (*s) {
        if (*s == '%' && s[1] && s[2]) {
            int h0 = isxdigit((unsigned char)s[1]) ?
                     (isdigit((unsigned char)s[1]) ? s[1] - '0' :
                      (tolower((unsigned char)s[1]) - 'a' + 10)) : -1;
            int h1 = isxdigit((unsigned char)s[2]) ?
                     (isdigit((unsigned char)s[2]) ? s[2] - '0' :
                      (tolower((unsigned char)s[2]) - 'a' + 10)) : -1;
            if (h0 >= 0 && h1 >= 0) {
                *o++ = (char)((h0 << 4) | h1);
                s += 3;
                continue;
            }
        }
        if (*s == '+') *o = ' ';
        else *o = *s;
        o++;
        s++;
    }
    *o = 0;
}

static int parse_form(char *body, struct field *F, int maxf)
{
    int nf = 0;
    char *save = NULL;
    for (char *kv = strtok_r(body, "&", &save); kv && nf < maxf;
         kv = strtok_r(NULL, "&", &save)) {
        char *eq = strchr(kv, '=');
        if (!eq) continue;
        *eq = 0;
        url_decode(kv);
        url_decode(eq + 1);
        snprintf(F[nf].name, sizeof(F[nf].name), "%.15s", kv);
        snprintf(F[nf].val, sizeof(F[nf].val), "%.23s", eq + 1);
        nf++;
    }
    return nf;
}

static const char *field_get(struct field *F, int nf, const char *k)
{
    for (int i = 0; i < nf; i++)
        if (!strcmp(F[i].name, k)) return F[i].val;
    return NULL;
}

/* checkbox semantics: absent / "", "0","off","false","no" -> 0 else 1 */
static int field_flag(struct field *F, int nf, const char *k)
{
    const char *v = field_get(F, nf, k);
    if (!v) return 0;
    if (!*v) return 0;
    if (!strcmp(v, "0") || !strcmp(v, "off") || !strcmp(v, "false") ||
        !strcmp(v, "no")) return 0;
    return 1;
}

static void note_add(sbuf *s, int *any, const char *fmt, ...)
{
    va_list ap;
    char t[128];
    va_start(ap, fmt);
    vsnprintf(t, sizeof(t), fmt, ap);
    va_end(ap);
    if (*any) sb_app(s, "; ");
    sb_app(s, t);
    *any = 1;
}

static void apply_camera_set(struct field *F, int nf)
{
    sbuf notes;
    sb_init(&notes);
    int any = 0, need_reconf = 0;

    g_lock();
    want_t snap = G.want;
    int old_mode = G.mode_idx;
    int old_scale = G.scale;
    int old_h = G.want.hflip, old_v = G.want.vflip;
    double line_us = G.line_us;
    g_unlock();
    (void)snap;

    /* 1) sensor scalars present in the form (clamped) */
    const char *names[NSCALARS + 2];
    long vals[NSCALARS + 2];
    int nk = 0;
    long submitted[NSCALARS];
    int has_sub[NSCALARS];
    int vb_submitted = 0;
    for (int i = 0; i < NSCALARS; i++) has_sub[i] = 0;
    for (int i = 0; i < NSCALARS; i++) {
        const char *t = field_get(F, nf, SCALAR_FIELD[i]);
        if (t && *t) {
            long v = clamp_name(SCALARS[i], (long)atof(t));
            submitted[i] = v;
            has_sub[i] = 1;
            note_add(&notes, &any, "%s=%ld", SCALAR_FIELD[i], v);
            names[nk] = SCALARS[i];
            vals[nk] = v;
            nk++;
            if (i == WVB) vb_submitted = 1;
        }
    }

    /* 2) fps preset overrides vblank (needs a measured line time) */
    const char *fpt = field_get(F, nf, "fpreset");
    if (fpt && *fpt && (int)atof(fpt) > 0) {
        int fp = (int)atof(fpt);
        if (line_us > 0) {
            long vb = (long)(1.0e6 / ((double)fp * line_us)) -
                      MODE_WH[old_mode][1];
            vb = clamp_name("vertical_blanking", vb);
            note_add(&notes, &any, "preset %d fps -> vblank %ld", fp, vb);
            g_lock();
            G.want.scal[WVB] = vb;
            g_unlock();
            vb_submitted = 1;
            int found = -1;
            for (int j = 0; j < nk; j++)
                if (!strcmp(names[j], "vertical_blanking")) found = j;
            if (found >= 0) vals[found] = vb;
            else {
                names[nk] = "vertical_blanking";
                vals[nk] = vb;
                nk++;
            }
        } else {
            note_add(&notes, &any,
                     "fps preset needs live line_time first (retry)");
        }
    }

    /* 3) mode + flips (rebuild triggers) */
    const char *mv = field_get(F, nf, "mode");
    int new_mode = old_mode;
    if (mv)
        for (int i = 0; i < NMODES; i++)
            if (!strcmp(mv, MODE_NM[i])) { new_mode = i; break; }
    int newh = field_flag(F, nf, "hflip");
    int newv = field_flag(F, nf, "vflip");
    if (new_mode != old_mode) {
        note_add(&notes, &any, "mode %s", MODE_NM[new_mode]);
        need_reconf = 1;
    }
    if (newh != old_h || newv != old_v) {
        note_add(&notes, &any, "flips %d/%d", newh, newv);
        need_reconf = 1;
    }

    /* 4) persist everything (python S["want"] = want, S[...] = ...) */
    const char *sc = field_get(F, nf, "scale");
    int new_scale = old_scale;
    if (sc)
        for (int i = 0; i < NSCALES; i++)
            if (!strcmp(sc, SCALE_FIELD[i])) { new_scale = i; break; }
    if (new_scale != old_scale)
        note_add(&notes, &any, "view %s", SCALE_LABEL[new_scale]);
    const char *qt = field_get(F, nf, "quality");
    int newq = -1;
    if (qt && *qt) {
        newq = (int)atof(qt);
        if (newq < 30) newq = 30;
        if (newq > 95) newq = 95;
    }
    g_lock();
    for (int i = 0; i < NSCALARS; i++)
        if (has_sub[i]) G.want.scal[i] = submitted[i];
    G.want.hflip = newh;
    G.want.vflip = newv;
    G.scale = new_scale;
    if (new_scale != old_scale)
        view_dims(G.mode_idx, G.scale, &G.vw, &G.vh);
    if (newq > 0) G.quality = newq;
    G.wb = field_flag(F, nf, "wb");
    G.gamma = field_flag(F, nf, "gamma");
    G.mode_idx = new_mode;
    G.reconf = need_reconf;
    g_unlock();

    /* 5) live-apply scalar changes (no rebuild needed) */
    if (nk > 0 && !need_reconf) {
        int rc = sensor_set(names, vals, nk);
        if (rc != 0) {
            note_add(&notes, &any, "driver rejected");
        } else {
            get_ctl();
            g_lock();
            for (int i = 0; i < NSCALARS; i++) {
                if (!has_sub[i]) continue;
                int ic = ctl_index(SCALARS[i]);
                if (ic >= 0) G.want.scal[i] = G.ctl[ic].val;
            }
            if (vb_submitted) {
                long hb = 0, pr = 0;
                int ihb = ctl_index("horizontal_blanking");
                int ipr = ctl_index("pixel_rate");
                if (ihb >= 0) hb = G.ctl[ihb].val;
                if (ipr >= 0) pr = G.ctl[ipr].val;
                if (pr > 0)
                    G.line_us =
                        (double)(MODE_WH[G.mode_idx][0] + hb) /
                        (double)pr * 1e6;
            }
            g_unlock();
        }
    }

    set_note("settings: %s", notes.len ? notes.b : "no changes in form");
    sb_free(&notes);
}

/* ------------------------------------------------------------------ *
 * /phone json (python key set, verbatim)
 * ------------------------------------------------------------------ */
static void json_escape_sb(sbuf *b, const char *s)
{
    for (const char *p = s ? s : ""; *p; p++) {
        char c = *p;
        if (c == '"' || c == '\\') sb_appc(b, '\\');
        if ((unsigned char)c >= 0x20) sb_appc(b, c);
    }
}

static void phone_json(sbuf *b)
{
    g_lock();
    int connected = G.connected, attempts = G.attempts;
    double fps = G.fps, jt = G.jpeg_t;
    double djt = G.det_jpg_t;        /* annotated-frame publish time     */
    unsigned long frames = G.frames;
    int sl = G.stream_live, sr = G.stream_raw;
    unsigned long dpubs = G.det_pubs, rpubs = G.raw_pubs;
    detstage_t dstg = dett;
    char backend[16], note[NOTE_MAX], url[128];
    memcpy(backend, G.backend, sizeof(backend));
    memcpy(note, G.note, sizeof(note));
    memcpy(url, G.url, sizeof(url));
    int wb = G.wb, gamma = G.gamma, qual = G.quality;
    int modei = G.mode_idx, scale = G.scale, vw = G.vw, vh = G.vh;
    double line_us = G.line_us;
    struct ctlent ctl[CTLS_MAX];
    int nctl = G.nctl;
    memcpy(ctl, G.ctl, sizeof(ctl));
    want_t w = G.want;
    g_unlock();

    double newest = jt > djt ? jt : djt;
    int fage = newest > 0.0 ? (int)((now_sec() - newest) * 1000.0) : -1;
    sb_app(b, "{\"connected\": ");
    sb_app(b, connected ? "true" : "false");
    sb_app(b, ", \"note\": \"");
    json_escape_sb(b, note);
    sb_app(b, "\", \"url\": \"");
    json_escape_sb(b, url);
    sb_appf(b, "\", \"attempts\": %d, \"retry_s\": 2", attempts);
    if (fage >= 0) sb_appf(b, ", \"frame_age_ms\": %d", fage);
    else sb_app(b, ", \"frame_age_ms\": null");
    sb_appf(b, ", \"frames\": %lu", frames);
    sb_appf(b, ", \"stream_live\": %d, \"stream_raw\": %d", sl, sr);
    sb_appf(b, ", \"det_pubs\": %lu, \"raw_pubs\": %lu", dpubs, rpubs);
    sb_appf(b, ", \"stage\": {\"snap\": %.0f, \"letterbox\": %.0f, "
               "\"fwd\": %.0f, \"parse\": %.0f, \"nms\": %.0f, "
               "\"cycle\": %.0f}",
            dstg.snap, dstg.letterbox, dstg.fwd, dstg.parse, dstg.nms,
            dstg.cycle);
    sb_appf(b, ", \"backend\": \"%s\", \"fps\": %.2f", backend, fps);
    sb_appf(b, ", \"wb\": %d, \"gamma\": %d", wb, gamma);
    sb_appf(b, ", \"mode\": \"%s\", \"scale\": \"%s\"", MODE_NM[modei],
            SCALE_LABEL[scale]);
    sb_appf(b, ", \"quality\": %d, \"preview\": [%d, %d]", qual, vw, vh);
    if (line_us > 0) sb_appf(b, ", \"line_us\": %.4f", line_us);
    else sb_app(b, ", \"line_us\": null");
    {
        long wsc[11];
        wsc[0] = w.scal[WEXP];
        wsc[1] = w.scal[WGAIN];
        wsc[2] = w.scal[WDIG];
        wsc[3] = w.scal[WVB];
        wsc[4] = w.scal[WPATT];
        wsc[5] = w.scal[WPR];
        wsc[6] = w.scal[WPGR];
        wsc[7] = w.scal[WPGB];
        wsc[8] = w.scal[WPB];
        wsc[9] = w.hflip;
        wsc[10] = w.vflip;
        static const char *const wk[11] = { "exposure", "analogue_gain",
            "digital_gain", "vertical_blanking", "test_pattern",
            "red_pixel_value", "green_red_pixel_value",
            "green_blue_pixel_value", "blue_pixel_value",
            "horizontal_flip", "vertical_flip" };
        sb_app(b, ", \"want\": {");
        for (int i = 0; i < 11; i++)
            sb_appf(b, "%s\"%s\": %ld", i ? ", " : "", wk[i], wsc[i]);
        sb_app(b, "}, \"ctl\": {");
        for (int i = 0; i < nctl; i++)
            sb_appf(b, "%s\"%s\": %ld", i ? ", " : "", ctl[i].name,
                    ctl[i].val);
        sb_app(b, "}}");
    }
}

/* ------------------------------------------------------------------ *
 * dashboard HTML (python _HTML_TPL with the placeholders substituted)
 * ------------------------------------------------------------------ */
static void opt_pattern(sbuf *b, int cur)
{
    static const char *const labels[5] =
        { "0 - off", "1", "2 - solid color", "3", "4" };
    for (int v = 0; v < 5; v++)
        sb_appf(b, "<option value=\"%d\"%s>%s</option>", v,
                v == cur ? " selected" : "", labels[v]);
}

static void opt_mode(sbuf *b, int cur)
{
    for (int i = 0; i < NMODES; i++)
        sb_appf(b, "<option value=\"%s\"%s>%s</option>", MODE_NM[i],
                i == cur ? " selected" : "", MODE_NM[i]);
}

static void opt_scale(sbuf *b, int cur)
{
    for (int i = 0; i < NSCALES; i++)
        sb_appf(b, "<option value=\"%s\"%s>%s</option>", SCALE_FIELD[i],
                i == cur ? " selected" : "", SCALE_LABEL[i]);
}

static void page_html(sbuf *b)
{
    g_lock();
    want_t w = G.want;
    int modei = G.mode_idx, scale = G.scale, qual = G.quality;
    int wb = G.wb, gamma = G.gamma;
    g_unlock();

    sb_app(b, "<!doctype html><html><head><title>Live Bird Detection (all "
              "on TI board)</title>\n"
              "<meta name=\"viewport\" content=\"width=device-width,initi"
              "al-scale=1\">\n<style>\n"
              "body{background:#0e1116;color:#dbe2ef;font-family:Arial,sa"
              "ns-serif;margin:0;padding:12px;font-size:14px}\n"
              "h1{font-size:18px;margin:0 0 4px}\n"
              ".sub{color:#8fa1c7;font-size:12px;margin-bottom:10px}\n"
              ".grid{display:flex;flex-wrap:wrap;gap:10px}\n"
              ".card{background:#161b24;border:1px solid #2a3245;border-r"
              "adius:8px;padding:6px}\n"
              ".card h3{margin:2px 0 4px;font-size:13px;color:#9ab4e8}\n"
              "img{max-width:96vw;border-radius:6px;background:#000}\n"
              ".stats{margin-top:10px;font-size:13px;color:#8fa1c7;line-h"
              "eight:1.8}\n"
              ".dot{display:inline-block;width:9px;height:9px;border-radi"
              "us:50%;background:#888;margin-right:4px}\n"
              ".dot.ok{background:#2ecc71}\n.dot.warn{background:#e67e22}"
              "\n.dot.bad{background:#e74c3c}\n"
              ".warnbox{background:#3a2a18;border:1px solid #7a5a2a;color:"
              "#f5c06a;border-radius:8px;padding:6px 10px;margin-top:8px;"
              "font-size:12px;display:none}\n"
              "#ph{color:#8fa1c7;font-size:13px;padding:4px}\n"
              "a{color:#66aaff}\n.cb{display:block;margin:3px 0;font-size"
              ":13px}\n.hc{color:#6d7fa5;font-size:11px}\n"
              "input[type=\"number\"],select{background:#0e1116;color:#db"
              "e2ef;border:1px solid #2a3245;border-radius:4px;padding:1p"
              "x 3px;font-size:12px}\n"
              ".links{margin:4px 0 0;padding-left:16px;font-size:12px;col"
              "or:#8fa1c7;line-height:1.7}\n"
              "button{background:#1c2740;border:1px solid #3d5a99;color:#"
              "dfe9ff;border-radius:6px;padding:3px 14px;margin-top:5px;c"
              "ursor:pointer;font-size:13px}\n"
              "</style></head><body>\n"
              "<h1>Live bird detection - running on the TI SK-AM62 board<"
              "/h1>\n"
              "<div class=\"sub\">camera &rarr; board capture &rarr; board"
              " YOLOv8 &rarr; boxes drawn on board &middot; live media on"
              "ly - capture runs only while this page (its embedded"
              " stream) is open</div>\n"
              "<div class=\"grid\">\n"
              " <div class=\"card\"><h3>Live - boxes drawn on board</h3>"
              "<img src=\"/stream\" alt=\"live\"></div>\n"
              " <div class=\"card\"><h3>Camera settings</h3>\n"
              "  <form method=\"post\" action=\"/camera-set\">\n");
    sb_app(b, "   <label class=\"cb\">exposure <input type=\"number\" nam"
              "e=\"exposure\" style=\"width:74px\" value=\"");
    sb_appf(b, "%ld", w.scal[WEXP]);
    sb_app(b, "\">\n"
              "    an. gain <input type=\"number\" name=\"gain\" style=\"w"
              "idth:54px\" value=\"");
    sb_appf(b, "%ld", w.scal[WGAIN]);
    sb_app(b, "\">\n"
              "    dig gain <input type=\"number\" name=\"dig\" style=\"wi"
              "dth:60px\" value=\"");
    sb_appf(b, "%ld", w.scal[WDIG]);
    sb_app(b, "\"></label>\n"
              "   <label class=\"cb\">vblank <input type=\"number\" name=\"v"
              "blank\" style=\"width:66px\" value=\"");
    sb_appf(b, "%ld", w.scal[WVB]);
    sb_app(b, "\">\n"
              "    fps preset <input type=\"number\" name=\"fpreset\" styl"
              "e=\"width:50px\" placeholder=\"keep\">\n"
              "    <span class=\"hc\">(blank = keep current)</span></label"
              ">\n"
              "   <label class=\"cb\">test pattern <select name=\"pattern\""
              ">");
    opt_pattern(b, (int)w.scal[WPATT]);
    sb_app(b, "</select></label>\n"
              "   <label class=\"cb\">pixels R <input type=\"number\" name"
              "=\"pr\" style=\"width:64px\" value=\"");
    sb_appf(b, "%ld", w.scal[WPR]);
    sb_app(b, "\">\n"
              "    gA <input type=\"number\" name=\"pgr\" style=\"width:64p"
              "x\" value=\"");
    sb_appf(b, "%ld", w.scal[WPGR]);
    sb_app(b, "\">\n"
              "    gB <input type=\"number\" name=\"pgb\" style=\"width:64p"
              "x\" value=\"");
    sb_appf(b, "%ld", w.scal[WPGB]);
    sb_app(b, "\">\n"
              "    bl <input type=\"number\" name=\"pb\" style=\"width:64px"
              "\" value=\"");
    sb_appf(b, "%ld", w.scal[WPB]);
    sb_app(b, "\">\n"
              "    <span class=\"hc\">(solid-color pixels, pats 2/3)</span"
              "></label>\n"
              "   <label class=\"cb\">sensor mode <select name=\"mode\">");
    opt_mode(b, modei);
    sb_app(b, "</select>\n"
              "    view <select name=\"scale\">");
    opt_scale(b, scale);
    sb_app(b, "</select>\n"
              "    JPEG q <input type=\"number\" name=\"quality\" style=\"w"
              "idth:50px\" value=\"");
    sb_appf(b, "%d", qual);
    sb_app(b, "\"></label>\n"
              "   <label class=\"cb\"><input type=\"checkbox\" name=\"hfli"
              "p\" value=\"1\"");
    if (w.hflip) sb_app(b, " checked");
    sb_app(b, "> horizontal flip\n"
              "    <input type=\"checkbox\" name=\"vflip\" value=\"1\"");
    if (w.vflip) sb_app(b, " checked");
    sb_app(b, "> vertical flip\n"
              "    <span class=\"hc\">(flips + mode rebuild the pipeline)<"
              "/span></label>\n"
              "   <label class=\"cb\"><input type=\"checkbox\" name=\"wb\""
              " value=\"1\"");
    if (wb) sb_app(b, " checked");
    sb_app(b, "> white balance\n"
              "    <input type=\"checkbox\" name=\"gamma\" value=\"1\"");
    if (gamma) sb_app(b, " checked");
    sb_app(b, "> shadow lift</label>\n"
              "   <button type=\"submit\">apply camera settings</button>\n"
              "  </form>\n"
              " </div>\n"
              " <div class=\"card\"><h3>Pages on this server</h3>\n"
              "  <ul class=\"links\">\n"
              "   <li><a href=\"/stream\">/stream</a> - live MJPEG feed (a"
              "nnotated; <a href=\"/stream?raw=1\">?raw=1</a> for clean"
              " view)</li>\n"
              "   <li><a href=\"/meta.json\">/meta.json</a> - detection JS"
              "ON</li>\n"
              "   <li><a href=\"/phone\">/phone</a> - camera status JSON</"
              "li>\n"
              "   <li><a id=\"lab\" href=\"#\">camera tuning lab (port 88"
              "89)</a> - separate service</li>\n"
              "  </ul>\n"
              " </div>\n"
              "</div>\n"
              "<div class=\"stats\">\n"
              " <span id=\"pdot\" class=\"dot\"></span><span id=\"phone\">c"
              "amera: checking...</span> &nbsp;|&nbsp;\n"
              " birds: <b id=\"n\">-</b> &nbsp;|&nbsp; infer: <b id=\"ms\">"
              "-</b> ms &nbsp;|&nbsp; updated: <b id=\"t\">-</b>\n"
              "</div>\n"
              "<div class=\"stats\" id=\"setstat\">settings: -</div>\n"
              "<div class=\"warnbox\" id=\"warn\"></div>\n"
              "<script>\n"
              "var pdot=document.getElementById('pdot'),pt=document.getEl"
              "ementById('phone'),warn=document.getElementById('warn');\n"
              "document.getElementById('lab').href = location.protocol + "
              "'//' + location.hostname + ':8889/';\n"
              "/* the <img src=\"/stream\"> tag itself keeps the camera"
              " alive; the polls below are passive and do not start"
              " capture */\n"
              "setInterval(function(){\n"
              "  if(document.hidden) return;\n"
              "  fetch('/meta.json',{cache:'no-store'}).then(function(r){r"
              "eturn r.json();}).then(function(j){\n"
              "    document.getElementById('n').textContent=(j.n==null?'-'"
              ":j.n);\n"
              "    document.getElementById('ms').textContent=(j.infer_ms=="
              "null?'-':Math.round(j.infer_ms));\n"
              "    document.getElementById('t').textContent=new Date().toL"
              "ocaleTimeString();\n"
              "  }).catch(function(){});\n},2000);\n"
              "setInterval(function(){\n"
              "  if(document.hidden) return;\n"
              "  fetch('/phone',{cache:'no-store'}).then(function(r){retur"
              "n r.json();}).then(function(j){\n"
              "    if(j.connected){\n"
              "      pdot.className='dot ok';\n"
              "      pt.textContent='camera: LIVE - '+(j.frame_age_ms/1000"
              ").toFixed(1)+' s old (through the board)';\n"
              "      warn.style.display='none';\n"
              "    } else {\n"
              "      pdot.className='dot warn';\n"
              "      pt.textContent='camera: not connected - retrying ever"
              "y '+j.retry_s+' s';\n"
              "      warn.style.display='block';\n"
              "      warn.textContent=j.note;\n"
              "    }\n"
              "    var st=document.getElementById('setstat');\n"
              "    st.textContent='settings: mode '+j.mode+' | view '+j.pr"
              "eview+'@q'+j.quality+\n"
              "      ' | driver exp '+(j.ctl.exposure!=null?j.ctl.exposure"
              ":j.want.exposure)+\n"
              "      ' gain '+(j.ctl.analogue_gain!=null?j.ctl.analogue_ga"
              "in:j.want.analogue_gain)+\n"
              "      ' dig '+(j.ctl.digital_gain!=null?j.ctl.digital_gain:"
              "j.want.digital_gain)+\n"
              "      ' vblank '+(j.ctl.vertical_blanking!=null?j.ctl.verti"
              "cal_blanking:j.want.vertical_blanking)+\n"
              "      (j.line_us!=null?(' | line '+j.line_us+' us'):'');\n"
              "  }).catch(function(){});\n},2000);\n"
              "</script></body></html>");
}

/* ------------------------------------------------------------------ *
 * HTTP server (thread per connection)
 * ------------------------------------------------------------------ */
static void emit(int fd, int code, const char *ctype,
                 const unsigned char *body, size_t len, int redirect)
{
    sbuf h;
    sb_init(&h);
    const char *t = code == 200 ? "OK" : (code == 303 ? "See Other" :
                   (code == 404 ? "Not Found" :
                    (code == 410 ? "Gone" : "OK")));
    sb_appf(&h, "HTTP/1.1 %d %s\r\n", code, t);
    if (ctype) sb_appf(&h, "Content-Type: %s\r\n", ctype);
    if (redirect) sb_app(&h, "Location: /\r\n");
    sb_appf(&h, "Content-Length: %zu\r\n", len);
    sb_app(&h, "Cache-Control: no-store\r\nConnection: close\r\n\r\n");
    int bad = write_all(fd, h.b, h.len) != 0;
    if (!bad && len && body) bad = write_all(fd, body, len) != 0;
    sb_free(&h);
}

static void emit_str(int fd, int code, const char *ctype, const char *body)
{
    emit(fd, code, ctype, (const unsigned char *)body,
         body ? strlen(body) : 0, 0);
}

static void emit_303(int fd)
{
    emit(fd, 303, NULL, (const unsigned char *)"", 0, 1);
}

static void conn_stream(int fd, int raw)
{
    gate_stream_open(raw);
    sbuf h;
    sb_init(&h);
    sb_app(&h, "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replac"
               "e; boundary=jpegframe\r\nCache-Control: no-store\r\n\r\n");
    if (write_all(fd, h.b, h.len) != 0) { sb_free(&h); gate_stream_close(raw); return; }
    sb_free(&h);
    for (;;) {
        unsigned char *local = NULL;
        size_t jl = 0;
        g_lock();
        if (raw) {
            if (G.jpg && G.jpg_len) {
                local = (unsigned char *)xmalloc(G.jpg_len);
                memcpy(local, G.jpg, G.jpg_len);
                jl = G.jpg_len;
            }
        } else {
            if (G.det_jpg && G.det_len) {
                local = (unsigned char *)xmalloc(G.det_len);
                memcpy(local, G.det_jpg, G.det_len);
                jl = G.det_len;
            }
        }
        g_unlock();
        if (local) {
            char part[128];
            int pn = snprintf(part, sizeof(part),
                              "--jpegframe\r\nContent-Type: image/jpeg\r\n"
                              "Content-Length: %zu\r\n\r\n", jl);
            int bad = write_all(fd, part, (size_t)pn) != 0 ||
                      write_all(fd, local, jl) != 0 ||
                      write_all(fd, "\r\n", 2) != 0;
            free(local);
            if (bad) { gate_stream_close(raw); return; }
        } else {
            /* no frame yet: peek the socket so that a client that has
             * gone is detected (nothing is ever written to it, so the
             * consumer slot would otherwise never be freed and the
             * gate would never suspend the pumper)                   */
            char peek[1];
            if (recv(fd, peek, 1, MSG_PEEK | MSG_DONTWAIT) == 0) {
                gate_stream_close(raw);
                return;
            }
        }
        usleep(60 * 1000);
    }
}

static void route_worker(int fd, const char *path, const char *query,
                         char *body)
{
    if (!strcmp(path, "/") || !strcmp(path, "/view") ||
        !strcmp(path, "/index.html")) {
        sbuf h;
        sb_init(&h);
        page_html(&h);
        emit_str(fd, 200, "text/html; charset=utf-8", h.b);
        sb_free(&h);
        return;
    }
    if (!strncmp(path, "/frame.jpg", 10) || !strcmp(path, "/last_frame.jpg") ||
        !strncmp(path, "/snapshot.jpg", 13)) {
        emit_str(fd, 410, "application/json",
                 "{\"err\":\"snapshot endpoints removed - use /stream (live "
                 "media only; capture is suspended when nobody watches)\"}");
        return;
    }
    if (!strcmp(path, "/det.jpg") || !strncmp(path, "/live_det.jpg", 13)) {
        emit_str(fd, 410, "application/json",
                 "{\"err\":\"det.jpg removed - use /stream (boxes drawn into "
                 "the live stream now)\"}");
        return;
    }
    if (!strcmp(path, "/stream")) {
        int raw = query && strstr(query, "raw=1") != NULL;
        conn_stream(fd, raw);
        return;
    }
    if (!strcmp(path, "/meta.json") || !strcmp(path, "/meta")) {
        g_lock();
        char meta[4096];
        memcpy(meta, G.meta_txt, sizeof(meta));
        g_unlock();
        emit_str(fd, 200, "application/json", meta);
        return;
    }
    if (!strcmp(path, "/phone")) {
        sbuf b;
        sb_init(&b);
        phone_json(&b);
        emit_str(fd, 200, "application/json", b.b);
        sb_free(&b);
        return;
    }
    if (!strcmp(path, "/ping")) {
        emit_str(fd, 200, "text/plain", "board-ok");
        return;
    }
    if (!strcmp(path, "/camera-set")) {
        gate_mark_activity();            /* a settings write is activity */
        struct field F[24];
        int nf = parse_form(body ? body : (char *)"", F, 24);
        apply_camera_set(F, nf);
        emit_303(fd);
        return;
    }
    emit_str(fd, 404, "application/json", "{\"err\":\"unknown endpoint\"}");
}

static void *conn_thread(void *arg)
{
    int fd = (int)(intptr_t)arg;
    struct timeval tv = { 5, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    char head[8192];
    size_t hl = 0;
    int hdr_done = 0;
    while (!hdr_done && hl < sizeof(head) - 1) {
        ssize_t n = recv(fd, head + hl, sizeof(head) - 1 - hl, 0);
        if (n <= 0) { close(fd); return NULL; }
        hl += (size_t)n;
        head[hl] = 0;
        if (strstr(head, "\r\n\r\n") || strstr(head, "\n\n")) hdr_done = 1;
    }
    if (!hdr_done) { close(fd); return NULL; }

    char *sp = strstr(head, "\r\n\r\n");
    size_t skip = sp ? (size_t)(sp - head) + 4 : hl;
    unsigned long bl = 0;
    char *cl = strcasestr(head, "content-length:");
    if (cl) bl = strtoul(cl + 15, NULL, 10);
    if (bl > 64 * 1024) bl = 64 * 1024;
    char *body = bl ? (char *)xmalloc(bl + 1) : NULL;
    size_t got = hl > skip ? hl - skip : 0;
    if (body && got > bl) got = bl;
    if (body && got) memcpy(body, head + skip, got);
    while (body && got < bl) {
        ssize_t n = recv(fd, body + got, bl - got, 0);
        if (n <= 0 && n != -1) break;
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            break;
        }
        got += (size_t)n;
    }
    if (body) body[got] = 0;

    char line[640] = "", method[8] = "";
    if (sscanf(head, " %7s %639s", method, line) != 2) {
        free(body);
        close(fd);
        return NULL;
    }
    char query[512] = "";
    char *qm = strchr(line, '?');
    if (qm) {
        *qm = 0;
        snprintf(query, sizeof(query), "%s", qm + 1);
    }

    /* /camera-set: GET uses the query, POST optionally appends it */
    if (!strcmp(line, "/camera-set")) {
        if (body && got && query[0]) {
            sbuf m;
            sb_init(&m);
            sb_appn(&m, body, got);
            sb_app(&m, "&");
            sb_app(&m, query);
            free(body);
            body = m.b;
            got = m.len;
            body[m.len] = 0;
        } else if ((!body || !got) && query[0]) {
            free(body);
            body = xstrdup(query);
            got = strlen(body);
        }
    }

    route_worker(fd, line, query, body);

    free(body);
    shutdown(fd, SHUT_WR);
    close(fd);
    return NULL;
}

static void http_serve(int port)
{
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) { logf_("socket failed"); exit(1); }
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_ANY);
    sa.sin_port = htons((unsigned short)port);
    if (bind(s, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        logf_("bind :%d failed", port);
        exit(1);
    }
    if (listen(s, 16) != 0) { logf_("listen failed"); exit(1); }
    logf_("birdcpp live on http://0.0.0.0:%d/  (camera: %s)", port, URLFILE);
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    for (;;) {
        int c = accept(s, NULL, NULL);
        if (c < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            usleep(50000);
            continue;
        }
        pthread_t th;
        if (pthread_create(&th, &at, conn_thread, (void *)(intptr_t)c) != 0)
            close(c);
    }
}

/* ------------------------------------------------------------------ *
 * tjtest: encode solid red/blue frames, verify with cv2 externally
 * ------------------------------------------------------------------ */
static int run_tjtest(void)
{
    G.tj = tjInitCompress();
    if (!G.tj) { logf_("tjInitCompress failed"); return 1; }
    const int W = 640, H = 480;
    unsigned char *src = (unsigned char *)xmalloc((size_t)W * H * 3);
    for (int c = 0; c < 2; c++) {
        /* c=0: BGR "red": B=0 G=0 R=255.  c=1: "blue": B=255 G=0 R=0   */
        for (int i = 0; i < W * H; i++) {
            src[(size_t)i * 3 + 0] = (unsigned char)(c ? 255 : 0);
            src[(size_t)i * 3 + 1] = 0;
            src[(size_t)i * 3 + 2] = (unsigned char)(c ? 0 : 255);
        }
        size_t jl = 0;
        unsigned char *jpg = encode_jpg(src, W, H, 85, &jl);
        if (!jpg) { free(src); return 1; }
        char path[80];
        snprintf(path, sizeof(path), "/tmp/tjtest_%s.jpg", c ? "blue" : "red");
        FILE *f = fopen(path, "wb");
        if (!f) {
            logf_("cannot write %s", path);
            free(jpg);
            free(src);
            return 1;
        }
        fwrite(jpg, 1, jl, f);
        fclose(f);
        free(jpg);
        logf_("wrote %s (%zu bytes)", path, jl);
    }
    free(src);
    logf_("decode /tmp/tjtest_red.jpg and /tmp/tjtest_blue.jpg with cv2; "
          "red must come back R=255,G=0,B=0 and blue B=255");
    return 0;
}

/* ------------------------------------------------------------------ *
 * main
 * ------------------------------------------------------------------ */
static void governor_perf(void)
{
    for (int cpu = 0; cpu < 8; cpu++) {
        char pth[128];
        snprintf(pth, sizeof(pth),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor",
                 cpu);
        FILE *f = fopen(pth, "w");
        if (!f) continue;
        fputs("performance", f);
        fclose(f);
    }
}

int main(int argc, char **argv)
{
    int port = 8888;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i + 1 < argc)
            port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc)
            opt_model = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc)
            port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tjtest"))
            return run_tjtest();
        else if (!strcmp(argv[i], "--orttest"))
            return run_orttest();
        else if (!strcmp(argv[i], "--ortlib") && i + 1 < argc)
            opt_ortlib = argv[++i];
        else if (!strcmp(argv[i], "--ort-threads") && i + 1 < argc)
            opt_ortthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--img") && i + 1 < argc)
            ORT_IMG = atoi(argv[++i]);
    }

    if (ORT_IMG < 320 || ORT_IMG > 640 || ORT_IMG % 32) {
        logf_("bad --img %d (must be a multiple of 32 in 320..640)", ORT_IMG);
        return 2;
    }

    /* opt-B (final config): the copier is pinned+niced and 1-threaded by
     * its own spawner. Pinning the WHOLE birdcpp process to the other
     * cores was tested and made the ORT forward SLOWER (~2.75 s vs
     * ~1.7 s): with only 3 visible cpus the ORT thread pool loses more
     * than the copier-park saves, so the main thread wards stay DARK.  */
    signal(SIGPIPE, SIG_IGN);
    setvbuf(stdout, NULL, _IONBF, 0);

    memset(&G, 0, sizeof(G));
    pthread_mutex_init(&G.mu, NULL);
    G.pp_pid = -1;                       /* no pumper until first session */
    G.activity_t = now_sec();
    oithreads = opt_ortthreads;
    G.quality = 80;
    G.wb = 1;
    G.gamma = 1;
    G.mode_idx = 0;                     /* 1640x1232 (user-tuned mode) */
    G.scale = SCALE_HALF;
    snprintf(G.backend, sizeof(G.backend), "?");
    snprintf(G.note, sizeof(G.note), "starting");
    snprintf(G.meta_txt, sizeof(G.meta_txt),
             "{\"n\":null,\"infer_ms\":null,\"err\":\"warming up\"}");
    /* user-tuned sensor defaults (match the python birdboard) */
    G.want.scal[WEXP] = 1759;
    G.want.scal[WGAIN] = 200;
    G.want.scal[WDIG] = 4095;
    G.want.scal[WVB] = 683;
    G.want.scal[WPATT] = 0;
    G.want.scal[WPR] = 1023;
    G.want.scal[WPGR] = 1023;
    G.want.scal[WPGB] = 1023;
    G.want.scal[WPB] = 1023;
    for (int i = 0; i < 3; i++) G.wb_ema[i] = 1.0;
    view_dims(G.mode_idx, G.scale, &G.vw, &G.vh);
    G.vw_alloc = G.vw;
    G.vh_alloc = G.vh;
    G.view_bgr = (unsigned char *)
        xmalloc((size_t)G.vw * G.vh * 3);
    memset(G.view_bgr, 32, (size_t)G.vw * G.vh * 3);
    /* the annotated-stream canvas must exist too: a pre-set vw_alloc
     * makes ensure_view_bufs_locked early-return, so it would never
     * lazily allocate det_bgr and every det encode would NULL out    */
    G.det_bgr = (unsigned char *)
        xmalloc((size_t)G.vw * G.vh * 3);

    G.tj = tjInitCompress();
    if (!G.tj) { logf_("tjInitCompress failed"); return 1; }

    governor_perf();

    pthread_t t1, t2, t3;
    pthread_create(&t1, NULL, capture_thread, NULL);
    pthread_create(&t2, NULL, infer_thread, NULL);
    pthread_create(&t3, NULL, gate_thread, NULL);
    pthread_detach(t1);
    pthread_detach(t2);
    pthread_detach(t3);

    http_serve(port);
    return 0;
}
