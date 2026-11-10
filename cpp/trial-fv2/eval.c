/* yf2eval - offline accuracy harness for the YOLO-FastestV2 trial.
 *
 * ALL IN C on purpose: it runs the exact pipeline the sandbox server
 * runs - BGR/255 bilinear letterbox at 352, the two-output fv2 head
 * decode (handel_preds/make_grid math, bird = COCO class 14), greedy
 * NMS - but driven by FILES instead of the camera:
 *
 *     DATADIR/images/*.jpg     val images (jpeg)
 *     DATADIR/labels/<stem>.txt  YOLO txt labels: "0 cx cy w h" normalized
 *
 * and reports:
 *     - AP@0.5 (all-point interpolation, bird class only)
 *     - precision / recall / F1 at fixed conf thresholds
 *     - average forward time per image
 *
 * Build:  make yf2eval                      (board gcc; ORT + turbojpeg)
 * Run:    ./yf2eval -m yf2_coco352.onnx -d /home/root/fv2val
 *
 * The decode block mirrors cpp/trial-fv2/birdcpp.c fv2_detect_cycle;
 * keep the two in sync if one changes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <math.h>
#include <ctype.h>
#include <dlfcn.h>
#include <time.h>

#include "onnxruntime_c_api.h"

#define ORT_LIB_DEFAULT "/home/root/ort/lib/libonnxruntime.so"
#define ORT_IMG         352
#define FV2_CH          95
#define FV2_OBJ_OFF     12
#define FV2_CLS_OFF     15
#define FV2_BIRD_CLS    14
#define MAX_DETS        256             /* kept boxes per image (nms)   */
#define MAXCAND         400000          /* global candidate arena       */
#define MAXGT           100000          /* global ground-truth arena    */

static const float FV2_ANCH[2][6] = {
    { 12.64f, 19.39f,   37.88f, 51.48f,  55.71f, 138.31f },  /* s16     */
    { 126.91f, 78.23f, 131.57f, 214.55f, 279.92f, 258.87f }, /* s32     */
};

/* turbojpeg - no headers on the board; declared from memory like the
 * server does (libjpeg-turbo 2.x signatures; decode side this time).  */
typedef void *tjhandle;
enum { TJPF_RGB = 0, TJPF_BGR, TJPF_GRAY, TJPF_RGBA, TJPF_BGRA,
       TJPF_ARGB, TJPF_ABGR };
extern tjhandle tjInitDecompress(void);
extern int tjDecompressHeader3(tjhandle handle,
                               const unsigned char *jpegBuf,
                               unsigned long jpegSize, int *width,
                               int *height, int *jpegSubsamp,
                               int *jpegColorspace);
extern int tjDecompress2(tjhandle handle, const unsigned char *jpegBuf,
                         unsigned long jpegSize, unsigned char *dstBuf,
                         int width, int pitch, int height, int pixelFormat,
                         int flags);
extern void tjFree(unsigned char *jpegBuf);
extern char *tjGetErrorStr2(tjhandle handle);
extern int tjDestroy(tjhandle handle);

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "oom %zu\n", n); exit(1); }
    return p;
}

static char *xstrdup(const char *s)
{
    char *d = strdup(s ? s : "");
    if (!d) { fprintf(stderr, "oom\n"); exit(1); }
    return d;
}

/* ------------------------------------------------------------ ORT --- */
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
static OrtValue *oinval, *ofval[2];
static char *oin_name, *ofname[2];
static float *oin_buf, *ofbuf[2], *ofptr[2];
static int fv2_g2, fv2_g3;

#define ORTCHK(c)                                                     \
    do {                                                              \
        OrtStatus *st_ = (c);                                         \
        if (st_) {                                                    \
            const char *m_ = OAPI->GetErrorMessage(st_);              \
            fprintf(stderr, "ort fatal: %s\n", m_ ? m_ : "?");        \
            OAPI->ReleaseStatus(st_);                                 \
            exit(1);                                                  \
        }                                                             \
    } while (0)

static void ort_init(const char *model_path, int threads)
{
    ort_lib = dlopen(ORT_LIB_DEFAULT, RTLD_NOW | RTLD_LOCAL);
    if (!ort_lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); exit(1); }
    ort_get_base = (OrtApiBaseMirror *(*)(void))dlsym(ort_lib,
                                                      "OrtGetApiBase");
    if (!ort_get_base) { fprintf(stderr, "no OrtGetApiBase\n"); exit(1); }
    OAPI = ort_get_base()->GetApi(ORT_API_VERSION);
    if (!OAPI) { fprintf(stderr, "GetApi NULL\n"); exit(1); }

    ORTCHK(OAPI->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "yf2eval", &oenv));
    ORTCHK(OAPI->CreateSessionOptions(&oopts));
    ORTCHK(OAPI->SetIntraOpNumThreads(oopts, threads));
    ORTCHK(OAPI->SetInterOpNumThreads(oopts, 1));
    double t0 = now_sec();
    ORTCHK(OAPI->CreateSession(oenv, model_path, oopts, &osess));
    fprintf(stderr, "yf2eval: session created in %.0f ms (threads=%d)\n",
            (now_sec() - t0) * 1000.0, threads);

    OrtAllocator *alloc = NULL;
    ORTCHK(OAPI->GetAllocatorWithDefaultOptions(&alloc));
    ORTCHK(OAPI->SessionGetInputName(osess, 0, alloc, &oin_name));
    ORTCHK(OAPI->SessionGetOutputName(osess, 0, alloc, &ofname[0]));
    ORTCHK(OAPI->SessionGetOutputName(osess, 1, alloc, &ofname[1]));

    ORTCHK(OAPI->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                     &omi));
    int64_t indims[4] = { 1, 3, ORT_IMG, ORT_IMG };
    size_t nb = (size_t)3 * ORT_IMG * ORT_IMG * sizeof(float);
    oin_buf = (float *)xmalloc(nb);
    ORTCHK(OAPI->CreateTensorWithDataAsOrtValue(
        omi, (void *)oin_buf, nb, indims, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &oinval));

    fv2_g2 = ORT_IMG / 16;
    fv2_g3 = ORT_IMG / 32;
    int64_t d2[4] = { 1, fv2_g2, fv2_g2, FV2_CH };
    int64_t d3[4] = { 1, fv2_g3, fv2_g3, FV2_CH };
    size_t n2 = (size_t)fv2_g2 * fv2_g2 * FV2_CH * sizeof(float);
    size_t n3 = (size_t)fv2_g3 * fv2_g3 * FV2_CH * sizeof(float);
    ofbuf[0] = (float *)xmalloc(n2);
    ofbuf[1] = (float *)xmalloc(n3);
    ORTCHK(OAPI->CreateTensorWithDataAsOrtValue(
        omi, (void *)ofbuf[0], n2, d2, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ofval[0]));
    ORTCHK(OAPI->CreateTensorWithDataAsOrtValue(
        omi, (void *)ofbuf[1], n3, d3, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ofval[1]));
    void *pv = NULL;
    ORTCHK(OAPI->GetTensorMutableData(ofval[0], &pv));
    ofptr[0] = (float *)pv;
    ORTCHK(OAPI->GetTensorMutableData(ofval[1], &pv));
    ofptr[1] = (float *)pv;
    fprintf(stderr, "yf2eval: fv2 grids %dx%d + %dx%d\n",
            fv2_g2, fv2_g2, fv2_g3, fv2_g3);
}

/* ----------------------------------------------------- letterbox --- */
static unsigned char *ocanvas;
static float olr;
static int oltop, olleft;
static int feed_resize = 0;              /* reference cv2.resize feed    */
static float olrw_v, olrh_v;             /* inverse scale per axis       */

/* plain resize feed: 352x352 stretched both axes, NO padding - what the
 * COCO training/test actually used (test.py does cv2.resize directly)  */
static void fv2_resize(const unsigned char *bgr, int w, int h)
{
    olleft = oltop = 0;
    olrw_v = (float)ORT_IMG / (float)w;
    olrh_v = (float)ORT_IMG / (float)h;

    float *chR = oin_buf, *chG = chR + ORT_IMG * ORT_IMG,
          *chB = chG + ORT_IMG * ORT_IMG;
    for (int y = 0; y < ORT_IMG; y++) {
        float sy = ((float)y + 0.5f) * (float)h / (float)ORT_IMG - 0.5f;
        int y0 = (int)sy;
        if (y0 < 0) y0 = 0;
        if (y0 > h - 1) y0 = h - 1;
        int y1 = y0 + 1 < h ? y0 + 1 : h - 1;
        float fy = sy - (float)y0;
        if (fy < 0) fy = 0;
        float gy = 1.0f - fy;
        const unsigned char *r0 = bgr + (size_t)y0 * w * 3;
        const unsigned char *r1 = bgr + (size_t)y1 * w * 3;
        float *oR = chR + y * ORT_IMG;
        float *oG = chG + y * ORT_IMG;
        float *oB = chB + y * ORT_IMG;
        for (int x = 0; x < ORT_IMG; x++) {
            float sx = ((float)x + 0.5f) * (float)w / (float)ORT_IMG -
                       0.5f;
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
            float b = r0[i0] * w00 + r0[i1] * w01 + r1[i0] * w10 +
                      r1[i1] * w11;
            float g = r0[i0 + 1] * w00 + r0[i1 + 1] * w01 +
                      r1[i0 + 1] * w10 + r1[i1 + 1] * w11;
            float r = r0[i0 + 2] * w00 + r0[i1 + 2] * w01 +
                      r1[i0 + 2] * w10 + r1[i1 + 2] * w11;
            oR[x] = b / 255.0f;          /* BGR order kept            */
            oG[x] = g / 255.0f;
            oB[x] = r / 255.0f;
        }
    }
}

/* BGR stays BGR (fv2 reference feed), /255, CHW float                  */
static void fv2_letterbox(const unsigned char *bgr, int w, int h)
{
    if (!ocanvas)
        ocanvas = (unsigned char *)xmalloc((size_t)ORT_IMG * ORT_IMG * 3);
    unsigned char *cv = ocanvas;
    float rh = (float)ORT_IMG / (float)h;
    float rw = (float)ORT_IMG / (float)w;
    olr = rh < rw ? rh : rw;
    int nh = (int)rintf((float)h * olr);
    int nw = (int)rintf((float)w * olr);
    int top = (ORT_IMG - nh) / 2;
    int left = (ORT_IMG - nw) / 2;
    oltop = top;
    olleft = left;
    olrw_v = olr;
    olrh_v = olr;

    memset(cv, 114, (size_t)ORT_IMG * ORT_IMG * 3);
    for (int y = 0; y < nh; y++) {
        float sy = ((float)y + 0.5f) * (float)h / (float)nh - 0.5f;
        int y0 = (int)sy;
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
            oR[x] = row[o + 0] / 255.0f;   /* BGR order kept           */
            oG[x] = row[o + 1] / 255.0f;
            oB[x] = row[o + 2] / 255.0f;
        }
    }
}

/* --------------------------------------------- decode + nms + AP --- */
static int nms_pick(const float *bx, const float *by, const float *bw,
                    const float *bh, const float *sc, int n, int *keep,
                    float iou_thr)
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
            if (iou > iou_thr) alive[idx[b]] = 0;
        }
    }
    return k;
}

typedef struct {
    int img;                /* index into the image list               */
    float score;
    float x1, y1, x2, y2;   /* ORIGINAL-image pixel coords             */
} cand_t;

typedef struct {
    int img;
    float x1, y1, x2, y2;
    int used;               /* claimed by a matching TP                */
} gt_t;

static cand_t *GC;
static long NGC;
static gt_t *GT;
static long NGT;

static float iou_box(float ax1, float ay1, float ax2, float ay2,
                     float bx1, float by1, float bx2, float by2)
{
    float xx1 = ax1 > bx1 ? ax1 : bx1;
    float yy1 = ay1 > by1 ? ay1 : by1;
    float xx2 = ax2 < bx2 ? ax2 : bx2;
    float yy2 = ay2 < by2 ? ay2 : by2;
    float iw = xx2 > xx1 ? xx2 - xx1 : 0.0f;
    float ih = yy2 > yy1 ? yy2 - yy1 : 0.0f;
    float inter = iw * ih;
    float a = (ax2 - ax1) * (ay2 - ay1);
    float b = (bx2 - bx1) * (by2 - by1);
    return inter / (a + b - inter + 1e-9f);
}

static int cand_cmp(const void *pa, const void *pb)
{
    const cand_t *a = pa, *b = pb;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    return 0;
}

/* one image: jpeg -> letterbox -> run -> decode -> nms; survivors are
 * appended to the global candidate arena in ORIGINAL pixel coords.    */
static void run_one(const char *path, int imgid, float floor_conf,
                    float nms_iou, double *fwd_ms)
{
    unsigned char *jb = NULL;
    long jsz;
    {
        FILE *f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "open %s\n", path); return; }
        fseek(f, 0, SEEK_END);
        jsz = ftell(f);
        fseek(f, 0, SEEK_SET);
        jb = (unsigned char *)xmalloc((size_t)jsz);
        if (fread(jb, 1, (size_t)jsz, f) != (size_t)jsz) {
            fclose(f);
            free(jb);
            return;
        }
        fclose(f);
    }
    tjhandle tj = tjInitDecompress();
    int w = 0, h = 0, sub = 0, cs = 0;
    if (!tj || tjDecompressHeader3(tj, jb, (unsigned long)jsz,
                                   &w, &h, &sub, &cs) != 0) {
        fprintf(stderr, "skip (bad jpeg): %s\n", path);
        if (tj) tjDestroy(tj);
        free(jb);
        return;
    }
    unsigned char *bgr = (unsigned char *)xmalloc((size_t)w * h * 3);
    if (tjDecompress2(tj, jb, (unsigned long)jsz, bgr, w, w * 3, h,
                      TJPF_BGR, 0) != 0) {
        fprintf(stderr, "decode fail: %s: %s\n", path,
                tjGetErrorStr2(tj));
        tjDestroy(tj);
        free(bgr);
        free(jb);
        return;
    }
    tjDestroy(tj);
    free(jb);

    if (feed_resize)
        fv2_resize(bgr, w, h);
    else
        fv2_letterbox(bgr, w, h);
    double t0 = now_sec();
    const char *in_names[1] = { oin_name };
    const char *out_names[2] = { ofname[0], ofname[1] };
    OrtStatus *st = OAPI->Run(osess, NULL, in_names,
                              (const OrtValue *const *)&oinval, 1,
                              out_names, 2, ofval);
    if (st) {
        fprintf(stderr, "run fail %s: %s\n", path,
                OAPI->GetErrorMessage(st));
        OAPI->ReleaseStatus(st);
        free(bgr);
        return;
    }
    *fwd_ms += (now_sec() - t0) * 1000.0;

    /* decode (mirror of fv2_detect_cycle)                              */
    static float bx[8400], by[8400], bwv[8400], bhv[8400], scv[8400];
    static int keep[8400];
    int cnt = 0;
    const int stride[2] = { 16, 32 };
    for (int s = 0; s < 2; s++) {
        int g = s == 0 ? fv2_g2 : fv2_g3;
        const float *A = FV2_ANCH[s];
        const float *ob = ofptr[s];
        for (int y = 0; y < g; y++) {
            size_t rowbase = (size_t)y * (size_t)g * FV2_CH;
            for (int x = 0; x < g; x++) {
                const float *row = ob + rowbase + (size_t)x * FV2_CH;
                float clb = row[FV2_CLS_OFF + FV2_BIRD_CLS];
                if (clb < floor_conf) continue;
                for (int a = 0; a < 3 && cnt < 8400; a++) {
                    float sc = row[FV2_OBJ_OFF + a] * clb;
                    if (sc <= floor_conf) continue;
                    const float *r = row + a * 4;
                    float e2 = r[2] * 2.0f, e3 = r[3] * 2.0f;
                    bx[cnt] = ((r[0] * 2.0f - 0.5f) + (float)x) *
                              (float)stride[s];
                    by[cnt] = ((r[1] * 2.0f - 0.5f) + (float)y) *
                              (float)stride[s];
                    bwv[cnt] = e2 * e2 * A[a * 2];
                    bhv[cnt] = e3 * e3 * A[a * 2 + 1];
                    scv[cnt++] = sc;
                }
            }
        }
    }
    free(bgr);
    bgr = NULL;

    if (cnt < 1) return;
    int k = nms_pick(bx, by, bwv, bhv, scv, cnt, keep, nms_iou);
    for (int i = 0; i < k && NGC < MAXCAND; i++) {
        int j = keep[i];
        cand_t *c = &GC[NGC++];
        c->img = imgid;
        c->score = scv[j];
        c->x1 = (bx[j] - (float)olleft) / olrw_v;
        c->y1 = (by[j] - (float)oltop) / olrh_v;
        c->x2 = (bx[j] + bwv[j] - (float)olleft) / olrw_v;
        c->y2 = (by[j] + bhv[j] - (float)oltop) / olrh_v;
    }
}

static const char *stem(const char *name, char *out, size_t cap)
{
    size_t n = strlen(name);
    if (n >= cap) return NULL;
    snprintf(out, cap, "%s", name);
    char *dot = strrchr(out, '.');
    if (dot) *dot = 0;
    return out;
}

/* --probe N: dump raw head activity + GT-vs-pred diagnostics for image
 * #N (0-based in readdir order - nondeterministic, caller beware). The
 * ORT outputs still hold image N's forward when this runs.            */
static void probe_dump(long gt0, int imgid)
{
    printf("PROBE image=%d\n", imgid);
    /* raw head: top-8 cells per scale by obj*cls14                     */
    typedef struct { float sc, obj, cl, tx, ty, tw, th; int s, x, y, a; }
        cell_t;
    cell_t top[16];
    int ntop = 0;
    for (int s = 0; s < 2; s++) {
        int g = s == 0 ? fv2_g2 : fv2_g3;
        for (int y = 0; y < g; y++)
        for (int x = 0; x < g; x++) {
            const float *row = ofptr[s] +
                ((size_t)y * g + x) * FV2_CH;
            float cl = row[FV2_CLS_OFF + FV2_BIRD_CLS];
            if (cl < 0.01f) continue;
            for (int a = 0; a < 3; a++) {
                cell_t c;
                c.sc = row[FV2_OBJ_OFF + a] * cl;
                if (c.sc < 0.02f) continue;
                c.obj = row[FV2_OBJ_OFF + a];
                c.cl = cl;
                c.tx = row[a * 4]; c.ty = row[a * 4 + 1];
                c.tw = row[a * 4 + 2]; c.th = row[a * 4 + 3];
                c.s = s; c.x = x; c.y = y; c.a = a;
                /* keep the 16 largest                                  */
                int min_i = 0;
                for (int m = 1; m < ntop + 1 && m < 16; m++)
                    if (top[m].sc < top[min_i].sc) min_i = m;
                if (ntop < 16) { top[ntop++] = c; }
                else if (c.sc > top[min_i].sc) top[min_i] = c;
            }
        }
    }
    for (int m = 0; m < ntop; m++) {
        cell_t *c = &top[m];
        int g = c->s == 0 ? fv2_g2 : fv2_g3;
        int stride = c->s == 0 ? 16 : 32;
        float e2 = c->tw * 2.0f, e3 = c->th * 2.0f;
        float bx = ((c->tx * 2.0f - 0.5f) + c->x) * stride;
        float by = ((c->ty * 2.0f - 0.5f) + c->y) * stride;
        float bw = e2 * e2 * FV2_ANCH[c->s][c->a * 2];
        float bh = e3 * e3 * FV2_ANCH[c->s][c->a * 2 + 1];
        printf("  cell s%d(%dx%d grid) stride=%d a%d xy=(%d,%d) "
               "obj=%.3f cls14=%.3f sc=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
               c->s, g, g, stride, c->a, c->x, c->y, c->obj, c->cl,
               c->sc, bx, by, bw, bh);
    }
    /* GTs and their best candidate                                     */
    for (long g = gt0; g < NGT; g++) {
        if (GT[g].img != imgid) continue;
        printf("  GT   (%.0f,%.0f,%.0f,%.0f) wh=%.0fx%.0f\n",
               GT[g].x1, GT[g].y1, GT[g].x2, GT[g].y2,
               GT[g].x2 - GT[g].x1, GT[g].y2 - GT[g].y1);
        float best = -1.0f;
        long bj = -1;
        for (long j = 0; j < NGC; j++) {
            if (GC[j].img != imgid) continue;
            float iou = iou_box(GC[j].x1, GC[j].y1, GC[j].x2, GC[j].y2,
                                GT[g].x1, GT[g].y1, GT[g].x2, GT[g].y2);
            if (iou > best) { best = iou; bj = j; }
        }
        if (bj >= 0)
            printf("     bestIoU=%.2f pred=(%.0f,%.0f,%.0f,%.0f) sc=%.3f\n",
                   (double)best, GC[bj].x1, GC[bj].y1, GC[bj].x2,
                   GC[bj].y2, GC[bj].score);
        else
            printf("     no candidate whatsoever\n");
    }
}

int main(int argc, char **argv)
{
    const char *model = "yf2_coco352.onnx";
    const char *dir = "/home/root/fv2val";
    float floor_conf = 0.05f;          /* AP-curve lower bound          */
    float nms_iou = 0.40f;             /* reference test.py iou         */
    float match_iou = 0.5f;            /* AP IoU match threshold        */
    int threads = 4;
    int probe_idx = -1;
    const char *probe_name = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--floor") && i + 1 < argc)
            floor_conf = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--iou") && i + 1 < argc)
            nms_iou = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probe") && i + 1 < argc)
            probe_idx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probe-name") && i + 1 < argc)
            probe_name = argv[++i];
        else if (!strcmp(argv[i], "--feed") && i + 1 < argc)
            feed_resize = !strcmp(argv[++i], "resize");
        else if (!strcmp(argv[i], "--ap-iou") && i + 1 < argc)
            match_iou = (float)atof(argv[++i]);
    }

    char imgdir[512], lbdir[512];
    snprintf(imgdir, sizeof(imgdir), "%s/images", dir);
    snprintf(lbdir, sizeof(lbdir), "%s/labels", dir);

    GC = (cand_t *)xmalloc(sizeof(cand_t) * MAXCAND);
    GT = (gt_t *)xmalloc(sizeof(gt_t) * MAXGT);

    ort_init(model, threads);

    DIR *d = opendir(imgdir);
    if (!d) { fprintf(stderr, "opendir %s\n", imgdir); return 1; }

    char *iname[2048];
    int nimg = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL && nimg < 2048) {
        const char *nm = de->d_name;
        size_t ln = strlen(nm);
        if (ln < 5) continue;
        const char *e = nm + ln - 4;
        if (strcasecmp(e, ".jpg") && strcasecmp(e, ".jpeg")) continue;
        iname[nimg] = xstrdup(nm);
        nimg++;
    }
    closedir(d);
    fprintf(stderr, "yf2eval: %d images\n", nimg);

    double fwd_total = 0.0;
    char stembuf[256], lbpath[800], imgpath[800];
    for (int i = 0; i < nimg; i++) {
        if (!stem(iname[i], stembuf, sizeof(stembuf))) continue;
        snprintf(imgpath, sizeof(imgpath), "%s/%s", imgdir, iname[i]);
        snprintf(lbpath, sizeof(lbpath), "%s/%s.txt", lbdir, stembuf);

        /* first: get image dims for GT mapping; we re-derive by
         * decompressing inside run_one, so run_one returns AFTER we
         * need W/H... simplest: read dims here again via header only */
        unsigned char *jb = NULL;
        long jsz;
        FILE *f = fopen(imgpath, "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        jsz = ftell(f);
        fseek(f, 0, SEEK_SET);
        jb = (unsigned char *)xmalloc((size_t)jsz);
        size_t got = fread(jb, 1, (size_t)jsz, f);
        fclose(f);
        if (got != (size_t)jsz) { free(jb); continue; }
        tjhandle tj = tjInitDecompress();
        int W = 0, H = 0, sub = 0, cs = 0;
        if (tjDecompressHeader3(tj, jb, (unsigned long)jsz, &W, &H,
                                &sub, &cs) != 0) {
            free(jb);
            continue;
        }
        tjDestroy(tj);
        free(jb);
        jb = NULL;

        long gt_start = NGT;
        FILE *lf = fopen(lbpath, "rb");
        if (lf) {
            char line[512];
            while (fgets(line, sizeof(line), lf) && NGT < MAXGT) {
                int cls;
                float cx, cy, bw_, bh_;
                if (sscanf(line, "%d %f %f %f %f", &cls, &cx, &cy,
                           &bw_, &bh_) != 5)
                    continue;
                if (cls != 0) continue;          /* bird only          */
                gt_t *g = &GT[NGT++];
                g->img = i;
                g->x1 = (cx - bw_ / 2.0f) * (float)W;
                g->y1 = (cy - bh_ / 2.0f) * (float)H;
                g->x2 = (cx + bw_ / 2.0f) * (float)W;
                g->y2 = (cy + bh_ / 2.0f) * (float)H;
                g->used = 0;
            }
            fclose(lf);
        }
        run_one(imgpath, i, floor_conf, nms_iou, &fwd_total);
        if (probe_idx == i ||
            (probe_name && !strcmp(iname[i], probe_name)))
            probe_dump(gt_start, i);
        if ((i + 1) % 50 == 0)
            fprintf(stderr, "yf2eval: %d/%d done\n", i + 1, nimg);
    }

    qsort(GC, (size_t)NGC, sizeof(cand_t), cand_cmp);

    /* TP/FP in global score order                                      */
    long *tp = (long *)xmalloc(sizeof(long) * (NGC + 1));
    long *fp = (long *)xmalloc(sizeof(long) * (NGC + 1));
    long tpc = 0, fpc = 0;
    for (long i = 0; i < NGC; i++) {
        cand_t *c = &GC[i];
        int best = -1;
        float biou = match_iou;
        for (long g = 0; g < NGT; g++) {
            if (GT[g].img != c->img || GT[g].used) continue;
            float iou = iou_box(c->x1, c->y1, c->x2, c->y2,
                                GT[g].x1, GT[g].y1, GT[g].x2, GT[g].y2);
            if (iou > biou) { biou = iou; best = g; }
        }
        if (best >= 0) { GT[best].used = 1; tpc++; }
        else fpc++;
        tp[i] = tpc;
        fp[i] = fpc;
    }

    long ngt = NGT;
    FILE *out = stdout;
    fprintf(out, "images=%d detections=%ld gt=%ld fwd_avg_ms=%.1f feed=%s "
                 "ap_iou=%.2f\n",
            nimg, NGC, ngt, nimg ? fwd_total / nimg : 0.0,
            feed_resize ? "resize" : "letterbox", (double)match_iou);

    /* AP@0.5, all-point interpolation                                  */
    double ap = 0.0;
    double prev_r = 0.0;
    double *prec = (double *)xmalloc(sizeof(double) * (NGC + 1));
    double *rec = (double *)xmalloc(sizeof(double) * (NGC + 1));
    for (long i = 0; i < NGC; i++) {
        prec[i] = NGC ? (double)tp[i] / (double)(i + 1) : 0.0;
        rec[i] = ngt ? (double)tp[i] / (double)ngt : 0.0;
    }
    for (long i = NGC - 2; i >= 0; i--)
        if (prec[i] < prec[i + 1]) prec[i] = prec[i + 1];
    for (long i = 0; i < NGC; i++) {
        ap += (rec[i] - prev_r) * prec[i];
        prev_r = rec[i];
    }
    fprintf(out, "AP=%.3f  (at IoU %.2f)\n", ap, (double)match_iou);

    fprintf(out, "conf     P        R        F1\n");
    static const float cuts[] = { 0.05f, 0.10f, 0.15f, 0.20f, 0.25f,
                                  0.30f, 0.40f, 0.50f, 0.60f };
    for (unsigned u = 0; u < sizeof(cuts) / sizeof(cuts[0]); u++) {
        float cut = cuts[u];
        long i = 0;
        while (i < NGC && GC[i].score >= cut) i++;
        long t = i > 0 ? tp[i - 1] : 0;
        long f = i > 0 ? fp[i - 1] : 0;
        double p = (t + f) ? (double)t / (double)(t + f) : 0.0;
        double r = ngt ? (double)t / (double)ngt : 0.0;
        double f1 = (p + r) > 0 ? 2 * p * r / (p + r) : 0.0;
        fprintf(out, "%.2f   %.3f    %.3f    %.3f\n", cut, p, r, f1);
    }
    return 0;
}
