"""DIAGNOSTIC ONLY (build-time): diff native fv2 torch forward vs my
ONNX export on the identical input, to locate any export divergence.
Not part of the runtime (board runs C only)."""
import sys
import numpy as np
import torch
import cv2
import onnxruntime as ort

FV2 = r"C:\Users\Jay singh\AppData\Local\Temp\fv2"
sys.path.insert(0, FV2)
import model.detector          # noqa: E402
import utils.utils             # noqa: E402

cfg = utils.utils.load_datafile(FV2 + r"\data\coco.data")
m = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True)
m.load_state_dict(torch.load(
    FV2 + r"\modelzoo\coco2017-0.241078ap-model.pth",
    map_location="cpu", weights_only=False))
m.eval()

img = cv2.imread(r"D:\samsung\BIRD-DETECTION\training\data_detection"
                 r"\val\images\24_647.jpg")
res = cv2.resize(img, (352, 352))
x = torch.from_numpy(res.reshape(1, 352, 352, 3).transpose(0, 3, 1, 2))
x = x.float() / 255.0

with torch.no_grad():
    reg2, obj2, cls2, reg3, obj3, cls3 = m(x)

on = obj2.sigmoid()[0]                 # [3,22,22]
cl = cls2.softmax(1)[0, 14]            # [22,22]
sc = (on * cl).max(0).values           # [22,22], best anchor per cell
flat = sc.reshape(-1)
top = torch.argsort(flat, descending=True)[:8]
print("NATIVE top8 (scale2, obj*cls14):")
for t in top:
    y, xx = divmod(int(t), 22)
    a = int(torch.argmax(on[:, y, xx]))
    print("  cell(%d,%d) a%d obj=%.3f cls14=%.3f sc=%.3f" %
          (xx, y, a, on[a, y, xx], cl[y, xx], flat[t]))

s = ort.InferenceSession(
    r"D:\samsung\BIRD-DETECTION\cpp\trial-fv2\yf2_coco352.onnx",
    providers=["CPUExecutionProvider"])
o2, o3 = s.run(None, {"images": x.numpy()})
onx = o2[0]                            # [22,22,95]
scn = onx[:, :, 12:15] * onx[:, :, 15 + 14:16 + 14][:, :, 0:1]
scn = scn.max(axis=2)
print("ONNX   top8 (scale2):")
fl2 = scn.reshape(-1)
top = torch.argsort(torch.from_numpy(fl2), descending=True)[:8]
for t in top:
    y, xx = divmod(int(t), 22)
    a = int(np.argmax(onx[y, xx, 12:15]))
    print("  cell(%d,%d) obj=%.3f cls14=%.3f sc=%.3f" %
          (xx, y, onx[y, xx, 12 + a], onx[y, xx, 15 + 14], scn[y, xx]))

# per-channel diff summary
n_obj = on.numpy()
n_cls = cl.numpy()
d_obj = np.abs(n_obj - onx[:, :, 12:15].transpose(2, 0, 1)).max()
d_cls = np.abs(n_cls - onx[:, :, 15 + 14]).max()
print("max |obj diff| native-vs-onnx:", d_obj)
print("max |cls14 diff| native-vs-onnx:", d_cls)
