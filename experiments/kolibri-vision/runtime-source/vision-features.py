"""Python bridge to the frozen native feature tap, for training only."""
import ctypes
from pathlib import Path

import numpy as np


class VisionFeatures:
    def __init__(self, root, *, gpu=False, min_tokens=1024, max_tokens=4096, text_model=None, projector=None):
        root = Path(root)
        self.library = ctypes.CDLL(str(root / "runtime-source/libkolibri-vision-features.so"))
        self.library.kolibri_vision_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
        self.library.kolibri_vision_open.restype = ctypes.c_void_p
        self.library.kolibri_vision_encode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.library.kolibri_vision_encode.restype = ctypes.c_int
        self.library.kolibri_vision_projected.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.library.kolibri_vision_projected.restype = ctypes.c_int
        self.library.kolibri_vision_last_error.restype = ctypes.c_char_p
        self.library.kolibri_vision_close.argtypes = [ctypes.c_void_p]
        self.handle = self.library.kolibri_vision_open(
            str(text_model or root / "compatibility/head128/afmoe-interface-fixture.gguf").encode(),
            str(projector or root / "compatibility/qwen-vision-prototype32.gguf").encode(), gpu, min_tokens, max_tokens)
        if not self.handle:
            raise RuntimeError(self.library.kolibri_vision_last_error().decode())

    def encode(self, image):
        rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")), dtype=np.uint8)
        pointer, tokens, width = ctypes.POINTER(ctypes.c_float)(), ctypes.c_int(), ctypes.c_int()
        status = self.library.kolibri_vision_encode(self.handle, rgb.shape[1], rgb.shape[0],
            rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), ctypes.byref(pointer), ctypes.byref(tokens), ctypes.byref(width))
        if status:
            raise RuntimeError(self.library.kolibri_vision_last_error().decode())
        # Copy before the next encode reuses native storage.
        result = np.ctypeslib.as_array(pointer, shape=(tokens.value * width.value,)).copy()
        return result.reshape(tokens.value, width.value)

    def projected(self):
        pointer, tokens, width = ctypes.POINTER(ctypes.c_float)(), ctypes.c_int(), ctypes.c_int()
        if self.library.kolibri_vision_projected(self.handle, ctypes.byref(pointer), ctypes.byref(tokens), ctypes.byref(width)):
            raise RuntimeError(self.library.kolibri_vision_last_error().decode())
        return np.ctypeslib.as_array(pointer, shape=(tokens.value * width.value,)).copy().reshape(tokens.value, width.value)

    def close(self):
        if self.handle:
            self.library.kolibri_vision_close(self.handle)
            self.handle = None
