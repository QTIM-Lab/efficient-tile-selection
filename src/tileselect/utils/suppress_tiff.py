"""
Suppress libtiff C-level warnings triggered by openslide on certain SVS files
Import this module *before* importing openslide.

Usage:
    from tileselect.utils.suppress_tiff import openslide
"""

import os
import ctypes
import contextlib

# 1. Disable TIFFSetWarningHandler at C level before openslide loads libtiff.
# (This doesn't always work if openslide statically links its own libtiff)
for _libtiff_name in ('libtiff.so.6', 'libtiff.so.5', 'libtiff.so'):
    try:
        _libtiff = ctypes.CDLL(_libtiff_name)
        for _handler_name in ('TIFFSetWarningHandler', 'TIFFSetWarningHandlerExt'):
            try:
                _set_warning_handler = getattr(_libtiff, _handler_name)
                _set_warning_handler.argtypes = [ctypes.c_void_p]
                _set_warning_handler.restype = ctypes.c_void_p
                _set_warning_handler(None)
            except Exception:
                pass
    except Exception:
        pass

# 2. Context manager to redirect fd 2 (stderr) at OS level — the only way
#    to silence C-level stderr output that Python's warnings module cannot catch.
@contextlib.contextmanager
def suppress_fd_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull_fd)


def _suppressed_call(fn, *args, **kwargs):
    with suppress_fd_stderr():
        return fn(*args, **kwargs)

# 3. Import openslide with stderr suppressed to catch any import-time warnings.
with suppress_fd_stderr():
    import openslide

# 4. Monkey-patch OpenSlide to suppress warnings during file open and reads.
_original_OpenSlide = openslide.OpenSlide
_original_open_slide = openslide.open_slide

class SuppressedOpenSlide(_original_OpenSlide):
    def __init__(self, filename):
        _suppressed_call(super().__init__, filename)

    @classmethod
    def detect_format(cls, filename):
        return _suppressed_call(_original_OpenSlide.detect_format, filename)

    def close(self):
        return _suppressed_call(super().close)

    @property
    def level_count(self):
        return _suppressed_call(lambda: super(SuppressedOpenSlide, self).level_count)

    @property
    def level_dimensions(self):
        return _suppressed_call(lambda: super(SuppressedOpenSlide, self).level_dimensions)

    @property
    def dimensions(self):
        return _suppressed_call(lambda: super(SuppressedOpenSlide, self).dimensions)

    @property
    def level_downsamples(self):
        return _suppressed_call(lambda: super(SuppressedOpenSlide, self).level_downsamples)

    @property
    def properties(self):
        return _suppressed_call(lambda: super(SuppressedOpenSlide, self).properties)

    @property
    def associated_images(self):
        return _suppressed_call(lambda: super(SuppressedOpenSlide, self).associated_images)

    @property
    def color_profile(self):
        return _suppressed_call(lambda: super(SuppressedOpenSlide, self).color_profile)

    def get_thumbnail(self, size):
        return _suppressed_call(super().get_thumbnail, size)

    def read_region(self, location, level, size):
        return _suppressed_call(super().read_region, location, level, size)

    def get_best_level_for_downsample(self, downsample):
        return _suppressed_call(super().get_best_level_for_downsample, downsample)

    def set_cache(self, cache):
        return _suppressed_call(super().set_cache, cache)


def suppressed_open_slide(filename):
    return _suppressed_call(_original_open_slide, filename)

openslide.OpenSlide = SuppressedOpenSlide
openslide.open_slide = suppressed_open_slide
