## Flux-Kontext extension for Forge2 webUI ##
### second implementation, because the first was too KISS ###

install:
**Extensions** tab, **Install from URL**, use URL for this repo

---
Input images are automatically resized and centre-cropped to match width/height.

Zero input images = normal inference (and don't need this extension).

One input image = slower inference, more VRAM needed.

Two input images = even slower inference, even more VRAM needed.

*reduced size inputs(s)* option shrinks the input images to half width and height - may reduce fine details but improves performance

---
