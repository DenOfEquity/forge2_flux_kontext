## Flux-Kontext extension for Forge2 webUI ##
### second implementation, because the first was too KISS ###

install:
**Extensions** tab, **Install from URL**, use URL for this repo

---

Zero input images = normal inference (and don't need this extension).

One input image = slower inference, more VRAM needed.

Two input images = even slower inference, even more VRAM needed.

There are three input size/crop options:
1. 'no change': *might* be useful for high resolution, but will require lots of VRAM and cause slower inference with large input images. *Might* be useful otherwise if you have already processed the input images to your own requirements. Generally, do not use. Does not resize or crop.
2. 'to output': the method used by previous versions. Resizes and centre-crops to match output resolution.
3. 'to BFL recommended': resizes and centre-crops to BFL preferred resolutions (all around 1MP), matching aspect ratio as best as possible. Generally, use this.

If your generation size and input sizes match, and conform to recommended resolutions, all options will result in the same output.

*reduce inputs to half width and height* option shrinks the input images to half width and height - may reduce details but improves performance. Applies after the above option. If generating at low resolution and using 'to BFL recommended', also using this option is recommended.

For reference, the BlackForestLabs preferred resolutions are:
* (672, 1568)
* (688, 1504)
* (720, 1456)
* (752, 1392)
* (800, 1328)
* (832, 1248)
* (880, 1184)
* (944, 1104)
* (1024, 1024)
* (1104, 944)
* (1184, 880)
* (1248, 832)
* (1328, 800)
* (1392, 752)
* (1456, 720)
* (1504, 688)
* (1568, 672)



---
