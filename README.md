## Flux-Kontext extension for Forge2 webUI ##
### second implementation, because the first was too KISS ###

install:
**Extensions** tab, **Install from URL**, use URL for this repo

---

Zero input images = normal inference (and don't need this extension).

One input image = slower inference, more VRAM needed.

Two input images = even slower inference, even more VRAM needed.

There are three input size/crop options:
1. 'no change': *might* be useful for high resolution, but will require lots of VRAM and cause slower inference with large input images. *Probably* useful if you have already processed the input images to your own requirements: perhaps manually pasting multiple references into one larger image. Generally, do not use. Does not resize or crop.
2. 'to output': the method used by previous versions. Resizes and centre-crops to match output resolution.
3. 'to BFL recommended': resizes and centre-crops to BFL preferred resolutions (all around 1MP), matching aspect ratio as best as possible. Generally, use this.

If your generation size and input sizes match, and conform to recommended resolutions, all options will result in the same output.

*reduce inputs to half width and height* option shrinks the input images to half width and height - may reduce details but improves performance. Applies after the above option. If generating at low resolution and using 'to BFL recommended', also using this option is recommended.

For reference, the BlackForestLabs preferred resolutions are:
* 672 × 1568 *and* 1568 × 672
* 688 × 1504 *and* 1504 × 688
* 720 × 1456 *and* 1456 × 720
* 752 × 1392 *and* 1392 × 752
* 800 × 1328 *and* 1328 × 800
* 832 × 1248 *and* 1248 × 832
* 880 × 1184 *and* 1184 × 880
* 944 × 1104 *and* 1104 × 944
* 1024 × 1024


---
[Prompting guide](https://docs.bfl.ai/guides/prompting_guide_kontext_i2i) from BlackForestLabs.

Euler Simple is a reliable sampler/scheduler combo, others will work too and may give better results.

In img2img, you need very high denoise to see effect so that probably isn't a good way to increase preservation of the original. Quick tests show that inpainting works well.

Works with [BlockCache](https://github.com/DenOfEquity/sd-forge-blockcache). I've had good results with TeaCache, 3 uncached starting steps, 0.25 threshold, 1 max. consecutive uncached - reducing inference time by 40%. (I haven't tried to find an optimal speed/quality trade-off, so ~~don't carve those parameters into stone and start a religion based on them~~ do your own testing.) Ideal parameters will vary based on sampler, image size.

Compatibility with FluxTools Canny/Depth LoRAs is 0/10 - generation will fail. I can increase this to 1/10 (maybe 1/100 would be more accurate) with a padding based hack which allows generation to proceed, but there is minimal following of the control image and significant quality degradation. Increasing LoRa strength to force following the control image has increasingly severe quality cost.
