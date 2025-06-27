## Flux-Kontext extension for Forge2 webUI ##
### initial janky implementation ###

install:
**Extensions** tab, **Install from URL**, use URL for this repo

---
Input images are automatically resized and centre-cropped to match width/height.

Zero input images = normal inference (and don't need this extension).

One input image = slower inference, more VRAM needed.

Two input images = even slower inference, even more VRAM needed.

*reduced size inputs(s)* option shrinks the input images to half width and height - may reduce fine details but improves performance

---
Examples, with ***Q2_K GGUF*** and ***not enough steps*** at ***too low resolution***, Euler Simple:

(source images from the BFL examples)

top row: input on left, prompt was *change the "YOU HAD ME AT CONTEXT" to "YOU HAD ME AT Q2 GGUF"* (15 steps)

middle row: two inputs, left and middle, prompt was *the woman and the man meet in a forest while hiking, maintain clothing and character details* (10 steps)

bottom row: input on left, middle prompt *make it realistic* (10 steps), right prompt *make this realistic and make the man's jacket green* (15 steps)

![kontext](https://github.com/user-attachments/assets/67f58f56-efec-4581-be10-b897c1ace3c6)
