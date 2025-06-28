import gradio
import torch, numpy

from modules import scripts, shared
from modules.ui_components import InputAccordion
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes
from backend.misc.image_resize import adaptive_resize

from backend.nn.flux import IntegratedFluxTransformer2DModel
from einops import rearrange, repeat


def patched_flux_forward(self, x, timestep, context, y, guidance=None, **kwargs):
    bs, c, h, w = x.shape
    input_device = x.device
    input_dtype = x.dtype
    patch_size = 2
    pad_h = (patch_size - x.shape[-2] % patch_size) % patch_size
    pad_w = (patch_size - x.shape[-1] % patch_size) % patch_size
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")
    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
    del x, pad_h, pad_w
    h_len = ((h + (patch_size // 2)) // patch_size)
    w_len = ((w + (patch_size // 2)) // patch_size)

    img_ids = torch.zeros((h_len, w_len, 3), device=input_device, dtype=input_dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=input_device, dtype=input_dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=input_device, dtype=input_dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    img_tokens = img.shape[1]

    if forgeKontext.latent != None:
        kh = forgeKontext.latentH
        kw = forgeKontext.latentW

        # why isn't this always h?
        if h+kw > w+kw:
            w_offset = w
            h_offset = 0
        else:
            w_offset = 0
            h_offset = h

        img = torch.cat([img, forgeKontext.latent.to(device=input_device, dtype=input_dtype)], dim=1)

        kh_len = ((kh + (patch_size // 2)) // patch_size)
        kw_len = ((kw + (patch_size // 2)) // patch_size)

        kontext_ids = torch.zeros((kh_len, kw_len, 3), device=input_device, dtype=input_dtype)
        kontext_ids[:, :, 0] = kontext_ids[:, :, 1] + 1
        kontext_ids[:, :, 1] += torch.linspace(h_offset, kh_len - 1 + h_offset, steps=kh_len, device=input_device, dtype=input_dtype)[:, None]
        kontext_ids[:, :, 2] += torch.linspace(w_offset, kw_len - 1 + w_offset, steps=kw_len, device=input_device, dtype=input_dtype)[None, :]
        kontext_ids = repeat(kontext_ids, "h w c -> b (h w) c", b=bs)
        
        img_ids = torch.cat([img_ids, kontext_ids], dim=1)

    txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)
    del input_device, input_dtype
    out = self.inner_forward(img, img_ids, context, txt_ids, timestep, y, guidance)
    del img, img_ids, txt_ids, timestep, context

    out = out[:, :img_tokens]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]

    del h_len, w_len, bs

    return out


# PREFERRED_KONTEXT_RESOLUTIONS = [
    # (672, 1568),
    # (688, 1504),
    # (720, 1456),
    # (752, 1392),
    # (800, 1328),
    # (832, 1248),
    # (880, 1184),
    # (944, 1104),
    # (1024, 1024),
    # (1104, 944),
    # (1184, 880),
    # (1248, 832),
    # (1328, 800),
    # (1392, 752),
    # (1456, 720),
    # (1504, 688),
    # (1568, 672),
# ]

class forgeKontext(scripts.Script):
    sorting_priority = 0
    original_forward = None
    latent = None

    def __init__(self):
        if forgeKontext.original_forward is None:
            forgeKontext.original_forward = IntegratedFluxTransformer2DModel.forward

    def title(self):
        return "Forge FluxKontext"

    def show(self, is_img2img):
        # useful in i2i ?
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            with gradio.Row():
                kontext_image1 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                kontext_image2 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
            with gradio.Row():
                swap12 = gradio.Button("swap Kontext 1 and 2")
                quarter = gradio.Checkbox(False, label="reduced size input(s)")

                def kontext_swap(imageA, imageB):
                    return imageB, imageA
                swap12.click(fn=kontext_swap, inputs=[kontext_image1, kontext_image2], outputs=[kontext_image1, kontext_image2])

        return enabled, kontext_image1, kontext_image2, quarter


    def process_before_every_sampling(self, params, *script_args, **kwargs):
        enabled, image1, image2, quarter = script_args
        if enabled and (image1 is not None or image2 is not None):
            if params.iteration > 0:    # batch count
                # setup done on iteration 0
                return

            if not params.sd_model.is_webui_legacy_model():
                x = kwargs['x']
                n, c, h, w = x.size()

                k_latents = []
                for image in [image1, image2]:
                    if image is not None:
                        k_image = image.convert('RGB')
                        k_image = numpy.array(k_image) / 255.0
                        k_image = numpy.transpose(k_image, (2, 0, 1))
                        k_image = torch.tensor(k_image).unsqueeze(0)

    #   width of input images must match latent width, padding OK
    #   height does not need to match

                        if quarter:
                            k_image = adaptive_resize(k_image, w*4, h*4, "lanczos", "center")
                            if image1 is None or image2 is None: # one input, pad
                                padr = (w*8) - k_image.shape[3]
                                k_image = torch.nn.functional.pad(k_image, (0, padr), mode='constant', value=0)
                        else:
                            k_image = adaptive_resize(k_image, w*8, h*8, "lanczos", "center")

                        k_latents.append(images_tensor_to_samples(k_image, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model))

                if quarter:
                    if image1 is not None and image2 is not None:   # two quarter-size inputs, stack horizontally
                        latent = torch.cat(k_latents, dim=3)
                    else:                                           # one quarter-size input, already padded
                        latent = k_latents[0]
                else:                                               # one or two full-size inputs, stack vertically
                    latent = torch.cat(k_latents, dim=2)

                forgeKontext.latentH = latent.shape[2]
                forgeKontext.latentW = latent.shape[3]
                forgeKontext.latent = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                
                IntegratedFluxTransformer2DModel.forward = patched_flux_forward

        return

    def postprocess(self, params, processed, *args):
        enabled = args[0]
        if enabled:
            forgeKontext.latent = None
            IntegratedFluxTransformer2DModel.forward = forgeKontext.original_forward

        return