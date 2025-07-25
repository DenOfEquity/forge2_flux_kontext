import gradio
import torch, numpy
import base64
import io
from PIL import Image

from modules import scripts, shared
from modules.ui_components import InputAccordion, ToolButton
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes
from backend.misc.image_resize import adaptive_resize

from backend.nn.flux import IntegratedFluxTransformer2DModel
from einops import rearrange, repeat


def patched_flux_forward(self, x, timestep, context, y, guidance=None, **kwargs):
    bs, c, h, w = x.shape

    if c != 16:
        # fix for the case where user is also using FluxTools extension, x has extra channels
        # spam message every step, so user might pay attention, or silently fix?
        # print ("\n[Kontext] ERROR: too many channels, excess channels will be stripped.\n")
        x = x[:, :16, :, :]

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
        img = torch.cat([img, forgeKontext.latent.repeat(bs, 1, 1)], dim=1)
        img_ids = torch.cat([img_ids, forgeKontext.ids.repeat(bs, 1, 1)], dim=1)

    txt_ids = torch.zeros((bs, context.shape[1], 3), device=input_device, dtype=input_dtype)
    del input_device, input_dtype
    out = self.inner_forward(img, img_ids, context, txt_ids, timestep, y, guidance)
    del img, img_ids, txt_ids, timestep, context

    out = out[:, :img_tokens]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]

    del h_len, w_len, bs

    return out


PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

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
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            gradio.Markdown("Select a FluxKontext model in the **Checkpoint** menu. Add reference image(s) here.")
            with gradio.Row():
                with gradio.Column():
                    kontext_image1 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                    with gradio.Row():
                        image1_info = gradio.Textbox(value="", show_label=False, interactive=False, max_lines=1)
                        image1_send = ToolButton(value='\U0001F4D0', interactive=False, variant='tertiary')
                        image1_dims = gradio.Textbox(visible=False, value='0')
                with gradio.Column():
                    kontext_image2 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                    with gradio.Row():
                        swap12 = ToolButton("\U000021C4")
                        image2_info = gradio.Textbox(value="", show_label=False, interactive=False, max_lines=1)
                        image2_send = ToolButton(value='\U0001F4D0', interactive=False, variant='tertiary')
                        image2_dims = gradio.Textbox(visible=False, value='0')

                def get_dims(image):
                    if image:
                        w = image.size[0]
                        h = image.size[1]
                        sw = 16 * ((15 + w) // 16)
                        sh = 16 * ((15 + h) // 16)
                        return f"{image.size[0]} × {image.size[1]} ({sw} × {sh})", gradio.update(interactive=True, variant='secondary'), f'{sw},{sh}'
                    else:
                        return  "", gradio.update(interactive=False, variant='tertiary'), '0'

                kontext_image1.change(fn=get_dims, inputs=kontext_image1, outputs=[image1_info, image1_send, image1_dims], show_progress=False)
                kontext_image2.change(fn=get_dims, inputs=kontext_image2, outputs=[image2_info, image2_send, image2_dims], show_progress=False)
 
                if self.is_img2img:
                    tab_id = gradio.State(value='img2img')
                else:
                    tab_id = gradio.State(value='txt2img')
 
                image1_send.click(fn=None, js="kontext_set_dimensions", inputs=[tab_id, image1_dims], outputs=None)
                image2_send.click(fn=None, js="kontext_set_dimensions", inputs=[tab_id, image2_dims], outputs=None)


            with gradio.Row():
                sizing = gradio.Dropdown(label="Kontext image size/crop", choices=["no change", "to output", "to BFL recommended"], value="to BFL recommended")
                reduce = gradio.Checkbox(False, info="This reduction is independent of the size/crop setting.", label="reduce to half width and height")


            def kontext_swap(imageA, imageB):
                return imageB, imageA
            swap12.click(fn=kontext_swap, inputs=[kontext_image1, kontext_image2], outputs=[kontext_image1, kontext_image2])

        return enabled, kontext_image1, kontext_image2, sizing, reduce


    def process_before_every_sampling(self, params, *script_args, **kwargs):
        enabled, image1, image2, sizing, reduce = script_args
        if enabled and (image1 is not None or image2 is not None):
            if params.iteration > 0:    # batch count
                # setup done on iteration 0
                return

            if not params.sd_model.is_webui_legacy_model():
                x = kwargs['x']
                n, c, h, w = x.size()
                input_device = x.device
                input_dtype = x.dtype

# todo?: cache results to avoid VAE encode

                k_latents = []
                k_ids = []
                accum_h = 0
                accum_w = 0
                # extra_mem = 0   # test if useful for large inputs
                for image in [image1, image2]:
                    if image is not None:
                        # --- START: ADDED CODE FOR API/Base64 SUPPORT ---
                        if isinstance(image, str):
                            try:
                                # If image is a string, it's Base64 from the API. Decode it.
                                decoded_image = base64.b64decode(image)
                                image = Image.open(io.BytesIO(decoded_image))
                            except Exception as e:
                                print(f"[Kontext] Error decoding Base64 image from API: {e}")
                                continue # Skip to the next image if this one is invalid
                        # --- END: ADDED CODE FOR API/Base64 SUPPORT ---

                        k_image = image.convert('RGB')
                        k_image = numpy.array(k_image) / 255.0
                        k_image = numpy.transpose(k_image, (2, 0, 1))
                        k_image = torch.tensor(k_image).unsqueeze(0)

                        # it seems that the img_id is always 1 for the context images
                        # so resize and combine here instead of in the forward function

                        # only go through the resize process if image is not already desired size
                        match sizing:
                            case "no change":
                                k_width = k_image.shape[3]
                                k_height = k_image.shape[2]
                            case "to output":
                                k_width = w * 8
                                k_height = h * 8
                            case "to BFL recommended":  # this snippet from ComfyUI
                                k_width = k_image.shape[3]
                                k_height = k_image.shape[2]
                                aspect_ratio = k_width / k_height
                                _, k_width, k_height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)

                        if reduce:
                            k_width //= 2
                            k_height //= 2

                        if k_image.shape[3] != k_width or k_image.shape[2] != k_height:
                            print ("[Kontext] resizing and center-cropping input to: ", k_width, k_height)
                            k_image = adaptive_resize(k_image, k_width, k_height, "lanczos", "center")
                        else:
                            print ("[Kontext] no image resize needed")

                        # VAE encode each input image - combined image could be large
                        k_latent = images_tensor_to_samples(k_image, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model)

                        # pad if needed - latent width and height must be multiple of 2
                        # could just adjust the resize to be *16, but the padding might be better for images that need only one extra row/col
                        patch_size = 2
                        pad_h = k_latent.shape[2] % patch_size
                        pad_w = k_latent.shape[3] % patch_size
                        k_latent = torch.nn.functional.pad(k_latent, (0, pad_w, 0, pad_h), mode="circular")

                        k_latents.append(rearrange(k_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size))
                        # imgs are combined in rearranged dimension 1 - so width/height can be independant of main latent and other inputs

                        latentH = k_latent.shape[2]
                        latentW = k_latent.shape[3]
                        # extra_mem += k_latent.shape[0] * k_latent.shape[1] * k_latent.shape[2] * k_latent.shape[3] * 4 * 1640 # tune this?
                       
                        kh_len = ((latentH + (patch_size // 2)) // patch_size)
                        kw_len = ((latentW + (patch_size // 2)) // patch_size)

                        # this offset + accumulation is based on Comfy.
                        offset_h = 0
                        offset_w = 0
                        if kh_len + accum_h > kw_len + accum_w:
                            offset_w = accum_w
                        else:
                            offset_h = accum_h

                        k_id = torch.zeros((kh_len, kw_len, 3), device=input_device, dtype=input_dtype)
                        k_id[:, :, 0] = 1
                        k_id[:, :, 1] += torch.linspace(offset_h, offset_h + kh_len - 1, steps=kh_len, device=input_device, dtype=input_dtype)[:, None]
                        k_id[:, :, 2] += torch.linspace(offset_w, offset_w + kw_len - 1, steps=kw_len, device=input_device, dtype=input_dtype)[None, :]

                        accum_w = max(accum_w, kw_len + offset_w)
                        accum_h = max(accum_h, kh_len + offset_h)

                        k_ids.append(repeat(k_id, "h w c -> b (h w) c", b=1)) # moved batch into patched_flux_forward

                forgeKontext.latent = torch.cat(k_latents, dim=1)
                forgeKontext.ids = torch.cat(k_ids, dim=1)

                # might as well move these now
                forgeKontext.latent.to(device=input_device, dtype=input_dtype)
                forgeKontext.ids.to(device=input_device, dtype=input_dtype)

                del k_latent, k_id


                IntegratedFluxTransformer2DModel.forward = patched_flux_forward

                # unet = params.sd_model.forge_objects.unet
                # print ("[Kontext] reserving extra memory (MB):", round(extra_mem/(1024*1024), 2))
                # unet.add_extra_preserved_memory_during_sampling(extra_mem)

        return

    def postprocess(self, params, processed, *args):
        enabled = args[0]
        if enabled:
            forgeKontext.latent = None
            forgeKontext.ids = None
            IntegratedFluxTransformer2DModel.forward = forgeKontext.original_forward

        return