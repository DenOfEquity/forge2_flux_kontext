import gradio
import torch, numpy

from modules import scripts, shared
from modules.ui_components import InputAccordion#, ToolButton
from modules.script_callbacks import on_cfg_denoiser, on_cfg_denoised, remove_current_script_callbacks
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes
from backend.misc.image_resize import adaptive_resize

import k_diffusion
from modules import sd_schedulers
def patched_to_d(x, s, d):
    return (x - d[:, :, 0:x.shape[2], :]) / s
sd_schedulers.to_d = patched_to_d
k_diffusion.sampling.to_d = patched_to_d


class forgeKontext(scripts.Script):
    sorting_priority = 0

    def title(self):
        return "Forge2 FluxKontext"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:

            gradio.Markdown("Select Kontext image(s).")
            with gradio.Row():
                kontext_image1 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                kontext_image2 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])

            with gradio.Row():
                swap12 = gradio.Button("swap Kontext 1 and 2", scale=0)

                def kontext_swap(imageA, imageB):
                    return imageB, imageA
                swap12.click(fn=kontext_swap, inputs=[kontext_image1, kontext_image2], outputs=[kontext_image1, kontext_image2])

        return enabled, kontext_image1, kontext_image2


    def process_before_every_sampling(self, params, *script_args, **kwargs):
        enabled, image1, image2 = script_args
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

                        k_image = adaptive_resize(k_image, w*8, h*8, "lanczos", "center")

                        k_latents.append(images_tensor_to_samples(k_image, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model))

                forgeKontext.latent = torch.cat(k_latents, dim=2)


                def apply_kontext(self):
                    forgeKontext.true_h = None
                    if forgeKontext.latent is not None:
                        forgeKontext.true_h = self.x.shape[2]
                        self.x = torch.cat([self.x, forgeKontext.latent], dim=2)
                def remove_kontext(self):
                    if forgeKontext.true_h is not None:
                        self.x = self.x[:, :, 0:forgeKontext.true_h, :]

                on_cfg_denoiser(apply_kontext)
                on_cfg_denoised(remove_kontext)

        return


    def postprocess(self, params, processed, *args):
        enabled = args[0]
        if enabled:
            forgeKontext.latent = None
            forgeKontext.true_h = None
            remove_current_script_callbacks()

        return
