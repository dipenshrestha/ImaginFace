# Create your views here.
import os
import subprocess
from django.shortcuts import render


def home_screen_view(request):
    return render(request, "personal/home.html")

# def home_screen_view(request):
#     #Run the Streamlit script as a subprocess
#     streamlit_script_path = os.path.join(os.path.dirname(__file__), "dfgan.py")
#     subprocess.Popen(["streamlit", "run", streamlit_script_path])
#     return render(request,"personal/home.html")

def helppage(request):
    return render(request,"help.html")

def contactpage(request):
    return render(request,"contact-us.html")

import subprocess 
from .dcgan import *
def process(request):
    if request.method == 'POST':
        input_text = request.POST.get('textinput')
        # gan_model = form.cleaned_data['gan_model']
        gan_model = request.POST.get('ganModel')

        if gan_model == 'dcganold':
            # Generate faces
            test_noise = torch.randn(size=(1, 100))
            test_embeddings = sentence_encoder.convert_text_to_embeddings([input_text])
            test_image = model1(test_noise, test_embeddings).detach().cpu()
            # t = show_grid(torchvision.utils.make_grid(test_image, normalize=True, nrow=1))

            grid_image=make_grid(test_image, normalize=True, nrow=1)
            pil_image=Image.fromarray(grid_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()) # Convert the PyTorch tensor to a PIL image
            buffered = io.BytesIO() # Convert the PIL image to a base64 encoded string
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Return the rendered template with the image path
            return render(request, 'personal/home.html', {'generated_image': img_str, 'caption':input_text})

        elif gan_model == 'dcgannew':
            # Generate faces
            test_noise = torch.randn(size=(1, 100))
            test_embeddings = sentence_encoder.convert_text_to_embeddings([input_text])
            test_image = model2(test_noise, test_embeddings).detach().cpu()
            # t = show_grid(torchvision.utils.make_grid(test_image, normalize=True, nrow=1))

            grid_image=make_grid(test_image, normalize=True, nrow=1)
            pil_image=Image.fromarray(grid_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()) # Convert the PyTorch tensor to a PIL image
            buffered = io.BytesIO() # Convert the PIL image to a base64 encoded string
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Return the rendered template with the image path
            return render(request, 'personal/home.html', {'generated_image': img_str, 'caption':input_text})

        return render(request, 'personal/home.html')

    return render(request, 'personal/home.html')