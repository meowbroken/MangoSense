from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from django.http import JsonResponse
from django.conf import settings
import numpy as np
import tensorflow as tf
from PIL import Image
import gc

IMG_SIZE = (224, 224)
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge',
               'Healthy', 'Powdery Mildew', 'Sooty Mould']


model_path = str(settings.MODEL_PATH)
model = tf.keras.models.load_model(model_path)

treatment_suggestions = {
    'Anthracnose': 'The diseased twigs should be pruned and burnt along with fallen leaves. Spraying twice with Carbendazirn (Bavistin 0.1%) at 15 days interval during flowering controls blossom infection. Spraying of copper fungicides (0.3%) is recommended for the control of foliar infection. Postharvest disease of mango caused by anthracnose could be controlled by dip treatment of fruits in 0 Carbendazim (0.1%) in hot water at 52 C for 15 minutes. ',
    'Bacterial Canker': 'Three sprays of Streptocycline (0.01%) or Agrimycin-100 (0.01%) after first visual symptom at 10 day intervals and monthly sprays of Carbendazim (Bavistin 0.1%) or Copper Oxychloride (0.3%) are effective in controlling the disease. ',
    'Cutting Weevil': 'Use recommended insecticides and remove infested plant material.',
    'Die Back': ' Pruning of the diseased twigs 2-3 inches below the affected portion and spraying Copper Oxychloride (0.3%) on infected trees controls the disease. The cut ends of the pruned twigs are pasted with Copper Oxychloride (0.3%).',
    'Gall Midge': 'Remove and destroy infested fruits; use appropriate insecticides.',
    'Healthy': 'No treatment needed. Maintain good agricultural practices.',
    'Powdery Mildew': 'Alternate spraying of Wettable sulphur 0.2 per cent (2 g Sulfex/litre), Tridemorph O.1 per cent (1 ml Calixin/litre) and Bavistin @ 0.1 %  at 15 days interval are recommended for effective control of the disease. The first spray is to be given at panicle emergence stage.',
    'Sooty Mould': 'Pruning of affected branches and their prompt destruction followed by spraying of Wettasulf (0.2% )+ Metacid (0.1 %)+ gum acacia (0.3%) helps to control the disease.'
}

def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img.close()
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array) # img/255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@api_view(['POST'])
@parser_classes([MultiPartParser])
def predict_image(request):
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image uploaded'}, status=400)
    try:
        img_array = preprocess_image(request.FILES['image'])
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index]) * 100
        treatment = treatment_suggestions.get(predicted_class, "No treatment information available.")
        gc.collect()
        return JsonResponse({
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'probabilities': prediction[0].tolist(),
            'treatment': treatment
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
