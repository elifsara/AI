import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('my_fv_model.h5')

def process_image(img):
    img=img.resize((170,170))  #boyutunu 170*170 pixel yaptık
    img=np.array(img)
    img=img/255.0  #Normalize ettik
    img=np.expand_dims(img,axis=0)
    return img

st.title('Meyze Sebze Siniflandirmasi :tomato:')
st.write('Resim seç ve hangi meyve/sebze olduğunu tahmin etsin')

file=st.file_uploader('Bir Resim Sec', type=['jpg','jpeg','png'])

if file is not None:
    img=Image.open(file)
    st.image(img,caption='yuklenen resim')
    image=process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)

    class_names=['apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot',
         'cauliflower','chilli pepper','corn','cucumber','eggplant','garlic','ginger','grapes',
         'jalepeno','kiwi','lemon','lettuce','mango','onion','orange','paprika','pear','peas',
         'pineapple','pomegranate','potato','raddish','soy beans','spinach','sweetcorn','sweetpotato',
         'tomato','turnip','watermelon']
    st.write(class_names[predicted_class])