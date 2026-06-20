import os
import uuid
import streamlit as st
from bi_predict import get_prediction

st.set_page_config(page_title="Urban Sound Classifier", layout="centered")
st.title("🔊 Классификатор городских звуков")
st.write("Запишите звук с микрофона или загрузите готовый WAV-файл.")

uploaded_file = st.file_uploader("Выбрать файл с диска", type=["wav", "mp3"])
recorded_file = st.audio_input("Записать через микрофон")

active_file = uploaded_file if uploaded_file is not None else recorded_file

if active_file is not None:
    if st.button("🔥 Распознать звук", type="primary"):
        with st.spinner("Нейросеть MiniVGG анализирует аудио..."):
            temp_path = f"temp_{uuid.uuid4()}.wav"
            try:
                with open(temp_path, "wb") as f:
                    f.write(active_file.read())

                # Прямой вызов функции из соседнего файла!
                result_class = get_prediction(temp_path)
                st.success(f"**Результат:** {result_class}")
                
            except Exception as e:
                st.error(f"Ошибка при анализе: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)