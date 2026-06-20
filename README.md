# 🔊 Urban Sound Classifier

Репозиторий содержит пет-проект по автоматическому распознаванию и классификации городских звуков в реальном времени. В основе решения лежит кастомная сверточная нейросеть на фреймворке **PyTorch**, а интерфейс реализован в виде веб-приложения на **Streamlit**.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-yellow)](https://huggingface.co/spaces)

---

## 🚀 Демо-версия
Проект развернут и доступен для тестирования в браузере:  
👉 **[Ссылка на Hugging Face Space](https://ortus776-sound-classification.hf.space)** *(замените на вашу ссылку)*

---

## 📊 Описание и возможности
Модель обучена классифицировать аудиозаписи на **10 категорий** (на основе известного датасета `UrbanSound8K`):
* Кондиционер (`air_conditioner`)
* Автомобильный гудок (`car_horn`)
* Играющие дети (`children_playing`)
* Лай собаки (`dog_bark`)
* Бурение (`drilling`)
* Холостой ход двигателя (`engine_idling`)
* Выстрел (`gun_shot`)
* Отбойный молоток (`jackhammer`)
* Сирена (`siren`)
* Уличная музыка (`street_music`)

**Функционал интерфейса:**
1. Загрузка готовых аудиофайлов в форматах `.wav` и `.mp3`.
2. Запись звука напрямую с микрофона вашего устройства прямо в браузере.

---

## Техническая реализация

### Архитектура модели (`model_def.py`)
Используется кастомная нейросеть `MiniVGGClassifier`. Она вдохновлена классической архитектурой VGG-16 и состоит из:
* **4 сверточных блоков** (каждый включает в себя слои `Conv2D` -> `ReLU` -> `BatchNorm2d` -> `MaxPool2d`).
* Слоя **AdaptiveAvgPool2d(1)** на выходе из сверточной части для детекции признаков независимо от размера спектрограммы.
* Полносвязного классификатора со слоем **Dropout (0.3)** для борьбы с переобучением.
* Веса сети инициализированы по методу *Kaiming Normal*.

### Пайплайн обработки звука (`bi_predict.py` & `model_def.py`)
Перед тем как попасть в нейросеть, аудиосигнал проходит строгую стандартизацию через утилиту `AudioUtil`:
1. Загрузка файла и извлечение сигнала с помощью `soundfile`.
2. Ресемплинг до единой частоты дискретизации — **44 100 Гц**.
3. Приведение к двум аудиоканалам (**Stereo**).
4. Обрезка или дополнение тишиной (Pad/Truncate) до фиксированной длины **4 секунды**.
5. Преобразование волны в **Mel-спектрограмму** (`torchaudio.transforms.MelSpectrogram`) с разрешением 64 mel-бэнда.
6. Логарифмирование амплитуды (перевод в децибелы) и нормализация данных.

---

## 📦 Инструкция по локальному запуску

Клонируйте репозиторий и установите зависимости:

```bash
# Клонирование проекта
git clone https://github.com/ortus776/sound_classifier
cd sound_classifier

# Установка библиотек
pip install -r requirements.txt

# Запуск веб-интерфейса Streamlit
streamlit run ui.py