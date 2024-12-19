from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.video import Video
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from kivy.uix.image import Image as KivyImage
import io

# Model tanımı
class TumorClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TumorClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Modeli yükleme
model_path = r"C:\Users\erdem\OneDrive\Masaüstü\Brain Tumor Project\best_model.pth"
model = TumorClassifier(num_classes=4)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Veri dönüştürme (resim ön işleme)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sınıf açıklamaları
class_descriptions = {
    "Glioma": "Glioma, beynin veya omurganın glial hücrelerinde başlayan bir tümör türüdür. Gliomlar, tüm beyin tümörlerinin yaklaşık %30'unu ve habis tümörlerin %80'ini oluşturur.",
    "Meningioma": "Meningioma, beyin ve omuriliği çevreleyen zarların hücrelerinde başlayan genellikle iyi huylu bir tümördür.",
    "No Tumor": "Seçilen MRI görüntüsünde tümör bulunmamaktadır.",
    "Pituitary": "Pituitary tümörleri, hipofiz bezinde başlayan genellikle iyi huylu tümörlerdir. Hormon üretimini etkileyebilir."
}

# Tahmin fonksiyonu
def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    transformed_image = data_transforms(image).unsqueeze(0)
    with torch.no_grad():
        output = model(transformed_image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Kivy Arayüzü
class TumorApp(App):
    def build(self):
        layout = FloatLayout()

        # Arka plan videosu
        video = Video(source="background.mp4", state='play', options={'eos': 'loop'})
        video.size = (1920, 1080)  # Video boyutu
        video.pos = (0, 0)  # Video konumu
        layout.add_widget(video)

        # Başlık etiketi
        self.title_label = Label(text="Beyin Tümörü Analizeri", font_size='24sp', color=(1, 1, 1, 1), size_hint=(None, None), size=(300, 50), pos_hint={'x': 0.5, 'top': 1})
        layout.add_widget(self.title_label)

        # Görsel seçme butonu
        self.btn_select = Button(text="Görsel Seç", font_size='16sp', size_hint=(None, None), size=(200, 50), pos_hint={'x': 0.575, 'top': 0.9})
        self.btn_select.bind(on_press=self.select_image)
        layout.add_widget(self.btn_select)

        # Görsel gösterimi
        self.image_label = KivyImage(size_hint=(None, None), size=(200, 200), pos_hint={'x': 0.575, 'top': 0.7})
        self.image_label.opacity = 0
        layout.add_widget(self.image_label)

        # Analiz butonu
        self.btn_analyze = Button(text="Analiz Et", font_size='16sp', size_hint=(None, None), size=(200, 50), pos_hint={'x': 0.575, 'top': 0.6})
        self.btn_analyze.bind(on_press=self.analyze_image)
        layout.add_widget(self.btn_analyze)

        # Sonuç etiketi
        self.result_label = Label(text="", font_size='20sp', color=(1, 0, 0, 1), size_hint=(None, None), size=(400, 50), pos_hint={'x': 0.525, 'top': 0.29}, halign='center')  # top: 0.3 olarak ayarlandı
        self.result_label.opacity = 0  # Başlangıçta görünmez
        layout.add_widget(self.result_label)

        # Açıklama etiketi
        self.description_label = Label(text="", font_size='16sp', color=(1, 1, 1, 1), size_hint=(None, None), size=(400, 100), pos_hint={'x': 0.5, 'top': 0.27}, halign='center')  # top: 0.1 olarak ayarlandı
        self.description_label.opacity = 0  # Başlangıçta görünmez
        self.description_label.text_size = self.description_label.size # Etiketin genişliğini text_size olarak ayarlıyoruz.
        layout.add_widget(self.description_label)

        return layout

    # Görsel seçme fonksiyonu
    def select_image(self, instance):
        file_chooser = FileChooserIconView()
        file_chooser.path = self.last_path if hasattr(self, 'last_path') else r"C:\Users\erdem\OneDrive\Masaüstü\Brain Tumor Project" #  Başlangıç lokasyonu
        file_chooser.bind(on_submit=self.on_file_selected)

        # Çıkış butonunu oluşturun
        exit_button = Button(text="Çıkış", size_hint=(None, None), size=(100, 40))
        exit_button.bind(on_press=lambda x: popup.dismiss()) # Çıkış butonuna basıldığında popup'ı kapat

        # Popup'ı oluşturun
        popup = Popup(title="Dosya Seç", content=file_chooser, size_hint=(0.9, 0.9))
        popup.content.add_widget(exit_button)  # Çıkış butonunu Popup'a ekleyin

        popup.open()

    def on_file_selected(self, instance, value, touch):
        image_path = value[0]
        self.selected_image_path = image_path
        img = Image.open(image_path).resize((200, 200))

        # PIL.Image.Image'i Kivy Texture'ına dönüştürün
        texture = Texture.create(size=img.size, colorfmt='rgb')
        texture.blit_buffer(img.tobytes(), bufferfmt='ubyte', colorfmt='rgb')

        # KivyImage nesnesini oluşturun ve texture'ı atayın
        self.image_label.texture = texture

        # Görsel seçildikten sonra sonuç ve açıklama etiketlerini görünür yap
        self.result_label.opacity = 1
        self.description_label.opacity = 1
        self.image_label.opacity = 1

        # Analiz et butonunu aşağı kaydır
        self.btn_analyze.pos_hint = {'x': 0.575, 'top': 0.4}  # Önceki konumu: 'top': 0.6

        # Görsel seçildikten sonra analiz et butonunu tekrar görünür yap (isteğe bağlı)
        self.btn_analyze.opacity = 1


    # Görseli analiz etme fonksiyonu
    def analyze_image(self, instance):
        if hasattr(self, 'selected_image_path'):
            class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
            prediction = predict_image(self.selected_image_path, model)
            predicted_class = class_names[prediction]
            self.result_label.text = f"Prediction: {predicted_class}"
            self.description_label.text = class_descriptions[predicted_class]
        else:
            self.result_label.text = "Lütfen bir MRI görseli seçin."
            self.description_label.text = ""

if __name__ == "__main__":
    TumorApp().run()
