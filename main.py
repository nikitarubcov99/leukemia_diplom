import sys
import cv2
import keras
import matplotlib
import numpy as np
import psycopg2
import tensorflow as tf
from models import *
from PIL.Image import Image
from IPython.display import display
from keras.applications import EfficientNetB3
from keras.applications.efficientnet import preprocess_input
from keras.preprocessing import image
from card import *


def get_contours(grad_cam):
    """
    Функция для получения контуров
    :param grad_cam:
    :return: Возвращает контур
    """
    # Применение порогового значения для получения бинарного изображения
    thresholded = cv2.threshold(grad_cam, 0, 255, cv2.THRESH_BINARY)[1]

    # Применение операций морфологического преобразования для улучшения контуров
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # Поиск контуров на полученном изображении
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Функция для сохранения в файл изображения тепловой карты
    :param img_path: путь к изображению
    :param heatmap: тепловая карта
    :param cam_path: путь к файлу изображения тепловой карты
    :param alpha: параметр
    :return: Возвращает контур
    """
    # Загружаем оригинальное изображение
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Скалируем тепловую карту в пределах 0-255
    heatmap = np.uint8(255 * heatmap)
    contours = get_contours(heatmap)
    # Испульзуем цветовую карту jet для раккраски тепловой карты
    jet = matplotlib.colormaps.get_cmap("jet")

    # Ихспоьзуем значение RGB для цветовой карты
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Создаем изображение с расскрашеной RGB тепловой картой
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Накладываем тепловую карту на оригинальное изображение
    superimposed_img = jet_heatmap * alpha + img

    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Сохраняем получившееся изображение
    superimposed_img.save(cam_path)
    # img = Image.open(cam_path)
    # # Отображаем получившееся изображение
    # display(img)
    return (contours)


def add_patient_to_db(family, name, second_name, age, snils, birth_date, illness_bool):
    """
    Функция для добавления(обновления) в БД информации о пациенте
    :param family: Фамилия пациента
    :param name: Имя пациента
    :param second_name: Отчество пациента
    :param age: Возраст пациента
    :param snils: СНИЛС пациента
    :param birth_date: Дата рождения пациента
    :param illness_bool: Хранит bool значение, если у пациента лейкемия True, иначе False
    :return: Ничего не возвращает
    """
    snils = snils.replace(' ', '')
    snils = snils.replace('-', '')
    patient = PatientModel.select().where(PatientModel.patient_snils == snils).get_or_none()
    if patient is None:
        PatientModel.create(patient_name=name, patient_second_name=second_name, patient_family=family,
                            patient_age=int(age), patient_birth_date=birth_date, patient_snils=snils,
                            patient_has_leukemia=illness_bool,
                            patient_analyses_count=1)
        return
    for patient in PatientModel.select().where(PatientModel.patient_snils == snils):
        patient_analyses_count = patient.patient_analyses_count
    PatientModel.update(patient_has_leukemia= illness_bool,patient_analyses_count = patient_analyses_count + 1).where(PatientModel.patient_snils == snils).execute()


def add_doctor_to_db(family, name, second_name):
    """
    Функция для добавления(обновления) информации о враче в БД
    :param family: Фамилия врача
    :param name: Имя врача
    :param second_name: Фамилия врача
    :return: ничего не возвращает
    """
    doctor = DoctorModel.select().where(DoctorModel.doctor_name == name and DoctorModel.doctor_family == family
                                        and DoctorModel.doctor_second_name == second_name).get_or_none()
    if doctor is None:
        DoctorModel.create(doctor_name=name, doctor_second_name=second_name, doctor_family=family)
        return


def add_card_to_db(snils, doc_name, doc_fam, doc_sec, start_file_name, cam_file_name):
    """
    Функция для создания(обновления) карточки пациента
    :param snils: СНИЛС пациента
    :param doc_name: Имя врача
    :param doc_fam: Фамилия врача
    :param doc_sec: Отчество врача
    :param start_file_name: Путь к исходному изображению
    :param cam_file_name: Путь к изображению с тепловой картой аномалий
    :return: Ничего не возвращает
    """
    snils = snils.replace(' ', '')
    snils = snils.replace('-', '')
    for patient in PatientModel.select().where(PatientModel.patient_snils == snils):
        patient_id = patient.patient_id
        patient_diagnose = patient.patient_has_leukemia
    for doctor in DoctorModel.select().where(
            DoctorModel.doctor_name == doc_name and DoctorModel.doctor_family == doc_fam
            and DoctorModel.doctor_second_name == doc_sec):
        doctor_id = doctor.doctor_id
    # Стартовое изображение
    fin = open(start_file_name, "rb")
    img = fin.read()
    binary = psycopg2.Binary(img)
    fin.close()
    # Изображение с тепловой картой
    fin = open(cam_file_name, "rb")
    img = fin.read()
    binary1 = psycopg2.Binary(img)
    fin.close()
    current_date = date.today().strftime("%d-%m-%y")
    if patient_diagnose == True:
        cur_diag = 'лейкемия'
        mkb_diag = 'C91.0'
    else:
        cur_diag = 'нет заболевания'
        mkb_diag = 'здоров'

    card = PatientsCardsModel.select().where(PatientsCardsModel.patient_card_patient_id == patient_id and
                                             PatientsCardsModel.patient_card_doctor_id == doctor_id).get_or_none()

    if card is None:
        PatientsCardsModel.create(patient_card_patient_id=patient_id, patient_card_doctor_id=doctor_id,
                                  card_creation_date=current_date, diagnose=cur_diag, mkb_diagnose=mkb_diag,
                                  start_image=binary, anomaly_image=binary1)
        return
    PatientsCardsModel.update(diagnose=cur_diag, mkb_diagnose=mkb_diag, start_image=binary, anomaly_image=binary1).where(
        PatientsCardsModel.patient_card_patient_id == patient_id and PatientsCardsModel.patient_card_doctor_id == doctor_id).execute()
    # query = card.update(diagnose=cur_diag, mkb_diagnose=mkb_diag, start_image=binary, anomaly_image=binary1)
    # query.execute()


class Window(QMainWindow, QTableWidget):
    """
    Класс инициализирующий главное окно программы и реализующий взаимодействие с ней
    """
    def __init__(self):
        """
        Конструктор класса
        """
        super().__init__()
        uic.loadUi('Forms/main_window.ui', self)
        self.show()
        self.pixmap = None
        self.showDialog = None
        self.file_name = None
        self.acceptDrops()
        self.progressBar.setValue(0)
        self.label.setGeometry(220, 40, 241, 221)
        self.pushButton.clicked.connect(self.get_file_path)
        self.pushButton_2.clicked.connect(self.load_model)
        self.pushButton_3.clicked.connect(self.create_report)
        self.pushButton_4.clicked.connect(self.openCard)

    def openCard(self):
        """
        Метод для инициализации окна программы для создания отчета
        :return: ничего не возвращает
        """
        self.ui = ui_card()
        self.ui.setupUi()

    def create_report(self):
        """
        Метод для создания отчет об анализе
        :return: ничего не возвращает
        """
        illness_text = self.label_2.text()
        start_file_name = self.label_13.text()
        if len(illness_text) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText('Для начала произведите анализ')
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        sub = 'лейкимия'
        if sub in illness_text:
            illness_bool = True
        else:
            illness_bool = False
        patient_family = self.lineEdit.text()
        patient_name = self.lineEdit_2.text()
        patient_second_name = self.lineEdit_3.text()
        patient_age = self.lineEdit_4.text()
        patient_snils = self.lineEdit_8.text()
        patient_snils.replace('-', '')
        patient_snils.replace(' ', '')
        patient_birth_date = self.dateEdit.dateTime().toString('dd.MM.yyyy')

        doctor_family = self.lineEdit_5.text()
        doctor_name = self.lineEdit_6.text()
        cam_file_name = 'cam.jpg'
        doctor_second_name = self.lineEdit_7.text()
        if len(patient_family) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText('Не заполнено поле фамилии')
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        if len(patient_name) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText('Не заполнено поле имени')
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        if len(patient_second_name) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText('Не заполнено поле отчества')
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        if len(patient_age) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText('Не заполнено поле возраста')
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        if len(patient_snils) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText('Не заполнено поле снилса')
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        if len(doctor_family) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText('Не заполнено поле возраста')
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        if len(doctor_name) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText('Не заполнено поле возраста')
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        if len(doctor_second_name) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText('Не заполнено поле возраста')
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        add_patient_to_db(patient_family, patient_name, patient_second_name, patient_age, patient_snils,
                          patient_birth_date, illness_bool)
        add_doctor_to_db(doctor_family, doctor_name, doctor_second_name)
        add_card_to_db(patient_snils, doctor_name, doctor_family, doctor_second_name, start_file_name, cam_file_name)

    def get_file_path(self):
        """
        Метод для получения пути к выбранному для анализа файлу.
        :return: ничего не возвращает
        """
        self.label_2.clear()
        self.label.clear()
        file_name = QFileDialog.getOpenFileName(self, 'Open file',
                                                '"C:/Users/nero1"')[0]
        # Вызов метода для вывода изображения\
        self.label_13.setText(file_name)
        self.print_image()
        # Вызов метода для классификации пневмонии на изображении
        # self.load_model(file_name)

    def print_image(self):
        """
        Метод для отображения выбранного пользователем изображения.
        :return: ничего не возвращает
        """
        file_name = self.label_13.text()
        self.pixmap = QPixmap(file_name)
        self.pixmap = self.pixmap.scaled(241, 221)

        # Добавление изображения в поле
        self.label.setPixmap(self.pixmap)

    def result_implementation(self, predict_classes, classes):
        """
        Метод для имплементации результатов работы нейронной сети
        :param predict_classes: хранит класс предсказанный нейронной сетью
        :param classes: список, хранит возможный набор классов [здоров, лейкемия]
        :return:
        """
        self.progressBar.setValue(6)
        index = np.argmax(predict_classes[0])
        klass = classes[index]
        probability = predict_classes[0][index] * 100
        if index == 0:
            self.label_2.setText(f'С вероятностью {probability:6.2f} % пациент {klass}')

        else:
            self.label_2.setText(f'С вероятностью {probability:6.2f} % на изображении {klass}')
        self.progressBar.setValue(7)

    def load_model(self):
        """
        Метод для загрузки и использования модели, модель загружается в память только при первом вызове метода.
        :return:
        """
        file_name = self.label_13.text()
        global saved_model, normal_data, img
        classes = ['здоров', 'лейкимия']
        model_k = 0
        img_size = 224
        self.progressBar.setValue(1)

        # Проверка загружена ли модель в память
        if model_k == 0:
            saved_model = tf.keras.models.load_model("model/model.h5")
            model_k += 1
        self.progressBar.setValue(2)

        # Пред обработка выбранного пользователем изображения для классификации, если файла не существует по пути,
        # вызывается исключение
        try:
            self.progressBar.setValue(3)
            img = cv2.imread(file_name)
            img = cv2.resize(img, (img_size, img_size))
            img = np.expand_dims(img, axis=0)
        except Exception as e:
            print(e)
        self.progressBar.setValue(4)
        # Использование модели для классификации выбранного пользователем изображения
        pred = saved_model.predict(img)
        self.progressBar.setValue(5)

        model = EfficientNetB3(weights='imagenet')
        # Получение последнего сверточного слоя внутри efficientnetb3
        last_conv_layer = model.get_layer('top_conv')
        # Создание модели с усеченной последовательностью слоев
        grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
        # Пример загрузки и пред обработки изображения
        img = image.load_img(file_name, target_size=(
            300, 300))  # Пример, входное изображение должно иметь размерность, соответствующую EfficientNetB3
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        # Вычисление Grad-CAM
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_idx = np.argmax(preds[0])
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        # Умножение каждого канала на соответствующий весовой коэффициент
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # Создание тепловой карты Grad-CAM
        heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        # Отображение тепловой карты
        save_and_display_gradcam(file_name, heatmap)

        # Вызов метода для интерпретации результатов работы модели
        self.result_implementation(pred, classes)

    @staticmethod
    def exit_app():
        """
        Метод для закрытия приложения.
        :return:
        """
        QApplication.quit()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    App.setStyleSheet("QLabel{font-size: 14pt;}")
    window = Window()
    sys.exit(App.exec())
