import os
import sys
import typing
from datetime import date, datetime
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtGui import QPixmap
from PIL import Image
import io
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import PIL
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QFileDialog, QLabel, \
    QPushButton, QProgressBar, QLineEdit, QTextEdit, QMessageBox, QDialog, QWidget
from models.models import PatientModel, PatientsCardsModel, DoctorModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
import fitz
import PyPDF2


def add_image_to_existing_pdf(pdf_path, image_path, image_path1, page_number):
    """
    Функция для добавления изображений в pdf файл
    :param pdf_path: путь к файлу в котором будет храниться отчет
    :param image_path: путь к исходному изображению
    :param image_path1: путь к изображению с тепловой картой аномалий
    :param page_number: номер страницы pdf файла на которую нужно вставить изображение
    :return:
    """
    # Открываем существующий PDF-файл
    pdf_document = fitz.open(pdf_path)

    # Открываем изображение
    pdf_document[page_number].insert_image(fitz.Rect(30, 255, 230, 455), stream=open(image_path, "rb").read())
    pdf_document[page_number].insert_image(fitz.Rect(330, 255, 530, 455), stream=open(image_path1, "rb").read())

    pdf_document.saveIncr()

    # Закрываем документ
    pdf_document.close()


class ui_card(QDialog):
    """
    Класс реализующий инициализацию компонентов и дальнейшее взаимодействие с окном программы "создание отчета"
    """
    def __init__(self):
        """
        Консструктор класса
        """
        super().__init__()
        # Пациент
        self.current_date_str = None
        self.output_date_str = None
        self.current_date = None
        self.patient_id = None
        self.patient_name = None
        self.patient_second_name = None
        self.patient_family = None
        self.patient_count = None
        self.patient_age = None
        self.patient_snils = None
        # Врач
        self.doctor_id = None
        self.doctor_family = None
        self.doctor_second_name = None
        self.doctor_name = None
        self.doctor_class = None
        # Карточка
        self.card_creation_date = None
        self.start_image = None
        self.anomaly_image = None
        self.mkb_diagnose = None
        self.diagnose = None

    def setupUi(self):
        """
        Метод для загрузки компонент интерфейса из ui файла
        :return: ничего не возвращает
        """
        uic.loadUi('Forms/Card.ui', self)
        self.show()
        self.pushButton.clicked.connect(self.generate_pdf)
        self.pushButton_2.clicked.connect(self.generate_report)
        self.pushButton_3.clicked.connect(self.close)

    def generate_report(self):
        """
        Метод для вывода на экран отчета
        :return: ничего не возвращает
        """
        global patient_id, doctor_id, card_creation_date, diagnose, anomaly_image, start_image, doctor_family, \
            doctor_second_name, doctor_name, patient_family, patient_second_name, patient_name, patient_count, \
            mkb_diagnose
        self.patient_snils = self.lineEdit.text()
        self.patient_snils = self.patient_snils.replace(' ', '')
        self.patient_snils = self.patient_snils.replace('-', '')
        if len(self.patient_snils) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText('Не заполнено поле СНИЛС')
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        for patient in PatientModel.select().where(PatientModel.patient_snils == self.patient_snils):
            self.patient_id = patient.patient_id
            self.patient_name = patient.patient_name
            self.patient_second_name = patient.patient_second_name
            self.patient_family = patient.patient_family
            self.patient_age = patient.patient_age
            self.patient_count = patient.patient_analyses_count
        for card in PatientsCardsModel.select().where(PatientsCardsModel.patient_card_patient_id == self.patient_id):
            self.doctor_id = card.patient_card_doctor_id
            self.diagnose = card.diagnose
            self.mkb_diagnose = card.mkb_diagnose
            self.card_creation_date = card.card_creation_date
            self.start_image = card.start_image
            self.anomaly_image = card.anomaly_image
        for doctor in DoctorModel.select().where(DoctorModel.doctor_id == self.doctor_id):
            self.doctor_name = doctor.doctor_name
            self.doctor_second_name = doctor.doctor_second_name
            self.doctor_family = doctor.doctor_family
            self.doctor_class = doctor.doctor_class
        self.label.setText(self.patient_name)
        self.label_2.setText(self.patient_second_name)
        self.label_3.setText(self.patient_family)
        self.label_5.setText(self.doctor_name)
        self.label_6.setText(self.doctor_second_name)
        self.label_4.setText(self.doctor_family)

        with open("image_from_db.jpg", "wb") as file:
            file.write(self.start_image)
            file.close()
        file_name = "C:/Users/Maxim/PycharmProjects/leukemia_diplom/image_from_db.jpg"
        self.pixmap = QPixmap(file_name)
        self.pixmap = self.pixmap.scaled(241, 221)
        self.label_10.setPixmap(self.pixmap)
        os.remove(file_name)

        with open("image_from_db1.jpg", "wb") as file:
            file.write(self.anomaly_image)
            file.close()
        file_name = "C:/Users/Maxim/PycharmProjects/leukemia_diplom/image_from_db1.jpg"
        self.pixmap = QPixmap(file_name)
        self.pixmap = self.pixmap.scaled(241, 221)
        self.label_11.setPixmap(self.pixmap)
        os.remove(file_name)

        if self.diagnose == 'лейкемия':
            self.label_9.setText(f'Обнаружена лейкемия, код по МКБ-10 {self.mkb_diagnose}')
        else:
            self.label_9.setText('Заболеваний не обнаружено')
        self.current_date = date.today()
        if self.current_date.year < 2000:
            self.current_date = self.current_date.replace(year=self.current_date.year + 2000)
        input_date = datetime.strptime(str(self.card_creation_date), '%Y-%m-%d')
        self.output_date_str = input_date.strftime('%d.%m.%Y')
        self.label_20.setText(str(self.output_date_str))
        input_date = datetime.strptime(str(self.current_date), '%Y-%m-%d')
        self.current_date_str = input_date.strftime('%d.%m.%Y')
        self.label_21.setText(str(self.current_date_str))
        self.label_22.setText(str(self.patient_count))

    def generate_pdf(self):
        """
        Метод для создания pdf файла с отчетом
        :return: ничего не возвращает
        """
        # инициализируем документ и шрифт
        pdf = FPDF()
        pdf.add_page()
        font_path = 'fonts/timesnrcyrmt.ttf'
        pdf.add_font("Times", "", font_path, uni=True)
        pdf.set_font("Times", size=14)
        # описываем все строки которые будут добавлены в pdf отчет
        intro = '            Отчет об анализе на наличие лейкемии по пятну крови'.encode('utf-8')
        fio_intro = 'Данные пациента: '.encode('utf-8')
        patient_fio = f'Фамилия: {self.label.text()}     Имя: {self.label_2.text()}     Отчество: {self.label_3.text()}     Полных лет: {self.patient_age}'.encode(
            'utf-8')
        patient = f'СНИЛС: {self.patient_snils}     диагноз по МКБ-10: {self.mkb_diagnose}     кол-во анализов {self.patient_count}'.encode(
            'utf-8')
        doctor_intro = 'Информация о лечащем враче: '.encode('utf-8')
        doctor_fio = f'Фамилия: {self.doctor_name}     Имя: {self.doctor_second_name}     Отчество: {self.doctor_family}'.encode(
            'utf-8')
        doctor_cat = f'Категория лечащего врача: {self.doctor_class}'.encode('utf-8')
        image_intro_norm = 'Изначальное изображение                          Изображение с аномалиями'.encode('utf-8')
        if self.diagnose == 'лейкемия':
            analys_result = '                            При анализе обнаружена лейкемия'.encode('utf-8')
        else:
            analys_result = '                              Заболеваний не обнаружено'.encode('utf-8')
        open_date = f'Дата создания карточки {self.output_date_str}                       Дата осмотра {self.current_date_str}'.encode(
            'utf-8')
        # добавляем полученные строки в pdf файл отчета
        pdf.set_font("Times", size=18)
        pdf.multi_cell(200, 10, str(intro.decode('utf-8')), align='С')
        pdf.set_font("Times", size=16)
        pdf.multi_cell(200, 10, str(fio_intro.decode('utf-8')))
        pdf.set_font("Times", size=14)
        pdf.multi_cell(400, 10, str(patient_fio.decode('utf-8')))
        pdf.multi_cell(400, 10, str(patient.decode('utf-8')))
        pdf.set_font("Times", size=16)
        pdf.multi_cell(400, 10, str(doctor_intro.decode('utf-8')))
        pdf.set_font("Times", size=14)
        pdf.multi_cell(400, 10, str(doctor_fio.decode('utf-8')))
        pdf.multi_cell(400, 10, str(doctor_cat.decode('utf-8')))
        pdf.set_font("Times", size=16)
        pdf.multi_cell(400, 10, str(image_intro_norm.decode('utf-8')))
        pdf.set_font("Times", size=18)
        pdf.ln(75)
        pdf.multi_cell(400, 10, str(analys_result.decode('utf-8')))
        pdf.set_font("Times", size=14)
        pdf.multi_cell(400, 10, str(open_date.decode('utf-8')))
        pdf_path = f"reports/{self.patient_name} {self.patient_second_name[0]}. {self.patient_family[0]}.  {self.current_date_str}.pdf"
        pdf.output(pdf_path)
        # добавляем исходное изображение и изображение с тепловой картой аномалий в pdf файл отчета
        with open("image_from_db.jpeg", "wb") as file:
            file.write(self.start_image)
            file.close()
        with open("image_from_db1.jpeg", "wb") as file:
            file.write(self.anomaly_image)
            file.close()
        image_path = "C:/Users/Maxim/PycharmProjects/leukemia_diplom/image_from_db.jpeg"
        image_path1 = "C:/Users/Maxim/PycharmProjects/leukemia_diplom/image_from_db1.jpeg"
        existing_pdf_path = pdf_path
        target_page_number = 0
        add_image_to_existing_pdf(existing_pdf_path, image_path, image_path1, target_page_number)
        os.remove(image_path1)
        os.remove(image_path)
