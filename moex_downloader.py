"""
Copyright (C) 2026 LEX
Apache 2.0 License  http://www.apache.org/licenses/
"""


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QLineEdit, QLabel, QPushButton, QHBoxLayout, QProgressBar, QMessageBox, QComboBox, QStyledItemDelegate
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QBrush, QPalette

import pandas as pd
import asyncio
import httpx
import qasync
import sys
import os
import json

from datetime import datetime
from dateutil.relativedelta import relativedelta
import re

from httpx_moex import candles
from httpx_moex import history
from httpx_moex import reference

T_STOCKS:str = "stocks"
T_FUTURES:str = "futures"
T_STOCKS_COLS = ["Load", "SECID", "SHORTNAME", "CURRENCYID", "ISIN", "LISTLEVEL", "STATUS", "Start","End"]
T_FUTURES_COLS = ["Load", "SECID", "name", "start_date", "expiration_date", "asset_code", "underlying_asset", "is_traded",  "Start","End"]
T_STOCKS_FILE = T_STOCKS+"_config.csv"
T_FUTURES_FILE = T_FUTURES+"_config.csv"
T_SETTINGS_FILE = "moex_downloader_settings.json"
T_DATA_PATH = os.path.join(".", "DATA")
T_DARK_THEME = False
T_DEFAULT_PALETTE = None
T_DEFAULT_STYLE_NAME = None

DARK_STYLE_SHEET = """
QWidget {
    background-color: #262626;
    color: #ebebeb;
}
QMainWindow,
QDialog,
QMessageBox,
QStatusBar {
    background-color: #262626;
    color: #ebebeb;
}
QLineEdit,
QComboBox,
QTableWidget,
QTableView,
QHeaderView,
QProgressBar {
    background-color: #1e1e1e;
    color: #ebebeb;
    border: 1px solid #555555;
}
QLineEdit:focus,
QComboBox:focus,
QTableWidget:focus {
    border: 1px solid #6e9ed6;
}
QPushButton {
    background-color: #323232;
    color: #ebebeb;
    border: 1px solid #666666;
    padding: 4px 10px;
}
QPushButton:hover {
    background-color: #3d3d3d;
}
QPushButton:pressed {
    background-color: #4a4a4a;
}
QPushButton:disabled {
    color: #919191;
    background-color: #2d2d2d;
}
QComboBox QAbstractItemView {
    background-color: #1e1e1e;
    color: #ebebeb;
    selection-background-color: #3c6eaa;
}
QTabWidget::pane {
    border: 1px solid #555555;
}
QTabBar::tab {
    background-color: #303030;
    color: #ebebeb;
    border: 1px solid #555555;
    padding: 5px 12px;
}
QTabBar::tab:selected {
    background-color: #404040;
}
QTableWidget {
    gridline-color: #555555;
    selection-background-color: #3c6eaa;
    selection-color: #ffffff;
}
QTableView::indicator,
QTableWidget::indicator {
    width: 13px;
    height: 13px;
    background-color: #f5f5f5;
    border: 1px solid #d8d8d8;
}
QTableView::indicator:unchecked,
QTableWidget::indicator:unchecked {
    background-color: #f5f5f5;
    border: 1px solid #d8d8d8;
}
QTableView::indicator:checked,
QTableWidget::indicator:checked {
    image: url("__CHECKBOX_CHECKED_DARK__");
    background-color: transparent;
    border: none;
}
QAbstractScrollArea {
    background-color: #1e1e1e;
}
QHeaderView::section {
    background-color: #2f3f4f;
    color: #f0f0f0;
    border: 1px solid #555555;
}
QProgressBar::chunk {
    background-color: #3c6eaa;
}
QScrollBar:vertical,
QScrollBar:horizontal {
    background-color: #262626;
    border: 1px solid #444444;
}
QScrollBar::handle:vertical,
QScrollBar::handle:horizontal {
    background-color: #555555;
    min-height: 20px;
    min-width: 20px;
}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {
    background-color: #666666;
}
QScrollBar::add-line,
QScrollBar::sub-line {
    background: none;
    border: none;
}
QScrollBar::add-page,
QScrollBar::sub-page {
    background: none;
}
QToolTip {
    color: #f0f0f0;
    background-color: #333333;
    border: 1px solid #666666;
}
"""

LIGHT_STYLE_SHEET = """
QTableView::indicator,
QTableWidget::indicator {
    width: 13px;
    height: 13px;
    background-color: #ffffff;
    border: 1px solid #555555;
}
QTableView::indicator:unchecked,
QTableWidget::indicator:unchecked {
    background-color: #ffffff;
    border: 1px solid #555555;
}
QTableView::indicator:checked,
QTableWidget::indicator:checked {
    image: url("__CHECKBOX_CHECKED_LIGHT__");
    background-color: transparent;
    border: none;
}
"""


def get_style_sheet(template, checkbox_file, placeholder):
    checkbox_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), checkbox_file)
    checkbox_path = checkbox_path.replace("\\", "/")
    return template.replace(placeholder, checkbox_path)


def get_dark_style_sheet():
    return get_style_sheet(DARK_STYLE_SHEET, "checkbox_checked_dark.xpm", "__CHECKBOX_CHECKED_DARK__")


def get_light_style_sheet():
    return get_style_sheet(LIGHT_STYLE_SHEET, "checkbox_checked_light.xpm", "__CHECKBOX_CHECKED_LIGHT__")


def normalize_date(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or "_" in text:
        return None
    datetime.strptime(text, '%Y-%m-%d')
    return text


def fit_button_width(button, min_width=0):
    button.setMinimumWidth(max(min_width, button.sizeHint().width() + 12))


def get_settings_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, T_SETTINGS_FILE)


def load_saved_theme():
    settings_path = get_settings_path()
    if not os.path.isfile(settings_path):
        return None

    try:
        with open(settings_path, "r", encoding="utf-8") as file:
            settings = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    theme = settings.get("theme")
    if theme in ("dark", "light"):
        return theme
    return None


def save_theme_setting(dark):
    settings = {"theme": "dark" if dark else "light"}
    settings_path = get_settings_path()
    try:
        with open(settings_path, "w", encoding="utf-8") as file:
            json.dump(settings, file, ensure_ascii=False, indent=2)
    except OSError as ex:
        print(f"Failed to save settings: {ex}")


def color_luminance(color):
    return 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()


def is_dark_system_theme(app):
    palette = app.palette()
    window = palette.color(QPalette.ColorRole.Window)
    text = palette.color(QPalette.ColorRole.WindowText)
    return color_luminance(window) < color_luminance(text)


def set_app_theme(app, dark):
    global T_DARK_THEME
    T_DARK_THEME = dark

    if not dark:
        if T_DEFAULT_STYLE_NAME:
            app.setStyle(T_DEFAULT_STYLE_NAME)
        palette = QPalette()
        light_window = QColor(240, 240, 240)
        light_base = QColor(255, 255, 255)
        light_button = QColor(240, 240, 240)
        dark_text = QColor(0, 0, 0)
        disabled_text = QColor(120, 120, 120)

        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            palette.setColor(group, QPalette.ColorRole.Window, light_window)
            palette.setColor(group, QPalette.ColorRole.Base, light_base)
            palette.setColor(group, QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
            palette.setColor(group, QPalette.ColorRole.Button, light_button)
            palette.setColor(group, QPalette.ColorRole.WindowText, dark_text)
            palette.setColor(group, QPalette.ColorRole.Text, dark_text)
            palette.setColor(group, QPalette.ColorRole.ButtonText, dark_text)
            palette.setColor(group, QPalette.ColorRole.Highlight, QColor(49, 106, 197))
            palette.setColor(group, QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Window, light_window)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, light_base)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, light_button)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
        app.setPalette(palette)
        app.setStyleSheet(get_light_style_sheet())
        return

    app.setStyle("Fusion")
    palette = app.palette()
    dark_window = QColor(38, 38, 38)
    dark_base = QColor(30, 30, 30)
    dark_button = QColor(50, 50, 50)
    light_text = QColor(235, 235, 235)
    disabled_text = QColor(145, 145, 145)

    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        palette.setColor(group, QPalette.ColorRole.Window, dark_window)
        palette.setColor(group, QPalette.ColorRole.Base, dark_base)
        palette.setColor(group, QPalette.ColorRole.AlternateBase, QColor(45, 45, 45))
        palette.setColor(group, QPalette.ColorRole.Button, dark_button)
        palette.setColor(group, QPalette.ColorRole.WindowText, light_text)
        palette.setColor(group, QPalette.ColorRole.Text, light_text)
        palette.setColor(group, QPalette.ColorRole.ButtonText, light_text)
        palette.setColor(group, QPalette.ColorRole.Highlight, QColor(60, 110, 170))
        palette.setColor(group, QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Window, dark_window)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, dark_base)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, dark_button)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
    app.setPalette(palette)
    app.setStyleSheet(get_dark_style_sheet())


def apply_system_theme(app):
    set_app_theme(app, is_dark_system_theme(app))


class MaskedDelegate(QStyledItemDelegate):
    def __init__(self, mask="9999-99-99;_", parent=None):
        super().__init__(parent)
        self.mask = mask

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setInputMask(self.mask)
        return editor

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        editor.setText("" if value in (None, "") else value)

    def setModelData(self, editor, model, index):
        text = editor.text()
        # Если остались только подчеркивания и дефисы — считаем пустым
        if all(c in "_-" for c in text) or text.strip() == "":
            model.setData(index, "", Qt.ItemDataRole.EditRole)
        else:
            model.setData(index, text, Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class MainWindow(QMainWindow):
    def __init__(self, df1, df2):
        super().__init__()
        self.setWindowTitle("MOEX Downloader")

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.df_stocks = df1
        self.df_futures = df2
        self.loading = False

        self.tab1, self.table_stocks, self.tf_stocks= self.create_tab(self.df_stocks, T_STOCKS)
        self.tab2, self.table_futures, self.tf_futures = self.create_tab(self.df_futures, T_FUTURES)

        self.tabs.addTab(self.tab1, "Акции")
        self.tabs.addTab(self.tab2, "Фьючерсы")

        theme_layout = QHBoxLayout()
        self.theme_button = QPushButton()
        self.theme_button.clicked.connect(self.toggle_theme)
        theme_layout.addStretch()
        theme_layout.addWidget(self.theme_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(theme_layout)
        main_layout.addWidget(self.tabs)
        self.update_theme_button()

        self.progress_w = QWidget()
        progress_box = QHBoxLayout()
        self.progress_label = QLabel("Loading...")
        progress_box.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        progress_box.addWidget(self.progress_bar)
        self.progress_w.setLayout(progress_box)
        self.statusBar().addWidget(self.progress_w)
        self.progress_w.setVisible(False)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Сохраняем ссылки на tasks чтобы избежать преждевременного удаления
        self._refresh_task = None
        self._load_task = None
        self._download_task = None
        QTimer.singleShot(0, self._start_refresh_all)


    def update_theme_button(self):
        self.theme_button.setText("Светлая тема" if T_DARK_THEME else "Темная тема")
        fit_button_width(self.theme_button, 120)


    def toggle_theme(self):
        set_app_theme(QApplication.instance(), not T_DARK_THEME)
        save_theme_setting(T_DARK_THEME)
        self.update_theme_button()
        self.apply_table_theme(self.table_stocks)
        self.apply_table_theme(self.table_futures)


    def apply_table_theme(self, table):
        if T_DARK_THEME:
            table.horizontalHeader().setStyleSheet("QHeaderView::section { background-color: #2f3f4f; color: #f0f0f0; }")
            selected_background = QColor("#1f6f3a")
            selected_text = QBrush(QColor("#ffffff"))
        else:
            table.horizontalHeader().setStyleSheet("QHeaderView::section { background-color: lightblue; }")
            selected_background = QColor("lightgreen")
            selected_text = QBrush(QColor("#000000"))

        table.viewport().update()

        for row in range(table.rowCount()):
            checkbox_item = table.item(row, 0)
            checked = checkbox_item is not None and checkbox_item.checkState() == Qt.CheckState.Checked
            for col in range(table.columnCount()):
                item = table.item(row, col)
                if item is None:
                    continue
                if checked:
                    item.setBackground(selected_background)
                    item.setForeground(selected_text)
                else:
                    item.setBackground(QBrush())
                    item.setForeground(QBrush())
        table.viewport().update()


    def _start_refresh_all(self):
        """Wrapper для запуска async задачи из синхронного контекста"""
        self._refresh_task = asyncio.create_task(self.refresh_all())


    def merge_new_df(self, df, tab_id):
        if tab_id == T_STOCKS:
            df = self.df_stocks = self.merge_df(df, self.df_stocks, T_STOCKS_COLS)
        else:
            df = self.df_futures = self.merge_df(df, self.df_futures, T_FUTURES_COLS)
            df = update_futures_date(df)
        return df


    def merge_df(self, df, existing_df, cols):
        if len(existing_df) > 0:
            df_new = pd.merge(df, existing_df[["Load", "Start", "End"]], left_index=True, right_index=True, how='left')
            df = df_new.fillna("")
        else:
            df["Load"] = ""
            df["Start"] = ""
            df["End"] = ""
        return df[cols]


    def table_load(self, table, df, tab_id, update=False):
        try:
            table.itemChanged.disconnect()
        except TypeError:
            pass

        if update:
            df = self.merge_new_df(df, tab_id)

        table.setRowCount(0)
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns.tolist())
        if T_DARK_THEME:
            table.horizontalHeader().setStyleSheet("QHeaderView::section { background-color: #2f3f4f; color: #f0f0f0; }")
            selected_background = QColor("#1f6f3a")
            selected_text = QBrush(QColor("#ffffff"))
        else:
            table.horizontalHeader().setStyleSheet("QHeaderView::section { background-color: lightblue; }")
            selected_background = QColor("lightgreen")
            selected_text = QBrush(QColor("#000000"))

        def handle_checkbox_change(row):
            checkbox_item = table.item(row, 0)
            for col_index in range(table.columnCount()):
                item = table.item(row, col_index)
                if checkbox_item.checkState() == Qt.CheckState.Checked:
                    if item:
                        item.setBackground(selected_background)
                        item.setForeground(selected_text)
                else:
                    if item:
                        item.setBackground(QBrush())
                        item.setForeground(QBrush())

        ncols = df.shape[1]
        for row_index in range(df.shape[0]):
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            val = df.iat[row_index, 0]
            checkbox_item.setCheckState(Qt.CheckState.Checked if val==1 else Qt.CheckState.Unchecked)
            table.setItem(row_index, 0, checkbox_item)

            # for col_index, value in enumerate(row):
            for col in range(1, df.shape[1]):
                val = df.iat[row_index, col]
                item = QTableWidgetItem(str(val))
                # enable edit only for Start/End
                if col<ncols-2:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row_index, col , item)

            handle_checkbox_change(row_index)

        delegate = MaskedDelegate("9999-99-99;_")
        table.setItemDelegateForColumn(ncols-1, delegate)
        table.setItemDelegateForColumn(ncols-2, delegate)
        table._date_delegate = delegate

        table.setColumnWidth(0, 28)
        if tab_id==T_STOCKS:
            table.setColumnWidth(3, 20)
            table.setColumnWidth(5, 40)
            table.setColumnWidth(6, 40)
        else:
            table.setColumnWidth(7, 20)

        def on_item_changed(item):
            if item.column() == 0:  # Only react to checkbox column changes
                handle_checkbox_change(item.row())

        table.itemChanged.connect(on_item_changed)


    def create_tab(self, dataframe, tab_id):
        tab = QWidget()
        layout = QVBoxLayout()

        filter_layout = QHBoxLayout()
        filter_label = QLabel("Find")
        filter_box = QLineEdit()
        filter_box.setPlaceholderText("Type to filter...  Use '/' to show only selected items")
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(filter_box)

        table = QTableWidget()
        self.table_load(table, dataframe, tab_id)

        def filter_table():
            filter_text = filter_box.text().lower()
            for row_index in range(table.rowCount()):
                match = False
                if filter_text=='/':
                    item = table.item(row_index, 0)
                    if item and item.checkState()==Qt.CheckState.Checked:
                        match = True
                else:
                    for col_index in range(1, table.columnCount()):
                        item = table.item(row_index, col_index)
                        if item and filter_text in item.text().lower():
                            match = True
                            break
                table.setRowHidden(row_index, not match)

        filter_box.textChanged.connect(filter_table)

        layout_combo = QHBoxLayout()
        combo_label = QLabel('Тimeframe:', self)
        combo_label.setFixedWidth(combo_label.sizeHint().width())
        combo = QComboBox()
        combo.addItem('1 мин', 'M1')
        combo.addItem('5 мин', 'M5')
        combo.addItem('10 мин', 'M10')
        combo.addItem('15 мин', 'M15')
        combo.addItem('30 мин', 'M30')
        combo.addItem('1 час', 'M60')
        combo.addItem('1 день', 'D')
        combo.addItem('1 месяц', 'M')
        combo.setCurrentText('1 день')
        combo.setFixedWidth(100)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_label = QLabel('Path:', self)
        path_label.setFixedWidth(path_label.sizeHint().width())
        path_edit = QLineEdit()
        path_edit.setText(script_dir)
        path_edit.setToolTip("Folder for saving quotes and config files")

        layout_combo.addWidget(combo_label)
        layout_combo.addWidget(combo)
        layout_combo.addSpacing(20)
        layout_combo.addWidget(path_label)
        layout_combo.addWidget(path_edit)

        button_layout = QHBoxLayout()
        refresh_button = QPushButton("Обновить список")
        fit_button_width(refresh_button, 120)
        refresh_button.setToolTip("Синхронизировать список тикеров с данными MOEX ")
        save_button = QPushButton("Сохранить")
        fit_button_width(save_button, 80)
        save_button.setToolTip("Сохранить конфигурацию таблиц в CSV файлах")
        run_button = QPushButton("Загрузить")
        fit_button_width(run_button, 80)
        run_button.setToolTip("Загрузить/обновить данные свечей для выбранных инструментов")

        button_layout.addStretch()
        button_layout.addWidget(refresh_button)
        button_layout.addSpacing(40)
        button_layout.addWidget(save_button)
        button_layout.addSpacing(40)
        button_layout.addWidget(run_button)
        button_layout.addStretch()

        save_button.clicked.connect(lambda: self.save_table_to_csv(table, tab_id, path_edit.text()))
        refresh_button.clicked.connect(lambda: self.refresh_data(table, tab_id))
        run_button.clicked.connect(lambda: self.run_downloading_candles(table, tab_id, combo.currentData(), path_edit.text()))

        layout.addLayout(filter_layout)
        layout.addWidget(table)
        layout.addLayout(layout_combo)
        layout.addLayout(button_layout)
        tab.setLayout(layout)
        return tab, table, combo


    def save_table_to_df(self, table):
        rows = []
        for row_index in range(table.rowCount()):
            row_data = []
            state = table.item(row_index, 0).checkState()
            row_data.append(1 if state==Qt.CheckState.Checked else 0)
            for col_index in range(1, table.columnCount()):
                item = table.item(row_index, col_index)
                row_data.append(item.text() if item else "")
            rows.append(row_data)
        if len(rows)>0:
            df = pd.DataFrame(rows, columns=[table.horizontalHeaderItem(i).text() for i in range(0, table.columnCount())])
            df["idx"] = df["SECID"]
            df.set_index("idx", inplace=True)
            return df
        else:
            return pd.DataFrame()

    def save_table_to_df_selected(self, table):
        rows = []
        for row_index in range(table.rowCount()):
            row_data = []
            state = table.item(row_index, 0).checkState()
            if state==Qt.CheckState.Checked:
                row_data.append(1)
                for col_index in range(1, table.columnCount()):
                    item = table.item(row_index, col_index)
                    row_data.append(item.text() if item else "")
                rows.append(row_data)
        if len(rows)>0:
            df = pd.DataFrame(rows, columns=[table.horizontalHeaderItem(i).text() for i in range(0, table.columnCount())])
            df["idx"] = df["SECID"]
            df.set_index("idx", inplace=True)
            return df
        else:
            return pd.DataFrame()

    def save_table_to_csv(self, table, tab_name, save_path=None):
        df = self.save_table_to_df(table)
        if not save_path:
            save_path = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, T_STOCKS_FILE if tab_name==T_STOCKS else T_FUTURES_FILE)
        df.to_csv(fname, index=False, lineterminator='\r\n', sep=';')
        QMessageBox.information(self, "Save", f"Data saved to {fname} ")
        # with open(f"{tab_name}_filtered.json", "w", encoding="utf-8") as file:
        #     df =pd.DataFrame(rows, columns=[table.horizontalHeaderItem(i).text() for i in range(1, table.columnCount())])
        #     df.to_json(file, orient="records", force_ascii=False, indent=4)


    def refresh_data(self, table, tab_id=T_STOCKS):
        df = self.save_table_to_df(table)
        if tab_id==T_STOCKS:
            self.df_stocks = df
        else:
            self.df_futures = df
        if self.loading:
            QMessageBox.information(self, "Already loading", "Please wait until the current operation finishes.")
            return
        # Сохраняем ссылку на task
        self._load_task = asyncio.create_task(self.load_info(table, tab_id))

    async def refresh_all(self):
        if len(self.df_stocks)==0:
            await self.load_info(self.table_stocks, T_STOCKS)
        if len(self.df_futures)==0:
            await self.load_info(self.table_futures, T_FUTURES)

    async def load_info(self, table, mode=T_STOCKS):
        self.loading = True
        self.progress_w.setVisible(True)
        self.progress_bar.setRange(0, 0)
        failed = False
        df = None
        try:
            async with httpx.AsyncClient() as client:
                if mode==T_STOCKS:
                    df = await load_list_stocks(client)
                else:
                    df = await load_list_futures(client)
        except Exception as ex:
            failed = True
            QMessageBox.warning(self, "Refresh Failed", str(ex))
        finally:
            self.loading = False
            self.progress_w.setVisible(False)
        if not failed:
            if df is not None and not df.empty:
                self.table_load(table, df=df, tab_id=mode, update=True)


    def run_downloading_candles(self, table, tab_id, tf, save_path=None):
        df = self.save_table_to_df_selected(table)
        if len(df)==0:
            return
        if self.loading:
            QMessageBox.information(self, "Already loading", "Please wait until the current operation finishes.")
            return
        # Сохраняем ссылку на task
        self._download_task = asyncio.create_task(self.download_candles(mode=tab_id, df=df, tf=tf, save_path=save_path))

    async def download_candles(self, mode=T_STOCKS, df=None, tf:str='D', save_path=None):
        if df is None or len(df)<1:
            return

        self.progress_w.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.loading = True
        failed = False
        tf = tf.upper()
        try:
            base_path = save_path if save_path else os.path.dirname(os.path.abspath(__file__))
            fpath = os.path.join(base_path, "DATA", "stocks" if mode==T_STOCKS else "futures", tf)
            market = "shares" if mode==T_STOCKS else "forts"
            engine = "stock" if mode==T_STOCKS else "futures"

            async with httpx.AsyncClient() as client:
                for row in range(df.shape[0]):
                    secid = df.iloc[row, df.columns.get_loc('SECID')]
                    self.progress_label.setText(f'Loading... {secid}')
                    
                    start = normalize_date(df.iloc[row, df.columns.get_loc('Start')])
                    end = normalize_date(df.iloc[row, df.columns.get_loc('End')]) if mode==T_FUTURES else None
                    if mode == T_FUTURES:
                        raw_name = df.iloc[row, df.columns.get_loc('name')]
                        raw_name = str(raw_name).strip() if raw_name else ""
                        fname = re.sub(r'[\\/:*?"<>|]', '_', raw_name) if raw_name else secid
                    else:
                        fname = secid
                    # await load_ticker_to_csv(client, secid, fpath, tf, start=None, end=None, market=market, engine=engine)
                    await load_ticker_to_csv(client, secid=secid, path=fpath, fname=fname,  tf=tf, start=start, end=end, market=market, engine=engine)

        except Exception as ex:
            print(ex)
            failed = True
            QMessageBox.warning(self, "Download Failed", str(ex))
        finally:
            self.loading = False
            self.progress_w.setVisible(False)
        if not failed:
            QMessageBox.information(self, "Download Complete", "Data loaded successfully!")



async def load_ticker_to_df(http_client:httpx.AsyncClient,
                            secid:str,
                            tf:str='D',
                            start:str|None=None,
                            end:str|None=None,
                            market='shares',
                            engine='stock'
    ):
    interval = candles.CANDLES_D
    resample = None
    match tf:
        case 'M1' | 'm1':
            interval = candles.CANDLES_M1
        case 'M5' | 'm5':
            interval = candles.CANDLES_M1
            resample = '5min'
        case 'M10' | 'm10':
            interval = candles.CANDLES_M10
        case 'M15' | 'm15':
            interval = candles.CANDLES_M1
            resample = '15min'
        case 'M30' | 'm30':
            interval = candles.CANDLES_M10
            resample = '30min'
        case 'M60' | 'm60':
            interval = candles.CANDLES_M60
        case 'D' | 'd':
            interval = candles.CANDLES_D
        case 'W' | 'w':
            interval = candles.CANDLES_W
        case 'M' | 'm':
            interval = candles.CANDLES_M
        case 'Q' | 'q':
            interval = candles.CANDLES_Q

    data = await candles.get_market_candles(http_client, security=secid, interval=interval, start=start, end=end, market=market, engine=engine)
    df = pd.DataFrame(data)
    if len(df)>0:
        df.rename(columns={'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'},
                inplace=True)
        df['Datetime'] = pd.to_datetime(df['begin'])
        df = df.set_index("Datetime")
        if resample is not None:
            df = df.resample(resample).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    return df


async def load_ticker_to_csv(http_client:httpx.AsyncClient,
                            secid:str,
                            path:str=os.path.join(".", "data"),
                            fname:str=None,
                            tf:str="D",
                            start:str|None=None,
                            end:str|None=None,
                            market='shares',
                            engine='stock'
    ):
    fname = fname if fname is not None else secid
    if len(path)>1:
        os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, fname) + f'_{tf.upper()}.csv.zip'
    df0 = None
    start = normalize_date(start)
    end = normalize_date(end)
    start_dt = datetime.strptime(start, '%Y-%m-%d').date() if start is not None else None

    if os.path.exists(fpath):
        df0_start = None
        df0_end = None
        df0 = pd.read_csv(fpath, sep=";")
        df0['Datetime'] = pd.to_datetime(df0['Datetime'])
        df0 = df0.set_index('Datetime')

        if len(df0)> 1:
            df0_start = df0.index[0].date()
            df0_end = df0.index[-1].date()
            if start_dt is not None:
                if start_dt < df0_start or start_dt >= df0_end:
                    df0 = None
                else:
                    start_dt = df0_end
                    df0 = df0[df0.index.date < start_dt]
            else:
                start_dt = df0_end
                df0 = df0[df0.index.date < start_dt]
        else:
            df0 = None

    df1 = await load_ticker_to_df(http_client, secid, tf,
                                start=(start_dt.strftime('%Y-%m-%d') if start_dt is not None else None),
                                end=end, market=market, engine=engine)
    df = df1
    if df0 is not None:
        df = pd.concat([df0, df1])
    if df.empty:
        return
    if start is not None:
        df = df[df.index.date >= datetime.strptime(start, '%Y-%m-%d').date()]
    if end is not None:
        df = df[df.index.date <= datetime.strptime(end, '%Y-%m-%d').date()]

    if len(df) > 0:
        col_list = ['Open','High', 'Low', 'Close','Volume']
        df.to_csv(fpath, index=True, lineterminator='\r\n', sep=';', compression='zip', columns=col_list)


async def load_list_stocks(http_client:httpx.AsyncClient):
    data = await history.get_board_securities(http_client, board='TQBR', market='shares', engine='stock',
                            columns=("SECID", "SHORTNAME", "CURRENCYID", "ISIN", "LISTLEVEL", "STATUS"))
    df = pd.DataFrame(data)
    df["idx"] = df["SECID"]
    df.set_index("idx", inplace=True)
    return df

async def load_list_futures(http_client:httpx.AsyncClient):
    data = await reference.get_statistics_series(http_client, market='forts', engine='futures')
    df = pd.DataFrame(data)
    df = df.rename(columns={"secid": "SECID"})
    df["idx"] = df["SECID"]
    df.set_index("idx", inplace=True)
    # Fill NaN separately: numeric columns with 0, string columns with ""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("")
    df["is_traded"] = df["is_traded"].astype(int)
    return df

def update_futures_date(df:pd.DataFrame):
    df = df.copy()
    df['expiration_date_dt'] = pd.to_datetime(df['expiration_date'], errors='coerce')
    # set End date
    empty_end = df['End'].isna() | (df['End'].astype(str).str.strip() == '')
    df.loc[empty_end, 'End'] = df.loc[empty_end, 'expiration_date']

    pattern = re.compile(r'^([A-Za-z]+)-(\d{1,2})\.(\d{2})$', re.IGNORECASE)
    # calc Start date
    for idx, row in df.iterrows():
        # skip Empty or Filled
        if pd.notna(row['Start']) and str(row['Start']).strip() != '':
            continue

        name = row['name']
        asset_code = row['asset_code']
        current_exp_date = row['expiration_date_dt']

        # skip bad dates
        if pd.isna(current_exp_date):
            continue

        # check name
        match = pattern.match(name)
        if not match:
            continue

        prefix, month, year_suffix = match.groups()

        # asset_code must be == prefix
        if prefix.upper() != str(asset_code).upper():
            continue

        # looking for future with prev date
        candidates = df[
            (df['asset_code'].astype(str).str.upper() == str(asset_code).upper()) &
            (df['expiration_date_dt'] < current_exp_date)
        ]

        if not candidates.empty:
            # looking for nearest
            prev_exp_date = candidates.sort_values(by='expiration_date_dt').iloc[-1]['expiration_date_dt']

            # substract 2 month
            new_start_date = prev_exp_date - relativedelta(months=2)

            df.at[idx, 'Start'] = new_start_date.strftime('%Y-%m-%d')

    df.drop(columns=['expiration_date_dt'], inplace=True)
    return df






if __name__ == "__main__":
    app = QApplication(sys.argv)
    T_DEFAULT_STYLE_NAME = app.style().objectName()
    T_DEFAULT_PALETTE = QPalette(app.palette())
    saved_theme = load_saved_theme()
    if saved_theme is None:
        apply_system_theme(app)
    else:
        set_app_theme(app, saved_theme == "dark")

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    df_stocks = pd.DataFrame()
    df_futures = pd.DataFrame()

    if os.path.isfile(T_STOCKS_FILE) and os.path.getsize(T_STOCKS_FILE) > 0:
        try:
            df = pd.read_csv(T_STOCKS_FILE, sep=';')
            if all(column in df.columns for column in T_STOCKS_COLS):
                df["idx"] = df["SECID"]
                df.set_index("idx", inplace=True)
                df_stocks = df.fillna("")
        except Exception:
            pass

    if os.path.isfile(T_FUTURES_FILE) and os.path.getsize(T_FUTURES_FILE) > 0:
        try:
            df = pd.read_csv(T_FUTURES_FILE, sep=';')
            if all(column in df.columns for column in T_FUTURES_COLS):
                df["idx"] = df["SECID"]
                df.set_index("idx", inplace=True)
                df_futures = df.fillna("")
                df_futures = update_futures_date(df_futures)
        except Exception:
            pass

    window = MainWindow(df_stocks, df_futures)
    window.resize(1400, 800)
    window.show()

    with loop:
        loop.run_forever()
