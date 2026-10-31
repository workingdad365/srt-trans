"""PySide6 기반 자막 번역 데스크톱 창."""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QAction, QDesktopServices, QFont, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QCompleter, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QTabWidget, QVBoxLayout, QWidget, QGroupBox,
)

from . import __version__
from .application import (
    SourceFile, TranslateRequest, build_tmdb_context, default_output_path,
    load_source, run_translation, save_result, validate_output_path,
)
from .config import ConfigManager
from .jobs import Job, MAX_LOG_LINES
from .providers import get_provider_class, list_providers, model_capabilities
from .providers.openrouter import ROUTE_VARIANTS, fetch_endpoints
from .srt_utils import extract_title_and_year, looks_like_series
from .tmdb import TMDBClient, TMDBError
from .workers import BackgroundTask

SRT_FILTER = "SRT 자막 (*.srt)"
ICON_PATH = Path(__file__).parent / "assets" / "icon.svg"


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _row(*widgets: QWidget) -> QHBoxLayout:
    layout = QHBoxLayout()
    for widget in widgets:
        layout.addWidget(widget)
    return layout


def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


def _seconds(value: float) -> str:
    seconds = int(max(0, value))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class DropArea(QFrame):
    """메인 창의 파일 입력과 같은 경로를 사용하는 드롭 영역."""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.window = window
        self.setAcceptDrops(True)
        self.setMinimumHeight(70)
        self.setStyleSheet("DropArea { border: 2px dashed palette(mid); border-radius: 6px; }")
        self.label = _label("SRT 파일을 여기에 끌어다 놓으세요")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        QVBoxLayout(self).addWidget(self.label)

    @staticmethod
    def local_file(event) -> str | None:
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            if Path(path).suffix.lower() == ".srt":
                return path
        return None

    def dragEnterEvent(self, event) -> None:
        if self.isEnabled() and self.local_file(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        if self.isEnabled() and (path := self.local_file(event)):
            event.acceptProposedAction()
            self.window.load_file(path)
        else:
            event.ignore()


class MainWindow(QMainWindow):
    def __init__(self, config: ConfigManager, initial_file: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.source: SourceFile | None = None
        self.result_source: SourceFile | None = None
        self.job: Job | None = None
        self.runner: BackgroundTask | None = None
        self._tasks: dict[str, BackgroundTask] = {}
        self._closing = False
        self._file_revision = 0
        self._tmdb_revision = 0
        self._capability_key: tuple[str, str] | None = None
        self._confirmed_selection: tuple[str, str] | None = None
        self._api_drafts: dict[str, str] = {}
        self._model_lists: dict[str, list[str]] = {}
        self._provider = ""
        self._started_at = 0.0
        self._start_index = 0
        self._last_logs = None
        self._tmdb_context = ""
        self._default_output = ""
        self._loading = True

        self.setWindowTitle(f"srt-trans {__version__} - SRT 자막 한국어 번역")
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(960, 850)
        self.setAcceptDrops(True)
        self._build_ui()
        self._model_timer = QTimer(self)
        self._model_timer.setSingleShot(True)
        self._model_timer.setInterval(250)
        self._model_timer.timeout.connect(self._refresh_capabilities)
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(100)
        self._progress_timer.timeout.connect(self._poll_job)
        self._load_settings()
        self._connect_signals()
        self._loading = False
        self._provider_changed()
        self.statusBar().showMessage(f"설정: {config.config_path}")
        if initial_file:
            QTimer.singleShot(0, partial(self.load_file, initial_file))

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)
        heading = QLabel("srt-trans")
        font = heading.font()
        font.setPointSize(18)
        font.setBold(True)
        heading.setFont(font)
        header = _row(heading, QLabel("SRT 자막 한국어 번역"))
        header.addStretch()
        header.addWidget(QLabel(f"v{__version__}"))
        root.addLayout(header)

        self.provider_combo = QComboBox()
        for info in list_providers():
            self.provider_combo.addItem(info.label, info.id)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_combo.setMinimumContentsLength(25)
        self.model_combo.lineEdit().setPlaceholderText("모델명 입력으로 검색, 목록에서 선택 또는 Enter로 확정")
        self.model_combo.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.model_combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self.model_combo.completer().setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        model_row = _row(QLabel("프로바이더"), self.provider_combo, QLabel("모델"), self.model_combo)
        root.addLayout(model_row)

        self.tabs = QTabWidget()
        self._build_translation_tab()
        self._build_api_tab()
        self._build_advanced_tab()
        root.addWidget(self.tabs, 1)

        self.start_button = QPushButton("번역 시작")
        self.cancel_button = QPushButton("취소")
        self.save_result_button = QPushButton("다른 이름으로 저장...")
        self.open_folder_button = QPushButton("결과 폴더 열기")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.save_result_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        root.addLayout(_row(self.start_button, self.cancel_button, self.save_result_button, self.open_folder_button))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)
        self.status_label = _label("SRT 파일을 열고 API 설정을 확인하세요.")
        root.addWidget(self.status_label)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(MAX_LOG_LINES)
        self.log_view.setMinimumHeight(90)
        self.log_view.setMaximumHeight(120)
        mono = QFont("Consolas" if sys.platform == "win32" else "Monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.log_view.setFont(mono)
        root.addWidget(self.log_view)
        self.setCentralWidget(central)

        open_action = QAction("열기", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file_dialog)
        self.addAction(open_action)
        save_action = QAction("다른 이름으로 저장", self)
        save_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_action.triggered.connect(self.save_result_dialog)
        self.addAction(save_action)

    def _tab(self, title: str) -> QVBoxLayout:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        self.tabs.addTab(scroll, title)
        return layout

    def _build_translation_tab(self) -> None:
        layout = self._tab("자막 및 작품 정보")
        self.source_box = QGroupBox("자막 파일")
        source_layout = QVBoxLayout(self.source_box)
        self.drop_area = DropArea(self)
        source_layout.addWidget(self.drop_area)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("SRT 파일 경로")
        self.open_button = QPushButton("파일 열기...")
        self.load_path_button = QPushButton("불러오기")
        source_layout.addLayout(_row(self.path_edit, self.load_path_button, self.open_button))
        self.file_info = _label()
        self.file_info.hide()
        source_layout.addWidget(self.file_info)
        layout.addWidget(self.source_box)

        output = QGroupBox("결과 저장")
        form = QFormLayout(output)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("원본과 같은 폴더의 원본명.kor.srt")
        self.output_browse = QPushButton("경로 선택...")
        form.addRow("출력 파일", _row(self.output_edit, self.output_browse))
        self.auto_save = QCheckBox("완료 또는 취소 시 번역된 결과 자동 저장")
        self.auto_save.setChecked(True)
        form.addRow("", self.auto_save)
        layout.addWidget(output)

        context = QGroupBox("작품 정보 및 번역 컨텍스트")
        form = QFormLayout(context)
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("파일명에서 자동 추출")
        self.year_edit = QLineEdit()
        self.year_edit.setPlaceholderText("연도")
        self.year_edit.setMaximumWidth(75)
        self.series_check = QCheckBox("TV 시리즈")
        self.tmdb_search_button = QPushButton("TMDB 검색")
        form.addRow("작품", _row(self.title_edit, self.year_edit, self.series_check, self.tmdb_search_button))
        self.tmdb_results_box = QWidget()
        results_layout = QVBoxLayout(self.tmdb_results_box)
        results_layout.setContentsMargins(0, 0, 0, 0)
        self.tmdb_results = QListWidget()
        self.tmdb_results.setMaximumHeight(110)
        self.tmdb_select_button = QPushButton("선택한 작품 적용")
        self.tmdb_clear_button = QPushButton("TMDB 정보 해제")
        results_layout.addWidget(self.tmdb_results)
        results_layout.addLayout(_row(self.tmdb_select_button, self.tmdb_clear_button))
        self.tmdb_results_box.hide()
        form.addRow("", self.tmdb_results_box)
        self.tmdb_note = _label("줄거리 입력란이 비어 있으면 선택한 TMDB 정보를 사용합니다.")
        form.addRow("", self.tmdb_note)
        self.tmdb_preview = QPlainTextEdit()
        self.tmdb_preview.setReadOnly(True)
        self.tmdb_preview.setMaximumHeight(100)
        self.tmdb_preview.hide()
        form.addRow("", self.tmdb_preview)
        self.story_edit = QPlainTextEdit()
        self.story_edit.setPlaceholderText(
            "[줄거리]\n상세 줄거리 입력\n\n[등장인물]\n한국어 이름, 관계, 나이, 말투, 존댓말/반말 관계 입력\n\n"
            "직접 입력한 내용이 TMDB 정보보다 우선합니다."
        )
        self.story_edit.setMinimumHeight(150)
        form.addRow("상세 줄거리 및\n등장인물 정보", self.story_edit)
        self.extra_edit = QPlainTextEdit()
        self.extra_edit.setPlaceholderText("작품과 무관하게 다음 실행에도 사용할 추가 번역 지시문")
        self.extra_edit.setMaximumHeight(80)
        form.addRow("추가 지시문", self.extra_edit)
        layout.addWidget(context)
        layout.addStretch()

    def _build_api_tab(self) -> None:
        layout = self._tab("API 설정")
        box = QGroupBox("API 키 및 모델")
        form = QFormLayout(box)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.tmdb_key_edit = QLineEdit()
        self.tmdb_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.tmdb_key_edit.setPlaceholderText("선택 사항: TMDB API 키 또는 읽기 액세스 토큰")
        self.key_link_button = QPushButton("API 키 발급 페이지")
        self.show_keys = QCheckBox("API 키 표시")
        self.save_keys_button = QPushButton("API 키 저장")
        self.load_models_button = QPushButton("모델 목록 불러오기")
        self.model_count = _label()
        form.addRow("현재 프로바이더 API 키", self.api_key_edit)
        form.addRow("TMDB API 키", self.tmdb_key_edit)
        form.addRow("", _row(self.show_keys, self.save_keys_button, self.key_link_button))
        form.addRow("모델 조회", self.load_models_button)
        form.addRow("", self.model_count)
        layout.addWidget(box)

        self.routing_box = QGroupBox("OpenRouter 라우팅")
        form = QFormLayout(self.routing_box)
        self.route_combo = QComboBox()
        self.route_combo.addItem("기본 (가격/속도 균형)", "")
        for value, item in ROUTE_VARIANTS.items():
            self.route_combo.addItem(item["label"], value)
        self.routing_providers = QLineEdit()
        self.routing_providers.setPlaceholderText("제공자 태그를 쉼표로 구분, 왼쪽부터 우선 사용")
        self.allow_fallbacks = QCheckBox("실패 시 다른 제공자 허용")
        self.deny_collection = QCheckBox("데이터 미수집 제공자만")
        self.endpoints_button = QPushButton("제공자 목록 불러오기")
        self.endpoint_list = QListWidget()
        self.endpoint_list.setMinimumHeight(100)
        self.endpoint_list.setMaximumHeight(180)
        self.endpoint_add_button = QPushButton("선택한 제공자 추가")
        form.addRow("라우팅 방식", self.route_combo)
        form.addRow("제공자 우선순위", self.routing_providers)
        form.addRow("", _row(self.allow_fallbacks, self.deny_collection))
        form.addRow("", _row(self.endpoints_button, self.endpoint_add_button))
        form.addRow("제공자 목록", self.endpoint_list)
        layout.addWidget(self.routing_box)
        layout.addStretch()

    def _build_advanced_tab(self) -> None:
        layout = self._tab("고급 설정")
        form = QFormLayout()
        self.batch_spin = _spin(1, 2000, 300)
        self.start_spin = _spin(1, 1_000_000, 1)
        self.language_edit = QLineEdit("ko")
        self.language_edit.setToolTip("번역 언어는 한국어이며 이 값은 출력 파일명에만 사용됩니다.")
        self.temperature_edit = QLineEdit()
        self.top_p_edit = QLineEdit()
        self.top_k_edit = QLineEdit()
        for edit in (self.temperature_edit, self.top_p_edit, self.top_k_edit):
            edit.setPlaceholderText("기본값 (비워 두면 전송하지 않음)")
        self.timeout_spin = _spin(10, 3600, 600)
        self.thinking_check = QCheckBox("Thinking 사용")
        self.budget_spin = _spin(0, 24576, 2048)
        self.effort_combo = QComboBox()
        self.streaming_check = QCheckBox("스트리밍 응답")
        self.strip_check = QCheckBox("문장 끝 마침표 제거")
        for title, widget in (
            ("배치 크기", self.batch_spin), ("시작 자막 번호 (1부터)", self.start_spin),
            ("출력 파일명 언어 코드", self.language_edit), ("Temperature (0~2)", self.temperature_edit),
            ("Top P (0~1)", self.top_p_edit), ("Top K (0 이상)", self.top_k_edit),
            ("응답 대기 시간 (초)", self.timeout_spin), ("사고 제어", self.thinking_check),
            ("Thinking 예산", self.budget_spin), ("추론 강도", self.effort_combo),
            ("", self.streaming_check), ("", self.strip_check),
        ):
            form.addRow(title, widget)
        layout.addLayout(form)
        self.model_notes = _label()
        layout.addWidget(self.model_notes)
        layout.addWidget(_label(
            "중단 후에는 저장된 부분 결과 파일을 열고 로그에 표시된 시작 자막 번호를 지정합니다.\n"
            "취소는 진행 중인 API 응답이 반환되거나 제한 시간에 도달한 뒤 처리됩니다."
        ))
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        self.model_combo.currentTextChanged.connect(self._model_text_changed)
        self.model_combo.activated.connect(self._model_changed)
        self.model_combo.lineEdit().returnPressed.connect(self._model_changed)
        self.load_models_button.clicked.connect(self.load_models)
        self.save_keys_button.clicked.connect(self.save_keys)
        self.show_keys.toggled.connect(self._show_keys)
        self.key_link_button.clicked.connect(self._open_key_page)
        self.open_button.clicked.connect(self.open_file_dialog)
        self.load_path_button.clicked.connect(lambda: self.load_file(self.path_edit.text()))
        self.path_edit.returnPressed.connect(lambda: self.load_file(self.path_edit.text()))
        self.output_browse.clicked.connect(self.choose_output)
        self.language_edit.textChanged.connect(self._update_output)
        self.tmdb_search_button.clicked.connect(self.search_tmdb)
        self.tmdb_select_button.clicked.connect(self.select_tmdb)
        self.tmdb_results.itemDoubleClicked.connect(self.select_tmdb)
        self.tmdb_clear_button.clicked.connect(self.clear_tmdb)
        self.title_edit.textEdited.connect(self._invalidate_tmdb)
        self.year_edit.textEdited.connect(self._invalidate_tmdb)
        self.series_check.toggled.connect(self._invalidate_tmdb)
        self.story_edit.textChanged.connect(self._update_tmdb_note)
        self.endpoints_button.clicked.connect(self.load_endpoints)
        self.endpoint_add_button.clicked.connect(self.add_endpoint)
        self.endpoint_list.itemDoubleClicked.connect(self.add_endpoint)
        self.start_button.clicked.connect(self.start_translation)
        self.cancel_button.clicked.connect(self.cancel_translation)
        self.save_result_button.clicked.connect(self.save_result_dialog)
        self.open_folder_button.clicked.connect(self.open_result_folder)
        self.thinking_check.toggled.connect(self._update_thinking)

    def _load_settings(self) -> None:
        values = self.config.all()
        index = self.provider_combo.findData(values.get("provider"))
        self.provider_combo.setCurrentIndex(max(0, index))
        self.tmdb_key_edit.setText(str(values.get("tmdb_api_key") or ""))
        for key, widget in (
            ("batch_size", self.batch_spin), ("thinking_budget", self.budget_spin),
            ("timeout", self.timeout_spin),
        ):
            try:
                widget.setValue(int(values[key]))
            except (ValueError, TypeError):
                pass
        for key, widget in (
            ("temperature", self.temperature_edit), ("top_p", self.top_p_edit),
            ("top_k", self.top_k_edit),
        ):
            widget.setText("" if values.get(key) is None else str(values[key]))
        for key, widget in (
            ("thinking", self.thinking_check), ("streaming", self.streaming_check),
            ("strip_trailing_period", self.strip_check),
        ):
            widget.setChecked(bool(values.get(key, True)))
        self.language_edit.setText(str(values.get("language_code") or "ko"))
        self.extra_edit.setPlainText(str(values.get("extra_instruction") or ""))
        self._preferred_effort = str(values.get("reasoning_effort") or "none")
        routing = values.get("routing") or {}
        self.route_combo.setCurrentIndex(max(0, self.route_combo.findData(routing.get("route_variant", ""))))
        self.routing_providers.setText(", ".join(routing.get("providers") or []))
        self.allow_fallbacks.setChecked(bool(routing.get("allow_fallbacks", True)))
        self.deny_collection.setChecked(bool(routing.get("deny_data_collection", False)))

    @property
    def selection(self) -> tuple[str, str]:
        return self.provider_combo.currentData(), self.model_combo.currentText().strip()

    def _provider_changed(self, *_args) -> None:
        if self._provider:
            self._api_drafts[self._provider] = self.api_key_edit.text()
        self._provider = self.provider_combo.currentData()
        self.api_key_edit.setText(self._api_drafts.get(self._provider, self.config.get_api_key(self._provider)))
        model = self.config.get_model(self._provider) or get_provider_class(self._provider).info.default_model
        self._set_models(self._model_lists.get(self._provider, []), model)
        self.routing_box.setVisible(self._provider == "openrouter")
        self.endpoint_list.clear()
        self._model_changed()

    def _set_models(self, models: list[str], current: str) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        self.model_combo.setEditText(current)
        self.model_combo.blockSignals(False)
        self.model_count.setText(
            f"불러온 모델 {len(models)}개. 상단 모델 입력란에 입력하면 일치하는 목록이 표시됩니다. "
            "목록에서 선택하거나 모델 ID를 입력한 뒤 Enter로 확정하세요."
        )

    def _model_text_changed(self, *_args) -> None:
        if self._loading:
            return
        # 검색 중인 문자열을 모델로 저장하거나 API에 조회하지 않음
        self._model_timer.stop()
        self._capability_key = None
        self.endpoint_list.clear()
        self._update_actions()

    def _model_changed(self, *_args) -> None:
        if self._loading:
            return
        self._confirmed_selection = self.selection
        self._capability_key = None
        self.endpoint_list.clear()
        self._model_timer.start()
        self._update_actions()

    def _refresh_capabilities(self) -> None:
        if self._closing or self.runner is not None or "capabilities" in self._tasks:
            return
        if self.selection != self._confirmed_selection:
            return
        provider, model = self.selection
        if not model:
            return
        self.config.update({"provider": provider, "models": {provider: model}})
        self._save_config()
        self.model_notes.setText("모델 지원 설정을 확인하는 중입니다.")
        self._start_task("capabilities", self.selection, partial(model_capabilities, provider, model))

    def _apply_capabilities(self, capabilities) -> None:
        for widget, supported in (
            (self.temperature_edit, capabilities.temperature), (self.top_p_edit, capabilities.top_p),
            (self.top_k_edit, capabilities.top_k), (self.streaming_check, capabilities.streaming),
        ):
            widget.setEnabled(supported)
        budget = capabilities.thinking_control == "budget"
        self.thinking_check.setEnabled(budget)
        self.budget_spin.setEnabled(budget and self.thinking_check.isChecked())
        preferred = self.effort_combo.currentText() or self._preferred_effort
        self.effort_combo.clear()
        self.effort_combo.addItems(capabilities.effort_choices)
        index = self.effort_combo.findText(preferred)
        self.effort_combo.setCurrentIndex(max(0, index))
        self.effort_combo.setEnabled(capabilities.thinking_control == "effort")
        self.model_notes.setText("\n".join(capabilities.notes) or "이 모델이 지원하는 설정만 활성화됩니다.")
        self._capability_key = self.selection

    def _update_thinking(self) -> None:
        self.budget_spin.setEnabled(self.thinking_check.isEnabled() and self.thinking_check.isChecked())

    def _show_keys(self, visible: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        self.api_key_edit.setEchoMode(mode)
        self.tmdb_key_edit.setEchoMode(mode)

    def _open_key_page(self) -> None:
        QDesktopServices.openUrl(QUrl(get_provider_class(self._provider).info.api_key_url))

    def _save_config(self) -> bool:
        if self.config.save():
            return True
        self.statusBar().showMessage("설정 파일을 저장하지 못했습니다. 저장 경로와 권한을 확인하세요.")
        return False

    def save_keys(self) -> None:
        self._api_drafts[self._provider] = self.api_key_edit.text().strip()
        self.config.update({
            "api_keys": {self._provider: self._api_drafts[self._provider]},
            "tmdb_api_key": self.tmdb_key_edit.text().strip(),
        })
        if self._save_config():
            self.statusBar().showMessage("API 키를 저장했습니다.", 5000)
        self._update_actions()

    def load_models(self) -> None:
        provider = self._provider
        key = self.api_key_edit.text().strip()
        if not key:
            self._error("API 키를 입력하세요.")
            return
        self._start_task("models", provider, partial(get_provider_class(provider).list_models, key))

    def load_endpoints(self) -> None:
        if self._provider == "openrouter" and self.selection[1]:
            self._start_task("endpoints", self.selection, partial(fetch_endpoints, self.selection[1]))

    def add_endpoint(self, *_args) -> None:
        item = self.endpoint_list.currentItem()
        if item is None:
            return
        providers = [part.strip() for part in self.routing_providers.text().split(",") if part.strip()]
        tag = item.data(Qt.ItemDataRole.UserRole)
        if tag not in providers:
            providers.append(tag)
        self.routing_providers.setText(", ".join(providers))

    def open_file_dialog(self) -> None:
        if self.runner is not None or self._closing:
            return
        start = str(self.source.path.parent) if self.source else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "SRT 자막 열기", start, SRT_FILTER)
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> None:
        if self.runner is not None or self._closing or "file" in self._tasks:
            return
        if not path.strip():
            self._error("SRT 파일 경로를 입력하세요.")
            return
        if not self._allow_discard_result():
            return
        self._start_task("file", None, partial(load_source, path))

    def _file_loaded(self, source: SourceFile) -> None:
        self.source = source
        self.job = None
        self.result_source = None
        self._file_revision += 1
        self.path_edit.setText(str(source.path))
        self.drop_area.label.setText(f"{source.name}\n{source.path.parent}")
        self.file_info.setText(f"자막 {source.subtitle_count:,}개")
        self.file_info.show()
        title, year = extract_title_and_year(source.name)
        self.title_edit.setText(title)
        self.year_edit.setText(year or "")
        self.series_check.setChecked(looks_like_series(source.name))
        self.story_edit.clear()
        self._invalidate_tmdb()
        self.start_spin.setRange(1, source.subtitle_count)
        self.start_spin.setValue(1)
        self.output_edit.clear()
        self._update_output()
        self.progress_bar.setValue(0)
        self.log_view.clear()
        self.status_label.setText(f"번역 준비: {source.name}")
        self.tabs.setCurrentIndex(0)

    def _update_output(self) -> None:
        if self.source is None:
            return
        code = self.language_edit.text().strip() or "ko"
        if not code.isascii() or not code.isalpha() or not 2 <= len(code) <= 8:
            return
        new_default = str(default_output_path(self.source, code))
        if not self.output_edit.text() or self.output_edit.text() == self._default_output:
            self.output_edit.setText(new_default)
        self._default_output = new_default

    def choose_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "번역 결과 저장 경로", self.output_edit.text(), SRT_FILTER)
        if path:
            self.output_edit.setText(path if Path(path).suffix else path + ".srt")

    def _invalidate_tmdb(self, *_args) -> None:
        self._tmdb_revision += 1
        self.tmdb_results.clear()
        self.tmdb_results_box.hide()
        self.clear_tmdb()

    def clear_tmdb(self) -> None:
        self._tmdb_revision += 1
        self._tmdb_context = ""
        self.tmdb_preview.clear()
        self.tmdb_preview.hide()
        self._update_tmdb_note()

    def _update_tmdb_note(self) -> None:
        if self.story_edit.toPlainText().strip():
            text = "직접 입력한 줄거리 및 등장인물 정보를 사용합니다."
        elif self._tmdb_context:
            text = "아래 TMDB 줄거리 및 출연진 정보를 사용합니다."
        else:
            text = "줄거리를 입력하거나 TMDB 검색 결과에서 작품을 선택하세요."
        self.tmdb_note.setText(text)

    def _tmdb_token(self) -> tuple:
        return (self._file_revision, self._tmdb_revision, self.title_edit.text(), self.year_edit.text(), self.series_check.isChecked())

    def search_tmdb(self) -> None:
        try:
            client = TMDBClient(self.tmdb_key_edit.text().strip())
        except TMDBError as exc:
            self._error(str(exc))
            return
        self._start_task("tmdb_search", self._tmdb_token(), partial(
            client.search, self.title_edit.text(), is_series=self.series_check.isChecked(), year=self.year_edit.text(),
        ))

    def select_tmdb(self, *_args) -> None:
        row = self.tmdb_results.currentItem()
        if row is None:
            return
        item = row.data(Qt.ItemDataRole.UserRole)
        try:
            client = TMDBClient(self.tmdb_key_edit.text().strip())
        except TMDBError as exc:
            self._error(str(exc))
            return
        self._start_task("tmdb_details", self._tmdb_token(), partial(client.details, item.id, is_series=item.is_series))

    def _start_task(self, kind: str, token, operation) -> None:
        if self._closing or self.runner is not None or kind in self._tasks:
            return
        worker = BackgroundTask(kind, token, operation, self)
        self._tasks[kind] = worker
        worker.finished.connect(self._task_finished)
        worker.start()
        self._update_actions()

    @Slot()
    def _task_finished(self) -> None:
        worker = self.sender()
        self._tasks.pop(worker.kind, None)
        try:
            if not self._closing:
                valid = self._task_is_current(worker)
                if valid and worker.error:
                    self._error(worker.error)
                elif valid:
                    self._apply_task_result(worker)
                if worker.kind == "capabilities" and not valid:
                    self._model_timer.start()
        finally:
            worker.deleteLater()
            self._update_actions()
            self._finish_close()

    def _task_is_current(self, worker: BackgroundTask) -> bool:
        if worker.kind in ("capabilities", "endpoints"):
            return worker.token == self.selection
        if worker.kind == "models":
            return worker.token == self._provider
        if worker.kind.startswith("tmdb_"):
            return worker.token == self._tmdb_token()
        return True

    def _apply_task_result(self, worker: BackgroundTask) -> None:
        result = worker.result
        if worker.kind == "file":
            self._file_loaded(result)
        elif worker.kind == "models":
            self._model_lists[self._provider] = result
            self._set_models(result, self.selection[1])
            if self.selection == self._confirmed_selection:
                self._model_changed()
        elif worker.kind == "capabilities":
            self._apply_capabilities(result)
        elif worker.kind == "endpoints":
            self.endpoint_list.clear()
            for endpoint in result:
                label = (
                    f"{endpoint['provider_name']} [{endpoint['tag']}] | "
                    f"컨텍스트 {endpoint.get('context_length') or '?'} | "
                    f"입력/출력 $/token: {endpoint.get('prompt_price') or '?'}/{endpoint.get('completion_price') or '?'} | "
                    f"구조화 출력 {'지원' if endpoint['supports_structured_outputs'] else '미지원'}"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, endpoint["tag"])
                item.setToolTip(
                    f"최대 출력: {endpoint.get('max_completion_tokens') or '?'} tokens\n"
                    f"최근 30분 가동률: {endpoint.get('uptime_last_30m') or '?'}\n"
                    f"양자화: {endpoint.get('quantization') or '?'}"
                )
                self.endpoint_list.addItem(item)
        elif worker.kind == "tmdb_search":
            self.tmdb_results.clear()
            for movie in result:
                item = QListWidgetItem(f"{movie.title} ({movie.year or '연도 미상'}) - {movie.original_title}")
                item.setData(Qt.ItemDataRole.UserRole, movie)
                item.setToolTip(movie.overview)
                self.tmdb_results.addItem(item)
            self.tmdb_results_box.setVisible(bool(result))
            self.tmdb_results.setCurrentRow(0)
            if not result:
                self.statusBar().showMessage("TMDB 검색 결과가 없습니다.", 5000)
        elif worker.kind == "tmdb_details":
            self.title_edit.setText(result.title)
            self.year_edit.setText(result.year)
            self._tmdb_context = build_tmdb_context(result)
            self.tmdb_preview.setPlainText(self._tmdb_context)
            self.tmdb_preview.show()
            self._update_tmdb_note()

    def _update_actions(self) -> None:
        idle = self.runner is None and not self._closing
        self.source_box.setEnabled(idle and "file" not in self._tasks)
        self.provider_combo.setEnabled(idle)
        self.model_combo.setEnabled(idle)
        self.tabs.setEnabled(idle)
        self.load_models_button.setEnabled(idle and "models" not in self._tasks)
        self.endpoints_button.setEnabled(idle and "endpoints" not in self._tasks)
        self.tmdb_search_button.setEnabled(idle and "tmdb_search" not in self._tasks)
        self.tmdb_select_button.setEnabled(idle and "tmdb_details" not in self._tasks)
        self.start_button.setEnabled(
            idle and self.source is not None and not self._tasks
            and self._capability_key == self.selection and bool(self.selection[1])
        )
        self.cancel_button.setEnabled(self.runner is not None and not self.job.cancel_event.is_set())
        self.save_result_button.setEnabled(idle and self.job is not None and self.job.result_text is not None)
        self.open_folder_button.setEnabled(idle and self.job is not None and bool(self.job.output_path))

    def _collect_request(self) -> TranslateRequest:
        def optional_number(edit: QLineEdit, kind):
            if not edit.isEnabled() or not edit.text().strip():
                return None
            try:
                return kind(edit.text().strip())
            except ValueError as exc:
                raise ValueError("Temperature, Top P, Top K에는 숫자를 입력하세요.") from exc

        return TranslateRequest(
            provider=self.selection[0], model=self.selection[1], title=self.title_edit.text().strip(),
            is_series=self.series_check.isChecked(), story_context=self.story_edit.toPlainText(),
            tmdb_context=self._tmdb_context, extra_instruction=self.extra_edit.toPlainText(),
            batch_size=self.batch_spin.value(), start_index=self.start_spin.value() - 1,
            temperature=optional_number(self.temperature_edit, float),
            top_p=optional_number(self.top_p_edit, float), top_k=optional_number(self.top_k_edit, int),
            thinking=self.thinking_check.isChecked(), thinking_budget=self.budget_spin.value(),
            reasoning_effort=self.effort_combo.currentText() if self.effort_combo.isEnabled() else None,
            streaming=self.streaming_check.isChecked() and self.streaming_check.isEnabled(),
            timeout=self.timeout_spin.value(), strip_trailing_period=self.strip_check.isChecked(),
            language_code=self.language_edit.text(), routing={
                "route_variant": self.route_combo.currentData(),
                "providers": [part.strip() for part in self.routing_providers.text().split(",") if part.strip()],
                "allow_fallbacks": self.allow_fallbacks.isChecked(),
                "deny_data_collection": self.deny_collection.isChecked(),
            },
        )

    def start_translation(self) -> None:
        if self.runner is not None or self.source is None or self._tasks or self._closing:
            return
        if self._capability_key != self.selection:
            return
        api_key = self.config.get_api_key(self._provider)
        if not api_key:
            self.tabs.setCurrentIndex(1)
            self._error("API 키를 먼저 저장하세요.")
            return
        try:
            payload = self._collect_request()
            target = None
            if self.auto_save.isChecked():
                if not self.output_edit.text().strip():
                    raise ValueError("출력 파일 경로를 지정하세요.")
                target = validate_output_path(Path(self.output_edit.text().strip()), self.source)
        except ValueError as exc:
            self._error(str(exc))
            return
        if target is not None and target.exists():
            answer = QMessageBox.question(self, "결과 파일 교체", f"기존 결과 파일을 교체하시겠습니까?\n{target}")
            if answer != QMessageBox.StandardButton.Yes:
                return
        if not self._allow_discard_result():
            return
        self.config.update(payload.settings())
        self._save_config()
        self.job = Job(self.source.name, target.name if target else default_output_path(self.source, payload.language_code).name)
        self.result_source = self.source
        self.job.set_progress(payload.start_index, self.source.subtitle_count)
        self._started_at = time.monotonic()
        self._start_index = payload.start_index
        self._last_logs = None
        self.log_view.clear()
        self.runner = BackgroundTask("translation", None, partial(
            run_translation, self.job, source=self.source, payload=payload, api_key=api_key, output_path=target,
        ), self)
        self.runner.finished.connect(self._translation_finished)
        self.runner.start()
        self._progress_timer.start()
        self._update_actions()

    def cancel_translation(self) -> None:
        if self.runner is not None:
            self.job.cancel()
            self._update_actions()
            self._poll_job()

    def _poll_job(self) -> None:
        if self.job is None:
            return
        snapshot = self.job.snapshot()
        total, done = snapshot["total"], snapshot["done"]
        self.progress_bar.setValue(round(done / total * 1000) if total else 0)
        elapsed = time.monotonic() - self._started_at
        status = {
            "pending": "준비 중", "running": "번역 중", "completed": "완료",
            "failed": "실패", "cancelled": "취소됨",
        }[snapshot["status"]]
        if self.runner is not None and self.job.cancel_event.is_set():
            status = "취소 대기 중 (현재 API 응답 정리)"
        text = f"{status} | {done:,}/{total:,} | 경과 {_seconds(elapsed)}"
        if self.runner is not None and done > self._start_index and snapshot["status"] == "running":
            remaining = elapsed / (done - self._start_index) * (total - done)
            text += f" | 예상 남은 시간 {_seconds(remaining)}"
        if snapshot["output_path"]:
            text += f"\n저장: {snapshot['output_path']}"
        elif snapshot["has_result"]:
            text += "\n결과가 메모리에 있습니다. '다른 이름으로 저장'을 사용하세요."
        self.status_label.setText(text)
        if snapshot["logs"] != self._last_logs:
            self._last_logs = snapshot["logs"]
            scrollbar = self.log_view.verticalScrollBar()
            follow = scrollbar.value() >= scrollbar.maximum()
            old_value = scrollbar.value()
            self.log_view.setPlainText("\n".join(
                f"[{entry['time']}] [{entry['level']}] {entry['message']}" for entry in snapshot["logs"]
            ))
            scrollbar.setValue(scrollbar.maximum() if follow else old_value)

    @Slot()
    def _translation_finished(self) -> None:
        worker = self.runner
        if worker.error:
            self.job.log("error", worker.error)
            self.job.set_status("failed", worker.error)
        self.runner = None
        self._progress_timer.stop()
        self._poll_job()
        worker.deleteLater()
        self._update_actions()
        self._finish_close()

    def save_result_dialog(self) -> None:
        if self.runner is not None or self.job is None or self.job.result_text is None:
            return
        start = self.job.output_path or str(self.result_source.path.parent / self.job.output_name)
        path, _ = QFileDialog.getSaveFileName(self, "번역 결과 다른 이름으로 저장", start, SRT_FILTER)
        if not path:
            return
        target = Path(path if Path(path).suffix else path + ".srt")
        try:
            target = save_result(self.job.result_text, target, self.result_source)
        except (OSError, ValueError) as exc:
            self._error(str(exc))
            return
        self.job.set_result(self.job.result_text, str(target))
        self.job.log("success", f"저장 완료: {target}")
        self._poll_job()
        self._update_actions()

    def open_result_folder(self) -> None:
        if self.job is not None and self.job.output_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.job.output_path).parent)))

    def _allow_discard_result(self) -> bool:
        if self.job is None or self.job.result_text is None or self.job.output_path:
            return True
        answer = QMessageBox.question(
            self, "저장하지 않은 번역 결과", "저장하지 않은 번역 결과가 있습니다. 결과를 버리고 계속하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _error(self, message: str) -> None:
        QMessageBox.warning(self, "srt-trans", message)

    def dragEnterEvent(self, event) -> None:
        if self.runner is None and not self._closing and DropArea.local_file(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        if self.runner is None and not self._closing and (path := DropArea.local_file(event)):
            event.acceptProposedAction()
            self.load_file(path)
        else:
            event.ignore()

    def _finish_close(self) -> None:
        if self._closing and not self._tasks and self.runner is None:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event) -> None:
        if self.runner is not None and not self._closing:
            answer = QMessageBox.question(
                self, "번역 진행 중", "번역을 취소하고 부분 결과를 정리한 뒤 종료하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if self.runner is not None or self._tasks:
            self._closing = True
            self._model_timer.stop()
            self.cancel_translation()
            self._update_actions()
            self.statusBar().showMessage("진행 중인 작업을 정리한 뒤 종료합니다.")
            event.ignore()
            return
        if not self._allow_discard_result():
            self._closing = False
            if self._capability_key != self.selection:
                self._model_timer.start()
            self._update_actions()
            event.ignore()
            return
        self._model_timer.stop()
        self._progress_timer.stop()
        event.accept()


def main(initial_file: str | None = None) -> int:
    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("drasys.srt-trans")
        except (AttributeError, OSError):
            pass
    app = QApplication([sys.argv[0]])
    app.setApplicationName("srt-trans")
    app.setApplicationDisplayName("srt-trans")
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow(ConfigManager(), initial_file)
    window.show()
    return app.exec()
