
from pathlib import Path
import json
import traceback

import numpy as np

from PySide6.QtCore import QThread, Signal, Qt, QSize
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFrame,
    QLabel,
    QPushButton,
    QToolButton,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QStatusBar,
    QProgressBar,
    QDoubleSpinBox,
    QCheckBox,
    QScrollArea,
    QStyle
)

from core.stl_loader import load_stl
from core.result_manager import ResultManager

from stages.stage1_geometry import execute_stage1
from stages.stage2_surface_identification import execute_stage2
from stages.stage3_spray_trajectory import (
    execute_stage3,
    DEFAULT_STANDOFF_MM,
    DEFAULT_SAMPLING_SPACING_MM,
    DEFAULT_SPRAY_ANGLE_DEG,
)

from ui.viewer_3d import ProductViewer


# ============================================================
# UNIT DEFINITIONS
# ============================================================

SUPPORTED_UNITS = {
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "in": "in",
    "inch": "in",
    "inches": "in",
    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
    "unknown": "unknown",
}

MM_PER_UNIT = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "ft": 304.8,
}

STAGE_PRODUCT_SETUP = 0
STAGE_GEOMETRY_INSPECTION = 1
STAGE_SURFACE_IDENTIFICATION = 2
STAGE_SPRAY_TRAJECTORY = 3
STAGE_ROBOT_CONFIGURATION = 4
STAGE_EXECUTION = 5

FACE_EXCLUDED = 0
FACE_PAINTABLE_CANDIDATE = 1
FACE_REQUIRES_INSPECTION = 2
FACE_INVALID = 3

DECISION_REQUIRES_INSPECTION = "requires_inspection"
DECISION_PAINT = "paint"
DECISION_NO_PAINT = "no_paint"

DECISION_TO_FACE_STATUS = {
    DECISION_REQUIRES_INSPECTION: FACE_REQUIRES_INSPECTION,
    DECISION_PAINT: FACE_PAINTABLE_CANDIDATE,
    DECISION_NO_PAINT: FACE_EXCLUDED,
}

WORKFLOW_STAGES = [
    {
        "name": "PRODUCT SETUP",
        "description": (
            "Load the product CAD/STL model and establish "
            "the active product coordinate system."
        ),
        "action": "LOAD STL PRODUCT",
    },
    {
        "name": "GEOMETRY INSPECTION",
        "description": (
            "Inspect mesh integrity, dimensions, connected "
            "components, and geometric surface patches."
        ),
        "action": "ANALYZE PRODUCT GEOMETRY",
    },
    {
        "name": "SURFACE IDENTIFICATION",
        "description": (
            "Inspect each geometric patch and assign its "
            "painting decision."
        ),
        "action": "IDENTIFY PAINTABLE SURFACES",
    },
    {
        "name": "SPRAY TRAJECTORY",
        "description": (
            "Generate spray paths from manually approved "
            "Stage 2 faces."
        ),
        "action": "GENERATE SPRAY TRAJECTORY",
    },
    {
        "name": "ROBOT CONFIGURATION",
        "description": (
            "Configure the dual-arm robot system, allocate "
            "painting tasks, and verify robot reachability."
        ),
        "action": "ROBOT CONFIGURATION NOT AVAILABLE",
    },
    {
        "name": "EXECUTION & MONITORING",
        "description": (
            "Simulate robot motion, verify collision clearance, "
            "and monitor painting execution."
        ),
        "action": "EXECUTION NOT AVAILABLE",
    },
]

PIPELINE_ITEMS = [
    "Product loading",
    "Geometry inspection",
    "Paintable surface identification",
    "Spray trajectory generation",
    "Robot task allocation",
    "Collision verification",
    "Simulation and execution",
]


# ============================================================
# UNIT UTILITIES
# ============================================================

def normalize_unit(unit):
    if not isinstance(unit, str):
        return "unknown"

    return SUPPORTED_UNITS.get(unit.strip().lower(), "unknown")


def get_mesh_source_unit(mesh):
    metadata = getattr(mesh, "metadata", {})

    if not isinstance(metadata, dict):
        return "unknown"

    return normalize_unit(metadata.get("source_unit"))


def convert_coordinate_value(value, source_unit, display_unit):
    source_unit = normalize_unit(source_unit)
    display_unit = normalize_unit(display_unit)

    if source_unit == "unknown" or display_unit == "unknown":
        return None

    mm_value = float(value) * MM_PER_UNIT[source_unit]
    return mm_value / MM_PER_UNIT[display_unit]


def get_display_area_factor(source_unit, display_unit):
    source_unit = normalize_unit(source_unit)
    display_unit = normalize_unit(display_unit)

    if source_unit == "unknown" or display_unit == "unknown":
        return None

    factor = MM_PER_UNIT[source_unit] / MM_PER_UNIT[display_unit]
    return factor ** 2


def get_display_volume_factor(source_unit, display_unit):
    source_unit = normalize_unit(source_unit)
    display_unit = normalize_unit(display_unit)

    if source_unit == "unknown" or display_unit == "unknown":
        return None

    factor = MM_PER_UNIT[source_unit] / MM_PER_UNIT[display_unit]
    return factor ** 3


# ============================================================
# OPERATION WORKER
# ============================================================

class OperationWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, *args, **kwargs):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.operation(*self.args, **self.kwargs)
            self.succeeded.emit(result)
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))


# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QMainWindow):

    def __init__(self, project_root: Path, fonts: dict):
        super().__init__()

        self.project_root = Path(project_root).resolve()
        self.fonts = fonts

        self.result_manager = ResultManager(self.project_root)

        self.current_mesh = None
        self.current_stl_path = None

        self.current_stage1_result = None
        self.current_report = None
        self.current_stage2_result = None
        self.current_stage3_result = None

        self.source_unit_from_stl = "unknown"
        self.coordinate_unit = "unknown"
        self.manual_unit_selected = False

        self.current_stage3_parameters = None
        self.pending_stage3_parameters = None

        self._worker = None
        self._operation_running = False

        self.patch_inspection = {}
        self.patch_row_widgets = {}
        self.selected_patch_id = None

        self._stage2_automatic_face_status_labels = None
        self.patch_inspection_path = None

        # Right sidebar visibility state.
        self._right_sidebar_visible = True

        # Individual right-panel section visibility state.
        self._right_section_bodies = {}
        self._right_section_buttons = {}

        self.setWindowTitle("OPENARMX PAINTING CONTROL SYSTEM")
        self.resize(1600, 950)
        self.setMinimumSize(1200, 750)

        self.active_stage = STAGE_PRODUCT_SETUP

        self.stage_completed = {
            STAGE_PRODUCT_SETUP: False,
            STAGE_GEOMETRY_INSPECTION: False,
            STAGE_SURFACE_IDENTIFICATION: False,
            STAGE_SPRAY_TRAJECTORY: False,
            STAGE_ROBOT_CONFIGURATION: False,
            STAGE_EXECUTION: False,
        }

        self._build_ui()
        self._refresh_workflow_ui()

        default_stl = self.project_root / "models" / "product.STL"

        if default_stl.exists():
            self.load_part(default_stl)

    # ========================================================
    # UI HELPERS
    # ========================================================

    def _panel(self):
        frame = QFrame()
        frame.setObjectName("Panel")
        return frame

    def _section(self, text):
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    def _metric(self, title, value="--"):
        name = QLabel(title)
        name.setObjectName("Muted")

        result = QLabel(value)
        result.setObjectName("MetricValue")

        return name, result

    def _make_collapsible_section(self, title, section_id, content):
        """
        Create an accordion-style right-sidebar section.

        Only one section can be expanded at a time.
        Clicking the expanded section collapses it.
        Existing content widgets are preserved.
        """

        container = QWidget()
        container.setObjectName("CollapsibleSection")

        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(3)

        # Check whether another section is already expanded.
        another_section_open = any(
            button.isChecked()
            for button in self._right_section_buttons.values()
        )

        header = QToolButton()
        header.setText(title)
        header.setCheckable(True)

        # Only the first section starts expanded.
        initially_expanded = not another_section_open
        header.setChecked(initially_expanded)

        header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        header.setArrowType(
            Qt.DownArrow if initially_expanded else Qt.RightArrow
        )
        header.setObjectName("CollapsibleSectionHeader")

        header.setSizePolicy(
            header.sizePolicy().horizontalPolicy(),
            header.sizePolicy().verticalPolicy(),
        )

        content.setObjectName("CollapsibleSectionBody")
        content.setVisible(initially_expanded)

        def on_toggled(expanded, body=content, button=header):
            # If this section opens, close all other sections.
            if expanded:
                for other_id, other_button in (
                    self._right_section_buttons.items()
                ):
                    if other_button is not button:
                        other_button.setChecked(False)

            body.setVisible(expanded)

            button.setArrowType(
                Qt.DownArrow if expanded else Qt.RightArrow
            )

        header.toggled.connect(on_toggled)

        container_layout.addWidget(header)
        container_layout.addWidget(content)

        # Register this section.
        self._right_section_bodies[section_id] = content
        self._right_section_buttons[section_id] = header

        return container



    def _toggle_right_sidebar(self):
        self._right_sidebar_visible = not self._right_sidebar_visible

        self.right_panel.setVisible(
            self._right_sidebar_visible
        )

        if self._right_sidebar_visible:
            # Sidebar visible → LEFT arrow → click to collapse.
            self.right_sidebar_toggle.setIcon(
                self.style().standardIcon(
                    QStyle.StandardPixmap.SP_ArrowRight
                )
            )

            self.right_sidebar_toggle.setToolTip(
                "Collapse right sidebar"
            )

        else:
            # Sidebar hidden → RIGHT arrow → click to expand.
            self.right_sidebar_toggle.setIcon(
                self.style().standardIcon(
                    QStyle.StandardPixmap.SP_ArrowLeft
                )
            )

            self.right_sidebar_toggle.setToolTip(
                "Expand right sidebar"
            )

    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    
    def _collapse_patch_inspection_section(self):
        """
        Collapse the Patch Inspection accordion section.
        Keep its header available in the sidebar.
        """

        button = self._right_section_buttons.get(
            "patch_inspection"
        )

        if button is not None:
            button.setChecked(False)

    def _expand_patch_inspection_section(self):
        button = self._right_section_buttons.get(
            "patch_inspection"
        )

        if button is not None:
            button.setChecked(True)


    def _set_patch_inspection_readonly(self, readonly=True):
        for widgets in self.patch_row_widgets.values():

            checkbox = widgets.get("checkbox")
            decision = widgets.get("decision")

            if checkbox is not None:
                checkbox.setEnabled(not readonly)

            if decision is not None:
                decision.setEnabled(not readonly)

            # Keep 3D patch selection available.
            select_button = widgets.get("select_button")

            if select_button is not None:
                select_button.setEnabled(True)

    def _build_ui(self):
        central = QWidget()

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(8)

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        root.addLayout(self._build_header())

        # ----------------------------------------------------
        # WORKFLOW NAVIGATION
        # ----------------------------------------------------

        # Keep workflow navigation objects for internal state,
        # but do not display the navigation bar.
        self.workflow_bar = self._build_workflow_bar()
        self.workflow_bar.hide()

        # ----------------------------------------------------
        # MAIN WORKSPACE
        # ----------------------------------------------------

        workspace = QHBoxLayout()
        workspace.setSpacing(6)

        # 3D Viewer
        self.viewer = ProductViewer()

        # Left sidebar
        self.left_panel = self._build_left_panel()
        self.left_panel.setFixedWidth(330)

        # Right sidebar
        self.right_panel = self._build_right_panel()
        self.right_panel.setFixedWidth(400)

        # ----------------------------------------------------
        # RIGHT SIDEBAR TOGGLE BUTTON
        # ----------------------------------------------------

        self._right_sidebar_visible = True

        self.right_sidebar_toggle = QToolButton()
        self.right_sidebar_toggle.setFixedSize(48, 78)

        self.right_sidebar_toggle.setIconSize(QSize(24, 24))
        self.right_sidebar_toggle.setToolButtonStyle(
            Qt.ToolButtonIconOnly
        )
        self.right_sidebar_toggle.setAutoRaise(False)

        # Initial state: sidebar visible, arrow points LEFT.
        self.right_sidebar_toggle.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_ArrowRight
            )
        )

        self.right_sidebar_toggle.setToolTip(
            "Collapse right sidebar"
        )

        self.right_sidebar_toggle.setStyleSheet("""
            QToolButton {
                background-color: #14232E;
                border: 2px solid #203B4B;
                border-radius: 8px;
                padding: 8px;
            }

            QToolButton:hover {
                background-color: #203B4B;
                border: 2px solid #00BCD4;
            }

            QToolButton:pressed {
                background-color: #0D1822;
            }
        """)

        self.right_sidebar_toggle.clicked.connect(
            self._toggle_right_sidebar
        )

        # ----------------------------------------------------
        # WORKSPACE WIDGET ORDER
        # ----------------------------------------------------

        workspace.addWidget(self.left_panel)
        workspace.addWidget(self.viewer, 1)

        workspace.addWidget(
            self.right_sidebar_toggle,
            0,
            Qt.AlignVCenter,
        )

        workspace.addWidget(self.right_panel)

        root.addLayout(workspace, 1)

        # ----------------------------------------------------
        # CENTRAL WIDGET
        # ----------------------------------------------------

        self.setCentralWidget(central)

        # ----------------------------------------------------
        # STATUS BAR
        # ----------------------------------------------------

        self.status_bar = QStatusBar()
        self.status_bar.showMessage(
            "SYSTEM READY  |  WAITING FOR PRODUCT"
        )
        self.setStatusBar(self.status_bar)



    # ========================================================
    # HEADER
    # ========================================================

    def _build_header(self):
        header = QHBoxLayout()
        title_block = QVBoxLayout()

        title = QLabel("OPENARMX PAINTING CONTROL SYSTEM")
        title.setObjectName("MainTitle")

        subtitle = QLabel(
            "DUAL-ARM INDUSTRIAL SPRAY PAINTING  "
            "•  PRODUCT → GEOMETRY → SURFACES → "
            "TRAJECTORY → ROBOTS → EXECUTION"
        )
        subtitle.setObjectName("Subtitle")

        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        header.addLayout(title_block)
        header.addStretch()

        self.system_status = QLabel("● SYSTEM ONLINE")
        self.system_status.setStyleSheet(
            "color: #00E676; font-weight: bold;"
        )
        header.addWidget(self.system_status)

        return header

    # ========================================================
    # WORKFLOW NAVIGATION
    # ========================================================

    def _build_workflow_bar(self):
        panel = self._panel()

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self.workflow_buttons = []

        for index, stage in enumerate(WORKFLOW_STAGES):
            button = QPushButton(stage["name"])
            button.setCheckable(True)

            button.clicked.connect(
                lambda checked=False, i=index:
                self._select_workflow_stage(i)
            )

            layout.addWidget(button, 1)
            self.workflow_buttons.append(button)

        return panel

    # ========================================================
    # LEFT PANEL
    # ========================================================

    def _build_left_panel(self):
        panel = self._panel()

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)

        layout.addWidget(self._section("PRODUCT"))

        self.part_name_label = QLabel("NO PRODUCT LOADED")
        self.part_name_label.setWordWrap(True)
        self.part_name_label.setObjectName("MetricValue")
        layout.addWidget(self.part_name_label)

        self.load_button = QPushButton("LOAD STL PRODUCT")
        self.load_button.clicked.connect(self.load_stl_dialog)
        layout.addWidget(self.load_button)

        layout.addSpacing(5)
        layout.addWidget(self._section("STL COORDINATE UNIT"))

        self.coordinate_unit_selector = QComboBox()
        self.coordinate_unit_selector.addItem(
            "Select coordinate unit",
            "unknown",
        )

        for unit in ("mm", "cm", "m", "in", "ft"):
            self.coordinate_unit_selector.addItem(unit, unit)

        self.coordinate_unit_selector.currentIndexChanged.connect(
            self._on_coordinate_unit_changed
        )
        layout.addWidget(self.coordinate_unit_selector)

        layout.addSpacing(5)
        layout.addWidget(self._section("DISPLAY UNIT"))

        self.unit_selector = QComboBox()
        self.unit_selector.addItems(["mm", "cm", "m", "in", "ft"])
        self.unit_selector.setCurrentText("mm")
        self.unit_selector.currentTextChanged.connect(
            self._on_unit_changed
        )
        layout.addWidget(self.unit_selector)

        layout.addSpacing(5)
        layout.addWidget(self._section("GEOMETRY"))

        self.metric_labels = {}
        self.metric_title_labels = {}

        metrics = [
            ("VERTICES", "vertices"),
            ("TRIANGLES", "triangles"),
            ("SURFACE AREA", "area"),
            ("VOLUME", "volume"),
        ]

        grid = QGridLayout()
        grid.setVerticalSpacing(10)

        for row, (name, key) in enumerate(metrics):
            label, value = self._metric(name)
            grid.addWidget(label, row, 0)
            grid.addWidget(value, row, 1)
            self.metric_title_labels[key] = label
            self.metric_labels[key] = value

        layout.addLayout(grid)

        layout.addSpacing(8)
        layout.addWidget(self._section("BOUNDING DIMENSIONS"))

        self.dimension_labels = {}
        self.dimension_title_labels = {}

        dimension_grid = QGridLayout()

        for row, (title, axis) in enumerate([
            ("WIDTH (X)", "W"),
            ("DEPTH (Y)", "D"),
            ("HEIGHT (Z)", "H"),
        ]):
            label, value = self._metric(title)
            dimension_grid.addWidget(label, row, 0)
            dimension_grid.addWidget(value, row, 1)
            self.dimension_title_labels[axis] = label
            self.dimension_labels[axis] = value

        layout.addLayout(dimension_grid)

        layout.addSpacing(8)
        layout.addWidget(self._section("VIEW"))

        self.reset_camera_button = QPushButton("RESET CAMERA")
        self.reset_camera_button.clicked.connect(
            self.viewer.reset_camera
        )
        layout.addWidget(self.reset_camera_button)

        layout.addSpacing(8)
        layout.addWidget(self._section("SPRAY PARAMETERS"))

        parameter_grid = QGridLayout()
        parameter_grid.setVerticalSpacing(7)

        parameter_grid.addWidget(QLabel("STANDOFF (mm)"), 0, 0)

        self.standoff_spin = QDoubleSpinBox()
        self.standoff_spin.setRange(1.0, 2000.0)
        self.standoff_spin.setDecimals(1)
        self.standoff_spin.setSingleStep(5.0)
        self.standoff_spin.setValue(float(DEFAULT_STANDOFF_MM))
        parameter_grid.addWidget(self.standoff_spin, 0, 1)

        parameter_grid.addWidget(QLabel("SPACING (mm)"), 1, 0)

        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(1.0, 500.0)
        self.spacing_spin.setDecimals(1)
        self.spacing_spin.setSingleStep(1.0)
        self.spacing_spin.setValue(
            float(DEFAULT_SAMPLING_SPACING_MM)
        )
        parameter_grid.addWidget(self.spacing_spin, 1, 1)

        parameter_grid.addWidget(QLabel("SPRAY ANGLE (°)"), 2, 0)

        self.spray_angle_spin = QDoubleSpinBox()
        self.spray_angle_spin.setRange(
            float(DEFAULT_SPRAY_ANGLE_DEG),
            float(DEFAULT_SPRAY_ANGLE_DEG),
        )
        self.spray_angle_spin.setDecimals(1)
        self.spray_angle_spin.setValue(
            float(DEFAULT_SPRAY_ANGLE_DEG)
        )
        self.spray_angle_spin.setEnabled(False)
        parameter_grid.addWidget(self.spray_angle_spin, 2, 1)

        layout.addLayout(parameter_grid)

        self.standoff_spin.valueChanged.connect(
            self._on_spray_parameter_changed
        )
        self.spacing_spin.valueChanged.connect(
            self._on_spray_parameter_changed
        )
        self.spray_angle_spin.valueChanged.connect(
            self._on_spray_parameter_changed
        )

        layout.addStretch()
        return panel

    # ========================================================
    # RIGHT PANEL - COLLAPSIBLE SECTIONS
    # ========================================================

    
    
    def _build_right_panel(self):
        panel = self._panel()

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ----------------------------------------------------
        # ACTIVE WORKFLOW SECTION
        # ----------------------------------------------------
        active_content = QWidget()
        active_layout = QVBoxLayout(active_content)
        active_layout.setContentsMargins(5, 2, 5, 4)
        active_layout.setSpacing(5)

        self.current_stage_label = QLabel("PRODUCT SETUP")
        self.current_stage_label.setObjectName("MetricValue")
        self.current_stage_label.setWordWrap(True)
        active_layout.addWidget(self.current_stage_label)

        self.workflow_description = QLabel(
            "Load a product model to begin."
        )
        self.workflow_description.setObjectName("Muted")
        self.workflow_description.setWordWrap(True)
        active_layout.addWidget(self.workflow_description)

        self.workflow_status_label = QLabel(
            "STATUS: WAITING FOR PRODUCT"
        )
        self.workflow_status_label.setStyleSheet(
            "color: #FFD740; font-weight: bold;"
        )
        self.workflow_status_label.setWordWrap(True)
        active_layout.addWidget(self.workflow_status_label)

        active_section = self._make_collapsible_section(
            "ACTIVE WORKFLOW",
            "active_workflow",
            active_content,
        )

        self._right_section_buttons[
            "active_workflow"
        ].setChecked(False)

        layout.addWidget(active_section)

        # ----------------------------------------------------
        # PROCESS OVERVIEW SECTION
        # ----------------------------------------------------
        process_content = QWidget()
        process_layout = QVBoxLayout(process_content)
        process_layout.setContentsMargins(5, 2, 5, 4)
        process_layout.setSpacing(4)

        self.pipeline_labels = []

        for item in PIPELINE_ITEMS:
            label = QLabel("○  " + item)
            label.setObjectName("Muted")
            label.setWordWrap(True)

            process_layout.addWidget(label)
            self.pipeline_labels.append(label)

        process_section = self._make_collapsible_section(
            "PROCESS OVERVIEW",
            "process_overview",
            process_content,
        )

        layout.addWidget(process_section)

        # ----------------------------------------------------
        # STAGE SUMMARY SECTION
        # ----------------------------------------------------
        summary_content = QWidget()
        summary_layout = QVBoxLayout(summary_content)
        summary_layout.setContentsMargins(5, 2, 5, 4)
        summary_layout.setSpacing(4)

        self.analysis_summary = QLabel("AWAITING PRODUCT")
        self.analysis_summary.setWordWrap(True)
        self.analysis_summary.setObjectName("Muted")
        summary_layout.addWidget(self.analysis_summary)

        self.summary_section = QLabel("STAGE SUMMARY")
        self.summary_section.setObjectName("Muted")
        summary_layout.addWidget(self.summary_section)

        summary_section = self._make_collapsible_section(
            "STAGE SUMMARY",
            "stage_summary",
            summary_content,
        )

        self._right_section_buttons[
            "stage_summary"
        ].setChecked(False)

        layout.addWidget(summary_section)

        # ----------------------------------------------------
        # PAINTING PATCHES SECTION
        # ----------------------------------------------------
        self.patch_inspection_panel = QFrame()
        self.patch_inspection_panel.setObjectName(
            "PatchInspectionPanel"
        )

        patch_outer_layout = QVBoxLayout(
            self.patch_inspection_panel
        )
        patch_outer_layout.setContentsMargins(0, 0, 0, 0)
        patch_outer_layout.setSpacing(0)

        patch_content = QWidget()
        patch_layout = QVBoxLayout(patch_content)
        patch_layout.setContentsMargins(5, 2, 5, 4)
        patch_layout.setSpacing(5)

        # ----------------------------------------------------
        # PAINTING SELECTION SUMMARY
        # ----------------------------------------------------
        self.patch_progress_label = QLabel(
            "PAINTABLE PATCHES: 0 / 0 (0%)"
        )
        self.patch_progress_label.setObjectName("MetricValue")
        patch_layout.addWidget(self.patch_progress_label)

        # ----------------------------------------------------
        # PAINTING SELECTION STATUS
        # Exactly one status label
        # ----------------------------------------------------
        self.patch_inspection_status_label = QLabel(
            "PAINTING SELECTION INCOMPLETE"
        )
        self.patch_inspection_status_label.setObjectName(
            "PatchInspectionStatus"
        )
        self.patch_inspection_status_label.setAlignment(
            Qt.AlignCenter
        )
        self.patch_inspection_status_label.setWordWrap(True)
        self.patch_inspection_status_label.setStyleSheet("""
            QLabel {
                color: #FFD740;
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.patch_inspection_status_label.hide()

        patch_layout.addWidget(
            self.patch_inspection_status_label
        )

        # ----------------------------------------------------
        # PAINTING SELECTION PROGRESS
        # Exactly one patch progress bar
        # ----------------------------------------------------
        self.patch_progress_bar = QProgressBar()
        self.patch_progress_bar.setRange(0, 100)
        self.patch_progress_bar.setValue(100)
        self.patch_progress_bar.setFormat("%p%")
        self.patch_progress_bar.setFixedHeight(10)

        patch_layout.addWidget(self.patch_progress_bar)

        # ----------------------------------------------------
        # INSTRUCTIONS
        # ----------------------------------------------------
        self.patch_instruction_label = QLabel(
            "All patches are selected for painting by default. "
            "Uncheck a patch to exclude it from painting. "
            "SPRAY TRAJECTORY becomes available when all "
            "patches are selected."
        )
        self.patch_instruction_label.setWordWrap(True)
        self.patch_instruction_label.setObjectName("Muted")

        patch_layout.addWidget(
            self.patch_instruction_label
        )

        # ----------------------------------------------------
        # PATCH LIST
        # ----------------------------------------------------
        self.patch_scroll = QScrollArea()
        self.patch_scroll.setWidgetResizable(True)
        self.patch_scroll.setMinimumHeight(320)
        self.patch_scroll.setMaximumHeight(520)
        self.patch_scroll.setFrameShape(QFrame.NoFrame)

        self.patch_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOn
        )
        self.patch_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        self.patch_rows_container = QWidget()

        self.patch_rows_layout = QVBoxLayout(
            self.patch_rows_container
        )
        self.patch_rows_layout.setContentsMargins(
            2, 2, 2, 2
        )
        self.patch_rows_layout.setSpacing(4)

        self.patch_scroll.setWidget(
            self.patch_rows_container
        )

        patch_layout.addWidget(
            self.patch_scroll
        )

        # ----------------------------------------------------
        # PATCH ACCORDION
        # Keep the internal ID unchanged because other
        # functions use "patch_inspection".
        # ----------------------------------------------------
        patch_section = self._make_collapsible_section(
            "PAINTING PATCHES",
            "patch_inspection",
            patch_content,
        )

        patch_outer_layout.addWidget(
            patch_section
        )

        # Initially hidden until Stage 2.
        self.patch_inspection_panel.hide()

        layout.addWidget(
            self.patch_inspection_panel
        )

        # ----------------------------------------------------
        # MAIN ACTION BUTTON
        # ----------------------------------------------------
        layout.addStretch()

        self.start_analysis_button = QPushButton(
            "LOAD STL PRODUCT"
        )
        self.start_analysis_button.setObjectName(
            "PrimaryButton"
        )
        self.start_analysis_button.clicked.connect(
            self._handle_active_stage_action
        )

        layout.addWidget(
            self.start_analysis_button
        )

        # ----------------------------------------------------
        # OPERATION PROGRESS
        # This is a separate progress indicator for running
        # operations, not the patch-selection progress bar.
        # ----------------------------------------------------
        self.operation_panel = QFrame()
        self.operation_panel.setObjectName(
            "OperationProgressPanel"
        )

        progress_layout = QVBoxLayout(
            self.operation_panel
        )
        progress_layout.setContentsMargins(
            8, 8, 8, 8
        )
        progress_layout.setSpacing(6)

        self.operation_status_label = QLabel(
            "PREPARING OPERATION..."
        )
        self.operation_status_label.setStyleSheet(
            "color: #00E5FF; font-weight: bold;"
        )

        progress_layout.addWidget(
            self.operation_status_label
        )

        self.operation_progress = QProgressBar()
        self.operation_progress.setRange(0, 0)
        self.operation_progress.setTextVisible(False)
        self.operation_progress.setFixedHeight(9)

        progress_layout.addWidget(
            self.operation_progress
        )

        self.operation_panel.setStyleSheet("""
            QFrame#OperationProgressPanel {
                background-color: #0D1822;
                border: 1px solid #203746;
                border-radius: 5px;
            }
        """)

        self.operation_panel.hide()

        layout.addWidget(
            self.operation_panel
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------
        self.results_path_label = QLabel(
            "RESULT DIRECTORY: --"
        )
        self.results_path_label.setWordWrap(True)
        self.results_path_label.setObjectName("Muted")

        layout.addWidget(
            self.results_path_label
        )

        return panel


    # ========================================================
    # STAGE 2 PATCH LIST
    # ========================================================

    def _clear_patch_rows(self):
        self.patch_row_widgets.clear()

        while self.patch_rows_layout.count():
            item = self.patch_rows_layout.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

    def _get_stage1_patch_ids(self):
        if self.current_stage1_result is None:
            return []

        labels = np.asarray(
            self.current_stage1_result.get(
                "patch_labels",
                [],
            )
        )

        if labels.ndim != 1 or labels.size == 0:
            return []

        return sorted(
            int(value)
            for value in np.unique(labels)
            if int(value) >= 0
        )

    
    
    def _initialize_patch_inspection(self):
        """
        Default painting configuration:
        - Every patch is included for painting.
        - Every patch checkbox is checked.
        - Every patch decision is Paint.
        - Initial progress is 100%.
        """

        patch_ids = self._get_stage1_patch_ids()

        self.patch_inspection = {
            patch_id: {
                "inspected": True,
                "decision": DECISION_PAINT,
            }
            for patch_id in patch_ids
        }

        self.selected_patch_id = None

        self._build_patch_rows()
        self._refresh_patch_progress()



    
    def _build_patch_rows(self):
        """
        Build the Stage 2 patch inspection list.

        Widget values are initialized from patch_inspection.
        """

        self._clear_patch_rows()

        if not self.patch_inspection:
            self.patch_inspection_panel.hide()
            return

        self.patch_inspection_panel.show()

        for patch_id in sorted(self.patch_inspection):

            state = self.patch_inspection[patch_id]

            row = QFrame()
            row.setObjectName("PatchRow")

            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(3, 2, 3, 2)
            row_layout.setSpacing(4)

            # Patch selection button
            select_button = QPushButton(f"PATCH {patch_id}")
            select_button.setFixedWidth(82)
            select_button.setToolTip(
                f"Select Patch {patch_id} in the 3D viewer"
            )

            select_button.clicked.connect(
                lambda checked=False, pid=patch_id:
                self._select_patch_in_viewer(pid)
            )

            # Inspected checkbox
            inspected_checkbox = QCheckBox("Inspected")
            inspected_checkbox.setToolTip(
                "Mark this Patch as visually inspected."
            )
            inspected_checkbox.setChecked(
                bool(state.get("inspected", True))
            )

            # Decision dropdown
            decision_selector = QComboBox()

            decision_selector.addItem("Paint", DECISION_PAINT)
            decision_selector.addItem("No Paint", DECISION_NO_PAINT)
            decision_selector.addItem(
                "Requires Inspection",
                DECISION_REQUIRES_INSPECTION
            )

            decision_selector.setMinimumWidth(125)
            decision_selector.setToolTip(
                "Manual painting decision for this Patch."
            )

            current_decision = state.get(
                "decision",
                DECISION_PAINT
            )

            decision_index = decision_selector.findData(
                current_decision
            )

            if decision_index < 0:
                decision_index = decision_selector.findData(
                    DECISION_PAINT
                )

            decision_selector.setCurrentIndex(decision_index)

            # Connect signals after initializing widget values.
            inspected_checkbox.toggled.connect(
                lambda checked, pid=patch_id:
                self._on_patch_inspected_changed(pid, checked)
            )

            decision_selector.currentIndexChanged.connect(
                lambda index, pid=patch_id, combo=decision_selector:
                self._on_patch_decision_changed(
                    pid,
                    combo.currentData()
                )
            )

            row_layout.addWidget(select_button)
            row_layout.addWidget(inspected_checkbox)
            row_layout.addWidget(decision_selector, 1)

            self.patch_rows_layout.addWidget(row)

            self.patch_row_widgets[patch_id] = {
                "row": row,
                "select_button": select_button,
                "checkbox": inspected_checkbox,
                "decision": decision_selector,
            }

        self.patch_rows_layout.addStretch()


    def _select_patch_in_viewer(self, patch_id):
        if patch_id not in self.patch_inspection:
            return

        self.selected_patch_id = int(patch_id)

        try:
            self.viewer.select_surface_patch(
                self.selected_patch_id
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Patch Selection",
                "The 3D viewer could not select this Patch.\n\n"
                f"{exc}",
            )
            return

        for current_id, widgets in self.patch_row_widgets.items():
            selected = current_id == self.selected_patch_id

            widgets["select_button"].setStyleSheet(
                "background-color: #D6A500; color: #101820;"
                if selected
                else ""
            )

        self.status_bar.showMessage(
            f"SELECTED PATCH {patch_id}  |  "
            "INSPECT THE HIGHLIGHTED SURFACE"
        )

    def _on_patch_inspected_changed(self, patch_id, checked):
        if patch_id not in self.patch_inspection:
            return

        self.patch_inspection[patch_id]["inspected"] = bool(
            checked
        )

        self._on_patch_inspection_updated()

    def _on_patch_decision_changed(self, patch_id, decision):
        if patch_id not in self.patch_inspection:
            return

        if decision not in DECISION_TO_FACE_STATUS:
            return

        state = self.patch_inspection[patch_id]

        state["decision"] = decision

        # Paint or No Paint automatically completes inspection.
        if decision in (
            DECISION_PAINT,
            DECISION_NO_PAINT,
        ):
            state["inspected"] = True
        else:
            state["inspected"] = False

        # Update the checkbox without triggering another callback.
        widgets = self.patch_row_widgets.get(patch_id)

        if widgets is not None:
            checkbox = widgets["checkbox"]

            checkbox.blockSignals(True)
            checkbox.setChecked(state["inspected"])
            checkbox.blockSignals(False)

        self._on_patch_inspection_updated()

    def _on_patch_inspection_updated(self):
        """
        Handle changes to patch inspection checkboxes
        and painting decisions.
        """

        try:
            self._apply_manual_patch_decisions()

        except Exception as exc:
            traceback.print_exc()

            QMessageBox.critical(
                self,
                "Patch Decision Error",
                str(exc),
            )

            return

        self._refresh_patch_progress()

        self._persist_patch_inspection()

        all_inspected = self._all_patches_inspected()

        self.stage_completed[
            STAGE_SURFACE_IDENTIFICATION
        ] = all_inspected

        self._refresh_workflow_ui()

        if all_inspected:
            self.status_bar.showMessage(
                "PAINTING SELECTION COMPLETE  |  "
                "100% OF PATCHES INCLUDED  |  "
                "SPRAY TRAJECTORY AVAILABLE"
            )
        else:
            self.status_bar.showMessage(
                "PAINTING SELECTION INCOMPLETE"
            )

        if self.active_stage == STAGE_SURFACE_IDENTIFICATION:
            self._show_surface_identification_summary()

    def _all_patches_inspected(self):
        if not self.patch_inspection:
            return False

        return all(
            item["inspected"]
            for item in self.patch_inspection.values()
        )

    def _refresh_patch_progress(self):
        """Update the percentage of patches selected for painting."""

        total = len(self.patch_inspection)

        selected = sum(
            1
            for state in self.patch_inspection.values()
            if state.get("inspected", True)
        )

        percentage = (
            round(selected * 100 / total)
            if total > 0
            else 0
        )

        self.patch_progress_label.setText(
            f"PAINTABLE PATCHES: {selected} / {total} "
            f"({percentage}%)"
        )

        self.patch_progress_bar.setValue(percentage)

        all_selected = total > 0 and selected == total

        self.stage_completed[
            STAGE_SURFACE_IDENTIFICATION
        ] = all_selected

        self._refresh_stage2_action_visibility()

    
    
    def _refresh_stage2_action_visibility(self):
        """Control trajectory generation based on painting selection."""

        if not hasattr(self, "start_analysis_button"):
            return

        if not hasattr(self, "patch_inspection_status_label"):
            return

        if self.active_stage != STAGE_SURFACE_IDENTIFICATION:
            self.patch_inspection_status_label.hide()
            return

        if self.current_stage2_result is None:
            self.patch_inspection_status_label.hide()
            return

        total = len(self.patch_inspection)

        selected = sum(
            1
            for state in self.patch_inspection.values()
            if state.get("inspected", True)
        )

        all_selected = total > 0 and selected == total

        self.stage_completed[
            STAGE_SURFACE_IDENTIFICATION
        ] = all_selected

        self.start_analysis_button.setText("SPRAY TRAJECTORY")
        self.start_analysis_button.setVisible(True)

        if all_selected:
            self.patch_inspection_status_label.hide()
            self.start_analysis_button.setEnabled(
                not self._operation_running
                and self._stage_available(STAGE_SPRAY_TRAJECTORY)
            )
        else:
            self.patch_inspection_status_label.setText(
                "PAINTING SELECTION INCOMPLETE — "
                "select all patches to enable spray trajectory."
            )
            self.patch_inspection_status_label.show()
            self.start_analysis_button.setEnabled(False)

    # ========================================================
    # STAGE 2 MANUAL DECISION APPLICATION
    # ========================================================

    def _apply_manual_patch_decisions(self):
        """
        Rebuild the final Stage 2 face labels from the immutable
        automatic Stage 2 baseline plus currently inspected
        manual patch decisions.

        IMPORTANT:
        Never modify the automatic baseline in-place.

        This gives us reversible behavior:

            automatic
                ↓
            inspected + Paint
                ↓
            uncheck Inspected
                ↓
            automatic again
        """

        if self.current_stage2_result is None:
            return

        if self.current_stage1_result is None:
            return

        automatic_labels = (
            self._stage2_automatic_face_status_labels
        )

        if automatic_labels is None:
            # Defensive fallback for older/reloaded state.
            automatic_labels = np.asarray(
                self.current_stage2_result.get(
                    "face_status_labels",
                    [],
                )
            ).copy()

            self._stage2_automatic_face_status_labels = (
                automatic_labels.copy()
            )

        source_labels = np.asarray(
            self.current_stage1_result["patch_labels"]
        )

        automatic_labels = np.asarray(
            automatic_labels
        )

        if (
            source_labels.ndim != 1
            or automatic_labels.ndim != 1
            or len(source_labels) != len(automatic_labels)
        ):
            raise ValueError(
                "Stage 1 Patch Labels and automatic Stage 2 "
                "Face Labels have incompatible dimensions."
            )

        # Start every time from the untouched automatic result.
        face_labels = automatic_labels.copy()

        for patch_id, state in self.patch_inspection.items():

            # ------------------------------------------------
            # CRITICAL RULE:
            #
            # An unchecked patch has NOT been manually
            # reviewed. Therefore its automatic Stage 2
            # classification must remain unchanged.
            # ------------------------------------------------

            if not bool(state.get("inspected", False)):
                continue

            decision = state.get("decision")

            if decision not in DECISION_TO_FACE_STATUS:
                continue

            mask = source_labels == int(patch_id)

            if not np.any(mask):
                continue

            manual_status = DECISION_TO_FACE_STATUS[
                decision
            ]

            # Preserve faces that Stage 2 itself marked INVALID.
            #
            # Manual operator decisions should not magically turn
            # geometrically invalid faces into valid spray faces.
            invalid_mask = (
                automatic_labels == FACE_INVALID
            )

            apply_mask = mask & ~invalid_mask

            face_labels[apply_mask] = manual_status

        self.current_stage2_result[
            "face_status_labels"
        ] = face_labels

        report = self.current_stage2_result.setdefault(
            "report",
            {},
        )

        report["manual_patch_inspection"] = {
            str(patch_id): {
                "inspected": bool(state["inspected"]),
                "decision": state["decision"],
            }
            for patch_id, state in self.patch_inspection.items()
        }

        report["manual_inspection_complete"] = (
            self._all_patches_inspected()
        )

        self._update_stage2_manual_summary()

    def _update_stage2_manual_summary(self):
        if self.current_stage2_result is None:
            return

        labels = np.asarray(
            self.current_stage2_result["face_status_labels"]
        )

        report = self.current_stage2_result.setdefault(
            "report",
            {},
        )

        summary = report.setdefault("summary", {})

        paint_faces = int(
            np.count_nonzero(
                labels == FACE_PAINTABLE_CANDIDATE
            )
        )

        excluded_faces = int(
            np.count_nonzero(
                labels == FACE_EXCLUDED
            )
        )

        inspection_faces = int(
            np.count_nonzero(
                labels == FACE_REQUIRES_INSPECTION
            )
        )

        invalid_faces = int(
            np.count_nonzero(
                labels == FACE_INVALID
            )
        )

        summary["manual_paint_faces"] = paint_faces
        summary["manual_no_paint_faces"] = excluded_faces
        summary["manual_requires_inspection_faces"] = (
            inspection_faces
        )
        summary["invalid_faces"] = invalid_faces

        # ----------------------------------------------------
        # IMPORTANT:
        # Patch decision counts represent completed human
        # decisions only.
        #
        # Uninspected patches retain the default decision value
        # "requires_inspection", but they are NOT counted as a
        # completed manual decision.
        # ----------------------------------------------------

        summary["manual_paint_patches"] = sum(
            1
            for state in self.patch_inspection.values()
            if state["inspected"]
            and state["decision"] == DECISION_PAINT
        )

        summary["manual_no_paint_patches"] = sum(
            1
            for state in self.patch_inspection.values()
            if state["inspected"]
            and state["decision"] == DECISION_NO_PAINT
        )

        summary["manual_requires_inspection_patches"] = sum(
            1
            for state in self.patch_inspection.values()
            if state["inspected"]
            and state["decision"]
            == DECISION_REQUIRES_INSPECTION
        )

        summary["manual_uninspected_patches"] = sum(
            1
            for state in self.patch_inspection.values()
            if not state["inspected"]
        )

        summary["manual_inspected_patch_count"] = sum(
            1
            for state in self.patch_inspection.values()
            if state["inspected"]
        )

        summary["manual_total_patch_count"] = len(
            self.patch_inspection
        )

        summary["manual_inspection_complete"] = (
            self._all_patches_inspected()
        )

        # Calculate the actual paintable area from source mesh
        # faces using the source coordinate unit.
        if (
            self.current_mesh is not None
            and self._source_unit_ready()
        ):
            face_areas = np.asarray(
                self.current_mesh.area_faces,
                dtype=float,
            )

            if len(face_areas) == len(labels):
                area_factor = (
                    MM_PER_UNIT[self.coordinate_unit] ** 2
                )

                candidate_area_mm2 = float(
                    np.sum(
                        face_areas[
                            labels == FACE_PAINTABLE_CANDIDATE
                        ]
                    ) * area_factor
                )

                summary["total_candidate_area_mm2"] = (
                    candidate_area_mm2
                )

    def _persist_patch_inspection(self):
        if (
            self.current_stage2_result is None
            or self.current_stl_path is None
        ):
            return

        try:
            output_path = self.result_manager.result_path(
                self.current_stl_path,
                stage_number=2,
                filename="surface_patch_inspection.json",
            )

            output_path = Path(output_path)
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            payload = {
                "source_file": str(self.current_stl_path),
                "coordinate_unit": self.coordinate_unit,
                "patch_count": len(self.patch_inspection),
                "inspected_count": sum(
                    1
                    for item in self.patch_inspection.values()
                    if item["inspected"]
                ),
                "all_inspected": self._all_patches_inspected(),
                "patches": {
                    str(patch_id): {
                        "inspected": bool(state["inspected"]),
                        "decision": state["decision"],
                    }
                    for patch_id, state in self.patch_inspection.items()
                },
                "face_status_values": {
                    "excluded": FACE_EXCLUDED,
                    "paintable_candidate": FACE_PAINTABLE_CANDIDATE,
                    "requires_inspection": FACE_REQUIRES_INSPECTION,
                    "invalid": FACE_INVALID,
                },
            }

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    payload,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            self.patch_inspection_path = str(output_path)

            self.current_stage2_result[
                "manual_inspection_json_path"
            ] = str(output_path)

        except Exception as exc:
            traceback.print_exc()
            self.status_bar.showMessage(
                f"WARNING: PATCH INSPECTION SAVE FAILED: {exc}"
            )

    # ========================================================
    # UNIT STATE
    # ========================================================

    def _source_unit_ready(self):
        return (
            self.current_mesh is not None
            and self.coordinate_unit in MM_PER_UNIT
        )

    def _set_coordinate_unit_controls(self):
        has_product = self.current_mesh is not None
        unit_known = self._source_unit_ready()

        self.coordinate_unit_selector.setEnabled(
            has_product
            and not self._operation_running
            and self.source_unit_from_stl == "unknown"
        )

        self.unit_selector.setEnabled(
            has_product
            and unit_known
            and not self._operation_running
        )

    def _reset_downstream_results(self):
        self.current_stage1_result = None
        self.current_report = None
        self.current_stage2_result = None
        self.current_stage3_result = None

        self.current_stage3_parameters = None
        self.pending_stage3_parameters = None

        self.patch_inspection = {}
        self.patch_row_widgets = {}
        self.selected_patch_id = None

        # IMPORTANT:
        # The automatic Stage 2 baseline must never survive
        # loading a new product or changing coordinate units.
        self._stage2_automatic_face_status_labels = None
        self.patch_inspection_path = None

        if hasattr(self, "patch_inspection_panel"):
            self._clear_patch_rows()
            self.patch_inspection_panel.hide()

        self.stage_completed[STAGE_GEOMETRY_INSPECTION] = False
        self.stage_completed[STAGE_SURFACE_IDENTIFICATION] = False
        self.stage_completed[STAGE_SPRAY_TRAJECTORY] = False

        if self.current_mesh is not None:
            self.active_stage = STAGE_GEOMETRY_INSPECTION

    def _on_coordinate_unit_changed(self, index):
        if self.current_mesh is None or self._operation_running:
            return

        selected_unit = normalize_unit(
            self.coordinate_unit_selector.currentData()
        )

        if selected_unit == self.coordinate_unit:
            self._set_coordinate_unit_controls()
            self._refresh_workflow_ui()
            return

        if self.source_unit_from_stl != "unknown":
            self.coordinate_unit_selector.blockSignals(True)

            restore_index = self.coordinate_unit_selector.findData(
                self.source_unit_from_stl
            )

            if restore_index >= 0:
                self.coordinate_unit_selector.setCurrentIndex(
                    restore_index
                )

            self.coordinate_unit_selector.blockSignals(False)

            QMessageBox.information(
                self,
                "STL Unit Detected",
                "A coordinate unit was detected in the STL metadata. "
                "The detected unit is used automatically.",
            )
            return

        self.coordinate_unit = selected_unit
        self.manual_unit_selected = (
            selected_unit != "unknown"
        )

        self._reset_downstream_results()

        self._set_coordinate_unit_controls()
        self._update_geometry_metrics()
        self._update_viewer_display_unit()
        self._refresh_workflow_ui()

        self.analysis_summary.setText(
            "COORDINATE UNIT CONFIGURATION UPDATED\n\n"
            f"ACTIVE COORDINATE UNIT: {selected_unit.upper()}\n\n"
            "PREVIOUS ANALYSIS RESULTS WERE INVALIDATED.\n"
            "RUN GEOMETRY INSPECTION AGAIN."
        )

    def _on_unit_changed(self, unit):
        if self.current_mesh is None:
            return

        self._update_geometry_metrics()
        self._update_viewer_display_unit()

        self.status_bar.showMessage(
            f"DISPLAY UNIT: {unit.upper()}  |  "
            f"STL COORDINATE UNIT: {self.coordinate_unit.upper()}  |  "
            "SOURCE COORDINATES UNCHANGED"
        )

    def _get_display_unit(self):
        return normalize_unit(
            self.unit_selector.currentText()
        )

    def _update_viewer_display_unit(self):
        if self.current_mesh is None:
            return

        self.viewer.set_source_unit(self.coordinate_unit)
        self.viewer.set_display_unit(
            self._get_display_unit()
        )

    def _display_length_factor(self):
        if not self._source_unit_ready():
            return None

        display_unit = self._get_display_unit()

        return (
            MM_PER_UNIT[self.coordinate_unit]
            / MM_PER_UNIT[display_unit]
        )

    def _update_unit_labels(self):
        if not self._source_unit_ready():
            self.metric_title_labels["area"].setText(
                "SURFACE AREA (COORDINATE UNITS²)"
            )
            self.metric_title_labels["volume"].setText(
                "VOLUME (COORDINATE UNITS³)"
            )

            names = {
                "W": "WIDTH (X)",
                "D": "DEPTH (Y)",
                "H": "HEIGHT (Z)",
            }

            for axis, label in self.dimension_title_labels.items():
                label.setText(
                    f"{names[axis]} (COORDINATE UNITS)"
                )
            return

        unit = self._get_display_unit()

        self.metric_title_labels["area"].setText(
            f"SURFACE AREA ({unit}²)"
        )
        self.metric_title_labels["volume"].setText(
            f"VOLUME ({unit}³)"
        )

        self.dimension_title_labels["W"].setText(
            f"WIDTH (X) ({unit})"
        )
        self.dimension_title_labels["D"].setText(
            f"DEPTH (Y) ({unit})"
        )
        self.dimension_title_labels["H"].setText(
            f"HEIGHT (Z) ({unit})"
        )

    # ========================================================
    # STAGE 3 PARAMETER STATE
    # ========================================================

    def _get_current_spray_parameters(self):
        return {
            "standoff_mm": float(self.standoff_spin.value()),
            "sampling_spacing_mm": float(self.spacing_spin.value()),
            "spray_angle_deg": float(self.spray_angle_spin.value()),
        }

    def _stage3_parameters_match(self):
        if self.current_stage3_parameters is None:
            return False

        return (
            self._get_current_spray_parameters()
            == self.current_stage3_parameters
        )

    def _on_spray_parameter_changed(self, value=None):
        if self._operation_running:
            return

        if self.current_stage3_result is None:
            if self.active_stage == STAGE_SPRAY_TRAJECTORY:
                self._show_stage3_ready_summary()
            return

        parameters_match = self._stage3_parameters_match()

        self.stage_completed[STAGE_SPRAY_TRAJECTORY] = (
            parameters_match
        )

        if self.active_stage == STAGE_SPRAY_TRAJECTORY:
            if parameters_match:
                self._show_spray_trajectory_summary()
            else:
                self._show_stage3_outdated_summary()

        self._refresh_workflow_ui()

    # ========================================================
    # WORKFLOW STATE
    # ========================================================

    
    def _stage_available(self, stage_index):

        if stage_index == STAGE_PRODUCT_SETUP:
            return True

        if stage_index == STAGE_GEOMETRY_INSPECTION:
            return (
                self.current_mesh is not None
                and self._source_unit_ready()
            )

        if stage_index == STAGE_SURFACE_IDENTIFICATION:
            return (
                self.current_mesh is not None
                and self._source_unit_ready()
                and self.stage_completed[
                    STAGE_GEOMETRY_INSPECTION
                ]
                and self.current_stage1_result is not None
            )

        if stage_index == STAGE_SPRAY_TRAJECTORY:

            all_patches_inspected = (
                self.current_stage2_result is not None
                and bool(self.patch_inspection)
                and self._all_patches_inspected()
            )

            return (
                self.current_mesh is not None
                and self._source_unit_ready()
                and self.stage_completed[
                    STAGE_GEOMETRY_INSPECTION
                ]
                and self.current_stage1_result is not None
                and all_patches_inspected
            )

        return False


    def _select_workflow_stage(self, stage_index):
        if self._operation_running:
            return

        if not 0 <= stage_index < len(WORKFLOW_STAGES):
            return

        if not self._stage_available(stage_index):
            return

        self.active_stage = stage_index
        self._refresh_workflow_ui()

        if stage_index == STAGE_SURFACE_IDENTIFICATION:
            self._show_surface_identification_summary()

        elif stage_index == STAGE_GEOMETRY_INSPECTION:
            if self.current_report is not None:
                self._show_analysis_summary(self.current_report)

        elif stage_index == STAGE_SPRAY_TRAJECTORY:
            if self.current_stage3_result is None:
                self._show_stage3_ready_summary()
            elif self._stage3_parameters_match():
                self._show_spray_trajectory_summary()
                self._display_current_trajectory()
            else:
                self._show_stage3_outdated_summary()
                self._display_current_trajectory()

    
    
    def _refresh_workflow_ui(self):

        stage = WORKFLOW_STAGES[self.active_stage]

        self.current_stage_label.setText(stage["name"])
        self.workflow_description.setText(stage["description"])

        self._set_coordinate_unit_controls()

        # ----------------------------------------------------
        # SYNCHRONIZE STAGE 2 COMPLETION
        # ----------------------------------------------------

        all_patches_inspected = (
            self.current_stage2_result is not None
            and bool(self.patch_inspection)
            and self._all_patches_inspected()
        )

        self.stage_completed[
            STAGE_SURFACE_IDENTIFICATION
        ] = all_patches_inspected

        # ----------------------------------------------------
        # WORKFLOW NAVIGATION BUTTONS
        # ----------------------------------------------------

        for index, button in enumerate(self.workflow_buttons):

            available = self._stage_available(index)

            button.setEnabled(
                available and not self._operation_running
            )

            button.setChecked(index == self.active_stage)

            if index == self.active_stage:
                button.setObjectName("StageActive")

            elif self.stage_completed.get(index, False):
                button.setObjectName("StageComplete")

            elif available:
                button.setObjectName("StageAvailable")

            else:
                button.setObjectName("StageDisabled")

            button.style().unpolish(button)
            button.style().polish(button)

        # ----------------------------------------------------
        # COLLAPSE PATCH INSPECTION WHEN ENTERING STAGE 3
        # ----------------------------------------------------

        if self.active_stage == STAGE_SPRAY_TRAJECTORY:
            self._collapse_patch_inspection_section()

        # ----------------------------------------------------
        # MAIN ACTION BUTTON
        # ----------------------------------------------------

        if self.active_stage == STAGE_PRODUCT_SETUP:

            self.start_analysis_button.setText(
                "LOAD STL PRODUCT"
            )

            self.start_analysis_button.setEnabled(
                not self._operation_running
            )

        elif self.active_stage == STAGE_GEOMETRY_INSPECTION:

            if self.stage_completed[STAGE_GEOMETRY_INSPECTION]:

                self.start_analysis_button.setText(
                    "GEOMETRY INSPECTION COMPLETE"
                )

                self.start_analysis_button.setEnabled(False)

            else:

                self.start_analysis_button.setText(
                    "ANALYZE PRODUCT GEOMETRY"
                )

                self.start_analysis_button.setEnabled(
                    self.current_mesh is not None
                    and self._source_unit_ready()
                    and not self._operation_running
                )

        elif self.active_stage == STAGE_SURFACE_IDENTIFICATION:

            if self.current_stage2_result is None:

                self.start_analysis_button.setText(
                    "IDENTIFY PAINTABLE SURFACES"
                )

                self.start_analysis_button.setEnabled(
                    self._stage_available(
                        STAGE_SURFACE_IDENTIFICATION
                    )
                    and not self._operation_running
                )

            elif all_patches_inspected:

                self.start_analysis_button.setText(
                    "SPRAY TRAJECTORY"
                )

                self.start_analysis_button.setEnabled(
                    self._stage_available(
                        STAGE_SPRAY_TRAJECTORY
                    )
                    and not self._operation_running
                )

            else:

                self.start_analysis_button.setText(
                    "PATCH INSPECTION IN PROGRESS"
                )

                self.start_analysis_button.setEnabled(False)

        elif self.active_stage == STAGE_SPRAY_TRAJECTORY:

            if (
                self.stage_completed[STAGE_SPRAY_TRAJECTORY]
                and self._stage3_parameters_match()
            ):

                self.start_analysis_button.setText(
                    "SPRAY TRAJECTORY COMPLETE"
                )

                self.start_analysis_button.setEnabled(False)

            else:

                self.start_analysis_button.setText(
                    "GENERATE SPRAY TRAJECTORY"
                )

                self.start_analysis_button.setEnabled(
                    self._stage_available(
                        STAGE_SPRAY_TRAJECTORY
                    )
                    and not self._operation_running
                )

        else:

            self.start_analysis_button.setText(
                stage["action"]
            )

            self.start_analysis_button.setEnabled(False)

        # ----------------------------------------------------
        # WORKFLOW STATUS
        # ----------------------------------------------------

        if self._operation_running:

            self._set_workflow_status(
                "STATUS: OPERATION RUNNING",
                "#FFD740",
            )

        elif self.current_mesh is None:

            self._set_workflow_status(
                "STATUS: WAITING FOR PRODUCT",
                "#FFD740",
            )

        elif not self._source_unit_ready():

            self._set_workflow_status(
                "STATUS: COORDINATE UNIT REQUIRED",
                "#FFD740",
            )

        elif (
            self.active_stage == STAGE_SPRAY_TRAJECTORY
            and self.current_stage3_result is not None
            and not self._stage3_parameters_match()
        ):

            self._set_workflow_status(
                "STATUS: PARAMETERS CHANGED - REGENERATION REQUIRED",
                "#FFD740",
            )

        elif (
            self.active_stage == STAGE_SPRAY_TRAJECTORY
            and self.stage_completed[STAGE_SPRAY_TRAJECTORY]
            and self._stage3_parameters_match()
        ):

            self._set_workflow_status(
                "STATUS: TRAJECTORY GENERATED - ENGINEERING REVIEW",
                "#FFD740",
            )

        elif self.active_stage == STAGE_SPRAY_TRAJECTORY:

            self._set_workflow_status(
                "STATUS: READY FOR TRAJECTORY GENERATION",
                "#00E5FF",
            )

        elif (
            self.active_stage == STAGE_SURFACE_IDENTIFICATION
            and all_patches_inspected
        ):

            self._set_workflow_status(
                "STATUS: ALL PATCHES INSPECTED",
                "#00E676",
            )

        elif (
            self.active_stage == STAGE_SURFACE_IDENTIFICATION
            and self.current_stage2_result is not None
        ):

            self._set_workflow_status(
                "STATUS: PATCH INSPECTION IN PROGRESS",
                "#FFD740",
            )

        elif (
            self.active_stage == STAGE_GEOMETRY_INSPECTION
            and self.stage_completed[STAGE_GEOMETRY_INSPECTION]
        ):

            self._set_workflow_status(
                "STATUS: GEOMETRY INSPECTION COMPLETE",
                "#00E676",
            )

        elif self.active_stage == STAGE_SURFACE_IDENTIFICATION:

            self._set_workflow_status(
                "STATUS: READY FOR SURFACE IDENTIFICATION",
                "#00E5FF",
            )

        elif self.active_stage == STAGE_GEOMETRY_INSPECTION:

            self._set_workflow_status(
                "STATUS: READY FOR GEOMETRY INSPECTION",
                "#00E5FF",
            )

        else:

            self._set_workflow_status(
                "STATUS: PRODUCT READY",
                "#00E5FF",
            )

        # ----------------------------------------------------
        # FINAL UI SYNCHRONIZATION
        # ----------------------------------------------------

        self._refresh_pipeline()
        self._refresh_stage2_action_visibility()



    def _refresh_stage2_action_visibility(self):
        """
        Refresh the Stage 2 inspection message and action button.

        If Stage 2 inspection is incomplete:
            - Show PATCH INSPECTION IN PROGRESS
            - Hide the main action button

        If all patches are inspected:
            - Hide PATCH INSPECTION IN PROGRESS
            - Show SPRAY TRAJECTORY
        """

        if not hasattr(
            self,
            "patch_inspection_status_label",
        ):
            return

        if not hasattr(
            self,
            "start_analysis_button",
        ):
            return

        if (
            self.active_stage
            != STAGE_SURFACE_IDENTIFICATION
        ):
            return

        if self.current_stage2_result is None:
            self.patch_inspection_status_label.hide()

            self.start_analysis_button.setVisible(
                True
            )

            return

        all_inspected = self._all_patches_inspected()

        if all_inspected:

            self.patch_inspection_status_label.hide()

            self.start_analysis_button.setVisible(
                True
            )

            self.start_analysis_button.setText(
                "SPRAY TRAJECTORY"
            )

            self.start_analysis_button.setEnabled(
                not self._operation_running
                and self._stage_available(
                    STAGE_SPRAY_TRAJECTORY
                )
            )

        else:

            self.patch_inspection_status_label.setText(
                "PATCH INSPECTION IN PROGRESS"
            )

            self.patch_inspection_status_label.setStyleSheet(
                "color: #FFD740; font-weight: bold;"
            )

            self.patch_inspection_status_label.show()

            self.start_analysis_button.setVisible(
                False
            )
    def _set_workflow_status(self, text, color):
        self.workflow_status_label.setText(text)
        self.workflow_status_label.setStyleSheet(
            f"color: {color}; font-weight: bold;"
        )

    # ========================================================
    # PIPELINE STATUS
    # ========================================================

    def _refresh_pipeline(self):
        pipeline_states = [
            "COMPLETE" if self.current_mesh is not None else "ACTIVE"
        ]

        if self.stage_completed[STAGE_GEOMETRY_INSPECTION]:
            pipeline_states.append("COMPLETE")
        elif self.current_mesh is not None:
            pipeline_states.append(
                "ACTIVE" if self._source_unit_ready() else "WAITING"
            )
        else:
            pipeline_states.append("WAITING")

        if self.stage_completed[STAGE_SURFACE_IDENTIFICATION]:
            pipeline_states.append("COMPLETE")
        elif self.current_stage2_result is not None:
            pipeline_states.append("ACTIVE")
        elif self._stage_available(STAGE_SURFACE_IDENTIFICATION):
            pipeline_states.append("ACTIVE")
        else:
            pipeline_states.append("WAITING")

        if (
            self.stage_completed[STAGE_SPRAY_TRAJECTORY]
            and self._stage3_parameters_match()
        ):
            pipeline_states.append("COMPLETE")
        elif self._stage_available(STAGE_SPRAY_TRAJECTORY):
            pipeline_states.append("ACTIVE")
        else:
            pipeline_states.append("WAITING")

        pipeline_states.extend([
            "LOCKED",
            "LOCKED",
            "LOCKED",
        ])

        for label, state, item in zip(
            self.pipeline_labels,
            pipeline_states,
            PIPELINE_ITEMS,
        ):
            if state == "COMPLETE":
                label.setText("●  " + item + "  [COMPLETE]")
                label.setStyleSheet("color: #00E676;")
            elif state == "ACTIVE":
                label.setText("◉  " + item + "  [READY]")
                label.setStyleSheet("color: #00E5FF;")
            elif state == "WAITING":
                label.setText("○  " + item + "  [WAITING]")
                label.setStyleSheet("color: #FFD740;")
            else:
                label.setText("○  " + item + "  [NOT IMPLEMENTED]")
                label.setStyleSheet("color: #71818B;")

    # ========================================================
    # MAIN WORKFLOW ACTION
    # ========================================================

    def _handle_active_stage_action(self):
        """
        Handle the main workflow action button.

        Stage 2:
            Verify all patches are inspected.
            Switch to Stage 3.
            Collapse Patch Inspection.
            Start spray trajectory generation immediately.

        Stage 3:
            Generate or regenerate the spray trajectory.
        """

        if self._operation_running:
            return

        # ----------------------------------------------------
        # STAGE 1 - PRODUCT SETUP
        # ----------------------------------------------------

        if self.active_stage == STAGE_PRODUCT_SETUP:

            self.load_stl_dialog()
            return

        # ----------------------------------------------------
        # STAGE 2 - GEOMETRY INSPECTION
        # ----------------------------------------------------

        elif self.active_stage == STAGE_GEOMETRY_INSPECTION:

            self.run_geometry_analysis()
            return

        # ----------------------------------------------------
        # STAGE 3 - SURFACE IDENTIFICATION
        # ----------------------------------------------------

        elif self.active_stage == STAGE_SURFACE_IDENTIFICATION:

            # Stage 2 has not been executed yet.
            if self.current_stage2_result is None:

                self.run_surface_identification()
                return

            # Verify the actual inspection records.
            all_inspected = self._all_patches_inspected()

            if not all_inspected:

                self.stage_completed[
                    STAGE_SURFACE_IDENTIFICATION
                ] = False

                self._refresh_workflow_ui()

                QMessageBox.warning(
                    self,
                    "Patch Inspection Incomplete",
                    "All patches must be inspected before "
                    "generating the spray trajectory.",
                )

                return

            # Synchronize Stage 2 completion.
            self.stage_completed[
                STAGE_SURFACE_IDENTIFICATION
            ] = True

            # Switch to Stage 3.
            if not self._stage_available(
                STAGE_SPRAY_TRAJECTORY
            ):

                QMessageBox.warning(
                    self,
                    "Stage 3 Not Available",
                    "Spray trajectory generation is not available.",
                )

                return

            self._select_workflow_stage(
                STAGE_SPRAY_TRAJECTORY
            )

            # Collapse the Patch Inspection accordion.
            self._collapse_patch_inspection_section()

            # Start Stage 3 immediately.
            self.run_spray_trajectory()
            self._set_patch_inspection_readonly(True)
            return

        # ----------------------------------------------------
        # STAGE 4 - SPRAY TRAJECTORY
        # ----------------------------------------------------

        elif self.active_stage == STAGE_SPRAY_TRAJECTORY:

            self.run_spray_trajectory()
            return

        # ----------------------------------------------------
        # OTHER STAGES
        # ----------------------------------------------------

        elif self.active_stage in (
            STAGE_ROBOT_CONFIGURATION,
            STAGE_EXECUTION,
        ):

            QMessageBox.information(
                self,
                "Not Available",
                "This workflow stage is not implemented yet.",
            )

            return

    # ========================================================
    # OPERATION UI CONTROL
    # ========================================================

    def _start_operation_ui(self, operation_name):
        self._operation_running = True

        self.operation_status_label.setText(
            operation_name.upper() + "..."
        )
        self.operation_progress.setRange(0, 0)
        self.operation_panel.show()

        self.load_button.setEnabled(False)
        self.coordinate_unit_selector.setEnabled(False)
        self.unit_selector.setEnabled(False)

        self.standoff_spin.setEnabled(False)
        self.spacing_spin.setEnabled(False)
        self.spray_angle_spin.setEnabled(False)

        self.system_status.setText("● PROCESSING")
        self.system_status.setStyleSheet(
            "color: #FFD740; font-weight: bold;"
        )

        self.status_bar.showMessage(
            operation_name.upper() + " IN PROGRESS..."
        )

        self._refresh_workflow_ui()

    def _finish_operation_ui(self, success):
        self._operation_running = False
        self.operation_panel.hide()

        self.load_button.setEnabled(True)

        self.standoff_spin.setEnabled(True)
        self.spacing_spin.setEnabled(True)
        self.spray_angle_spin.setEnabled(False)

        self._set_coordinate_unit_controls()

        self.system_status.setText("● SYSTEM ONLINE")
        self.system_status.setStyleSheet(
            "color: #00E676; font-weight: bold;"
        )

        self._refresh_workflow_ui()

        if not success:
            self.status_bar.showMessage("OPERATION FAILED")

    # ========================================================
    # STL LOADING
    # ========================================================

    def load_stl_dialog(self):
        if self._operation_running:
            return

        initial_dir = self.project_root / "models"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select STL Product",
            str(initial_dir),
            "STL Files (*.stl *.STL)",
        )

        if file_path:
            self.load_part(file_path)

    def load_part(self, file_path):
        if self._operation_running:
            return

        try:
            mesh = load_stl(file_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "STL Loading Error",
                str(exc),
            )
            return

        self.current_mesh = mesh
        self.current_stl_path = Path(file_path).resolve()

        self._reset_downstream_results()

        self.stage_completed = {
            STAGE_PRODUCT_SETUP: True,
            STAGE_GEOMETRY_INSPECTION: False,
            STAGE_SURFACE_IDENTIFICATION: False,
            STAGE_SPRAY_TRAJECTORY: False,
            STAGE_ROBOT_CONFIGURATION: False,
            STAGE_EXECUTION: False,
        }

        self.source_unit_from_stl = get_mesh_source_unit(mesh)

        if self.source_unit_from_stl != "unknown":
            self.coordinate_unit = self.source_unit_from_stl
            self.manual_unit_selected = False
        else:
            self.coordinate_unit = "unknown"
            self.manual_unit_selected = False

        selector_index = self.coordinate_unit_selector.findData(
            self.coordinate_unit
        )

        if selector_index >= 0:
            self.coordinate_unit_selector.blockSignals(True)
            self.coordinate_unit_selector.setCurrentIndex(
                selector_index
            )
            self.coordinate_unit_selector.blockSignals(False)

        self.coordinate_unit_selector.setEnabled(
            self.source_unit_from_stl == "unknown"
        )

        self.active_stage = STAGE_GEOMETRY_INSPECTION

        self.viewer.show_product(mesh)

        self.part_name_label.setText(
            self.current_stl_path.name
        )

        self._set_coordinate_unit_controls()
        self._update_geometry_metrics()
        self._update_viewer_display_unit()

        part_dir = self.result_manager.part_dir(
            self.current_stl_path
        )

        self.results_path_label.setText(
            f"RESULT DIRECTORY:\n{part_dir}"
        )

        self.analysis_summary.setStyleSheet("")

        if self.coordinate_unit == "unknown":
            self.analysis_summary.setText(
                "PRODUCT LOADED\n"
                "STL MODEL READY\n\n"
                "COORDINATE UNIT: UNKNOWN\n\n"
                "SELECT THE ACTUAL STL COORDINATE UNIT "
                "BEFORE GEOMETRY INSPECTION."
            )
        else:
            self.analysis_summary.setText(
                "PRODUCT LOADED\n"
                "STL MODEL READY\n\n"
                f"COORDINATE UNIT: {self.coordinate_unit.upper()}\n\n"
                "NEXT OPERATION:\n"
                "GEOMETRY INSPECTION"
            )

        self.summary_section.setText("STAGE SUMMARY")
        self._refresh_workflow_ui()

        self.status_bar.showMessage(
            f"PRODUCT LOADED  |  "
            f"{self.current_stl_path.name}  |  "
            f"COORDINATE UNIT: {self.coordinate_unit.upper()}  |  "
            f"DISPLAY UNIT: {self.unit_selector.currentText()}"
        )

    # ========================================================
    # GEOMETRY DISPLAY
    # ========================================================

    
    def _update_geometry_metrics(self):
        if self.current_mesh is None:
            return

        mesh = self.current_mesh

        # Source mesh statistics remain unchanged.
        self.metric_labels["vertices"].setText(
            f"{len(mesh.vertices):,}"
        )
        self.metric_labels["triangles"].setText(
            f"{len(mesh.faces):,}"
        )

        # Get volume from the topology-reconstructed Stage 1 result.
        # Do not use mesh.is_volume on the original STL mesh.
        volume_coordinate_units_cubed = None

        if isinstance(self.current_report, dict):
            geometry = self.current_report.get(
                "geometry",
                {},
            )

            if isinstance(geometry, dict):
                raw_volume = geometry.get(
                    "volume_coordinate_units_cubed"
                )

                if raw_volume is not None:
                    try:
                        candidate_volume = float(raw_volume)

                        if np.isfinite(candidate_volume):
                            volume_coordinate_units_cubed = abs(
                                candidate_volume
                            )
                    except (TypeError, ValueError):
                        volume_coordinate_units_cubed = None

        # Coordinate unit is not configured yet.
        if not self._source_unit_ready():
            self.metric_labels["area"].setText(
                f"{mesh.area:,.3f}"
            )

            self.metric_labels["volume"].setText(
                "N/A"
            )

            for axis, value in zip(("W", "D", "H"), mesh.extents):
                self.dimension_labels[axis].setText(
                    f"{value:.3f}"
                )

            self._update_unit_labels()
            return

        display_unit = self._get_display_unit()

        length_factor = self._display_length_factor()

        area_factor = get_display_area_factor(
            self.coordinate_unit,
            display_unit,
        )

        volume_factor = get_display_volume_factor(
            self.coordinate_unit,
            display_unit,
        )

        dimensions = mesh.extents * length_factor
        display_area = mesh.area * area_factor

        # Convert the volume calculated by Stage 1.
        display_volume = None

        if (
            volume_coordinate_units_cubed is not None
            and volume_factor is not None
        ):
            display_volume = (
                volume_coordinate_units_cubed * volume_factor
            )

        self.metric_labels["area"].setText(
            f"{display_area:,.3f}"
        )

        self.metric_labels["volume"].setText(
            f"{display_volume:,.3f}"
            if display_volume is not None
            else "N/A"
        )

        for axis, value in zip(("W", "D", "H"), dimensions):
            self.dimension_labels[axis].setText(
                f"{value:.3f}"
            )

        self._update_unit_labels()


    # ========================================================
    # STAGE 1 - GEOMETRY ANALYSIS
    # ========================================================

    def run_geometry_analysis(self):
        if self.current_mesh is None or self._operation_running:
            return

        if not self._source_unit_ready():
            QMessageBox.warning(
                self,
                "Coordinate Unit Required",
                "The STL file does not contain a recognized "
                "coordinate unit.\n\n"
                "Select the actual coordinate unit before "
                "running Geometry Inspection.",
            )
            return

        if self.stage_completed[STAGE_GEOMETRY_INSPECTION]:
            return

        manual_override = (
            self.coordinate_unit
            if self.manual_unit_selected
            else None
        )

        self._start_operation_ui("GEOMETRY INSPECTION")

        self._worker = OperationWorker(
            execute_stage1,
            mesh=self.current_mesh,
            stl_path=self.current_stl_path,
            result_manager=self.result_manager,
            source_unit=manual_override,
        )

        self._worker.succeeded.connect(
            self._on_geometry_analysis_complete
        )
        self._worker.failed.connect(
            self._on_geometry_analysis_failed
        )
        self._worker.finished.connect(
            self._on_worker_thread_finished
        )

        self._worker.start()

    
    def _on_geometry_analysis_complete(self, result):
        try:
            report = result["report"]
            labels = result["patch_labels"]

            self.viewer.show_surface_patches(
                self.current_mesh,
                labels,
            )

            self._update_viewer_display_unit()

            screenshot_path = self.result_manager.result_path(
                self.current_stl_path,
                stage_number=1,
                filename="surface_patches.png",
            )

            try:
                self.viewer.save_screenshot(screenshot_path)
            except Exception as exc:
                print("Screenshot export failed:", exc)

            self.current_stage1_result = result
            self.current_report = report

            # Refresh geometry metrics using the Stage 1 report.
            self._update_geometry_metrics()

            self.stage_completed[
                STAGE_GEOMETRY_INSPECTION
            ] = True

            self.active_stage = STAGE_SURFACE_IDENTIFICATION

            self._show_analysis_summary(report)
            self._finish_operation_ui(True)

            self.status_bar.showMessage(
                "GEOMETRY INSPECTION COMPLETE | "
                f"UNIT: {self.coordinate_unit.upper()} | "
                f"RESULT: {result['json_path']}"
            )

        except Exception as exc:
            traceback.print_exc()
            self._finish_operation_ui(False)

            QMessageBox.critical(
                self,
                "Geometry Display Error",
                str(exc),
            )


    def _on_geometry_analysis_failed(self, error_message):
        self._finish_operation_ui(False)

        self.workflow_status_label.setText(
            "STATUS: GEOMETRY ANALYSIS FAILED"
        )

        QMessageBox.critical(
            self,
            "Geometry Inspection Error",
            error_message,
        )

    # ========================================================
    # STAGE 2 - SURFACE IDENTIFICATION
    # ========================================================

    def run_surface_identification(self):
        if self.current_mesh is None or self._operation_running:
            return

        if self.current_stage1_result is None:
            QMessageBox.warning(
                self,
                "Stage 1 Required",
                "Run Geometry Inspection before "
                "Surface Identification.",
            )
            return

        if self.current_stage2_result is not None:
            return

        self._start_operation_ui(
            "AUTOMATIC SURFACE CLASSIFICATION"
        )

        self._worker = OperationWorker(
            execute_stage2,
            mesh=self.current_mesh,
            stl_path=self.current_stl_path,
            result_manager=self.result_manager,
            stage1_result=self.current_stage1_result,
            coordinate_unit=self.coordinate_unit,
        )

        self._worker.succeeded.connect(
            self._on_surface_identification_complete
        )
        self._worker.failed.connect(
            self._on_surface_identification_failed
        )
        self._worker.finished.connect(
            self._on_worker_thread_finished
        )

        self._worker.start()

    def _on_surface_identification_complete(self, result):
        try:
            self.current_stage2_result = result

            # ------------------------------------------------
            # CRITICAL:
            #
            # Save an immutable copy of the automatic Stage 2
            # face classification BEFORE any manual decisions
            # are applied.
            # ------------------------------------------------

            automatic_labels = np.asarray(
                result.get("face_status_labels", [])
            )

            if automatic_labels.ndim != 1:
                raise ValueError(
                    "Stage 2 face_status_labels must be a 1D array."
                )

            if (
                self.current_mesh is not None
                and len(automatic_labels)
                != len(self.current_mesh.faces)
            ):
                raise ValueError(
                    "Stage 2 face_status_labels count does not "
                    "match the STL face count."
                )

            self._stage2_automatic_face_status_labels = (
                automatic_labels.copy()
            )

            # Create a row for every actual Stage 1 patch.
            self._initialize_patch_inspection()

            # Keep the patch-color view while the operator
            # inspects individual patches.
            self.viewer.show_surface_patches(
                self.current_mesh,
                self.current_stage1_result["patch_labels"],
            )

            self._update_viewer_display_unit()

            self.active_stage = STAGE_SURFACE_IDENTIFICATION
            self.stage_completed[
                STAGE_SURFACE_IDENTIFICATION
            ] = False

            self._set_patch_inspection_readonly(False)
            self._expand_patch_inspection_section()
            # This now rebuilds the result from the untouched
            # automatic baseline. Since no patches are inspected
            # yet, the automatic labels remain unchanged.
            self._apply_manual_patch_decisions()
            self._persist_patch_inspection()

            self._show_surface_identification_summary()
            self._finish_operation_ui(True)

            self.status_bar.showMessage(
                f"STAGE 2 AUTOMATIC CLASSIFICATION COMPLETE  |  "
                f"{len(self.patch_inspection)} PATCHES REQUIRE REVIEW"
            )

        except Exception as exc:
            traceback.print_exc()
            self._finish_operation_ui(False)

            QMessageBox.critical(
                self,
                "Surface Display Error",
                str(exc),
            )

    def _on_surface_identification_failed(self, error_message):
        self._finish_operation_ui(False)

        self.workflow_status_label.setText(
            "STATUS: SURFACE IDENTIFICATION FAILED"
        )

        QMessageBox.critical(
            self,
            "Surface Identification Error",
            error_message,
        )

    # ========================================================
    # STAGE 3 - SURFACE TYPE MAPPING
    # ========================================================

    def _build_face_surface_types(self):
        if self.current_stage1_result is None:
            return None

        patch_labels = np.asarray(
            self.current_stage1_result.get(
                "patch_labels",
                [],
            )
        )

        if (
            patch_labels.ndim != 1
            or len(patch_labels) != len(self.current_mesh.faces)
        ):
            return None

        report = self.current_stage1_result.get("report", {})

        segmentation = report.get(
            "surface_segmentation",
            {},
        )

        patches = segmentation.get("patches", [])

        if not patches:
            return None

        patch_type_map = {}

        for patch in patches:
            patch_id = patch.get("patch_id")

            classification = str(
                patch.get("classification", "unknown")
            ).lower()

            if "planar" in classification:
                patch_type_map[patch_id] = "planar"
            elif "curved" in classification:
                patch_type_map[patch_id] = "curved"
            else:
                patch_type_map[patch_id] = "unknown"

        face_types = np.full(
            len(patch_labels),
            "unknown",
            dtype="<U16",
        )

        for patch_id, classification in patch_type_map.items():
            try:
                numeric_patch_id = int(patch_id)
            except (TypeError, ValueError):
                continue

            face_types[
                patch_labels == numeric_patch_id
            ] = classification

        return face_types

    # ========================================================
    # STAGE 3 - SPRAY TRAJECTORY GENERATION
    # ========================================================

    def run_spray_trajectory(self):
        if self.current_mesh is None or self._operation_running:
            return

        if not self._source_unit_ready():
            QMessageBox.warning(
                self,
                "Coordinate Unit Required",
                "Select the actual STL coordinate unit first.",
            )
            return

        if self.current_stage2_result is None:
            QMessageBox.warning(
                self,
                "Stage 2 Required",
                "Run Surface Identification before "
                "generating a spray trajectory.",
            )
            return

        if not self._all_patches_inspected():
            QMessageBox.warning(
                self,
                "Patch Inspection Incomplete",
                "Every Patch must be marked Inspected "
                "before Stage 3 can run.",
            )
            return

        if (
            self.stage_completed[STAGE_SPRAY_TRAJECTORY]
            and self._stage3_parameters_match()
        ):
            return

        # Ensure the final manual decisions are reflected
        # in the face labels passed to Stage 3.
        self._apply_manual_patch_decisions()
        self._persist_patch_inspection()

        parameters = self._get_current_spray_parameters()
        self.pending_stage3_parameters = parameters.copy()

        self._start_operation_ui(
            "SPRAY TRAJECTORY GENERATION"
        )

        self._worker = OperationWorker(
            execute_stage3,
            mesh=self.current_mesh,
            stl_path=self.current_stl_path,
            result_manager=self.result_manager,
            stage2_result=self.current_stage2_result,
            stage1_report=self.current_report,
            coordinate_unit=self.coordinate_unit,
            standoff_mm=parameters["standoff_mm"],
            sampling_spacing_mm=parameters["sampling_spacing_mm"],
            spray_angle_deg=parameters["spray_angle_deg"],
        )

        self._worker.succeeded.connect(
            self._on_spray_trajectory_complete
        )
        self._worker.failed.connect(
            self._on_spray_trajectory_failed
        )
        self._worker.finished.connect(
            self._on_worker_thread_finished
        )

        self._worker.start()

    def _display_current_trajectory(self):
        if self.current_stage3_result is None:
            return

        trajectory = self.current_stage3_result["trajectory"]
        face_surface_types = self._build_face_surface_types()

        self.viewer.show_spray_trajectory(
            self.current_mesh,
            trajectory,
            face_surface_types=face_surface_types,
        )

        self._update_viewer_display_unit()

    def _on_spray_trajectory_complete(self, result):
        try:
            self.current_stage3_result = result

            self.current_stage3_parameters = (
                self.pending_stage3_parameters.copy()
                if self.pending_stage3_parameters is not None
                else self._get_current_spray_parameters()
            )

            self.pending_stage3_parameters = None

            self.stage_completed[
                STAGE_SPRAY_TRAJECTORY
            ] = True

            self.active_stage = STAGE_SPRAY_TRAJECTORY

            self._display_current_trajectory()

            screenshot_path = self.result_manager.result_path(
                self.current_stl_path,
                stage_number=3,
                filename="spray_trajectory.png",
            )

            try:
                self.viewer.save_screenshot(screenshot_path)
            except Exception as exc:
                print(
                    "Stage 3 screenshot export failed:",
                    exc,
                )

            self._show_spray_trajectory_summary()
            self._finish_operation_ui(True)

            self.status_bar.showMessage(
                "SPRAY TRAJECTORY GENERATED  |  "
                f"SAMPLES: {result['sample_count']:,}  |  "
                f"RESULT: {result['json_path']}"
            )

        except Exception as exc:
            traceback.print_exc()
            self._finish_operation_ui(False)

            QMessageBox.critical(
                self,
                "Trajectory Display Error",
                str(exc),
            )

    def _on_spray_trajectory_failed(self, error_message):
        self.pending_stage3_parameters = None
        self._finish_operation_ui(False)

        self.workflow_status_label.setText(
            "STATUS: SPRAY TRAJECTORY GENERATION FAILED"
        )
        self.workflow_status_label.setStyleSheet(
            "color: #FF5252; font-weight: bold;"
        )

        QMessageBox.critical(
            self,
            "Spray Trajectory Error",
            error_message,
        )

    # ========================================================
    # STAGE 3 SUMMARIES
    # ========================================================

    @staticmethod
    def _format_status(value):
        if value is None:
            return "NOT AVAILABLE"

        return str(value).replace("_", " ").upper()

    def _show_stage3_ready_summary(self):
        self.summary_section.setText("STAGE 3 - READY")
        self.analysis_summary.setStyleSheet("color: #D8E5ED;")

        self.analysis_summary.setText(
            "STAGE 2 COMPLETE\n\n"
            "ALL PATCHES HAVE BEEN INSPECTED.\n"
            "MANUAL PAINT DECISIONS ARE APPLIED.\n\n"
            "CURRENT PARAMETERS\n"
            f"STANDOFF: {self.standoff_spin.value():.1f} mm\n"
            f"SAMPLING: {self.spacing_spin.value():.1f} mm\n"
            f"SPRAY ANGLE: {self.spray_angle_spin.value():.1f}°\n\n"
            f"STL COORDINATE UNIT: {self.coordinate_unit.upper()}\n\n"
            "ACTION:\n"
            "GENERATE SPRAY TRAJECTORY"
        )

    def _show_stage3_outdated_summary(self):
        if self.current_stage3_result is None:
            self._show_stage3_ready_summary()
            return

        self.summary_section.setText(
            "STAGE 3 - TRAJECTORY OUTDATED"
        )
        self.analysis_summary.setStyleSheet("color: #FFD740;")

        previous = self.current_stage3_parameters or {}

        self.analysis_summary.setText(
            "PREVIOUS TRAJECTORY IS OUTDATED\n\n"
            "SPRAY PARAMETERS HAVE CHANGED.\n\n"
            "PREVIOUS PARAMETERS\n"
            f"STANDOFF: {previous.get('standoff_mm', 0.0):.1f} mm\n"
            f"SPACING: {previous.get('sampling_spacing_mm', 0.0):.1f} mm\n"
            f"SPRAY ANGLE: {previous.get('spray_angle_deg', 0.0):.1f}°\n\n"
            "CURRENT PARAMETERS\n"
            f"STANDOFF: {self.standoff_spin.value():.1f} mm\n"
            f"SPACING: {self.spacing_spin.value():.1f} mm\n"
            f"SPRAY ANGLE: {self.spray_angle_spin.value():.1f}°\n\n"
            "ACTION REQUIRED:\n"
            "GENERATE SPRAY TRAJECTORY AGAIN"
        )

    def _show_spray_trajectory_summary(self):
        if self.current_stage3_result is None:
            self._show_stage3_ready_summary()
            return

        report = self.current_stage3_result.get("report", {})
        trajectory = self.current_stage3_result.get("trajectory", {})

        analysis = report.get("analysis", {})
        normal_info = report.get("normal_orientation", {})
        metrics = report.get("path_metrics", {})

        overall_status = analysis.get(
            "overall_status",
            report.get("overall_status", "REVIEW"),
        )

        geometric_status = analysis.get(
            "geometric_generation_status",
            "NOT AVAILABLE",
        )

        orientation_status = analysis.get(
            "normal_orientation_status",
            normal_info.get("status", "NOT AVAILABLE"),
        )

        parameters = self.current_stage3_parameters or {}

        standoff = parameters.get(
            "standoff_mm",
            DEFAULT_STANDOFF_MM,
        )

        spacing = parameters.get(
            "sampling_spacing_mm",
            DEFAULT_SAMPLING_SPACING_MM,
        )

        spray_angle = parameters.get(
            "spray_angle_deg",
            DEFAULT_SPRAY_ANGLE_DEG,
        )

        sample_count = self.current_stage3_result.get(
            "sample_count",
            len(trajectory.get("surface_points", [])),
        )

        segment_ids = trajectory.get("segment_ids", [])

        segment_count = (
            len(set(np.asarray(segment_ids).tolist()))
            if len(segment_ids)
            else 0
        )

        path_length_mm = metrics.get(
            "total_path_length_mm",
            metrics.get("path_length_mm"),
        )

        if path_length_mm is None:
            source_length = metrics.get(
                "total_path_length_coordinate_units",
                metrics.get("path_length_coordinate_units"),
            )

            if (
                isinstance(source_length, (int, float))
                and self._source_unit_ready()
            ):
                path_length_mm = (
                    source_length
                    * MM_PER_UNIT[self.coordinate_unit]
                )

        skipped_normals = report.get(
            "skipped_invalid_normals",
            normal_info.get("skipped_invalid_normals", 0),
        )

        invalid_normal_faces = normal_info.get(
            "invalid_normal_faces",
            "N/A",
        )

        self.summary_section.setText(
            "STAGE 3 - ANALYSIS RESULTS"
        )

        summary_lines = [
            f"OVERALL: {self._format_status(overall_status)}",
            "",
            f"GEOMETRIC GENERATION: "
            f"{self._format_status(geometric_status)}",
            f"NORMAL ORIENTATION: "
            f"{self._format_status(orientation_status)}",
            "",
            f"SURFACE SAMPLES: {sample_count:,}",
            f"TRAJECTORY SEGMENTS: {segment_count:,}",
            "",
            f"STL COORDINATE UNIT: "
            f"{self.coordinate_unit.upper()}",
            f"STANDOFF: {standoff:.1f} mm",
            f"SAMPLING SPACING: {spacing:.1f} mm",
            f"SPRAY ANGLE: {spray_angle:.1f}°",
            "",
            f"INVALID-NORMAL FACES: {invalid_normal_faces}",
            f"SKIPPED CANDIDATE FACES: {skipped_normals}",
        ]

        if isinstance(path_length_mm, (int, float)):
            summary_lines.extend([
                "",
                "WITHIN-SEGMENT PATH LENGTH:",
                f"{path_length_mm:,.2f} mm",
                "(Inter-segment travel is not included.)",
            ])

        summary_lines.extend([
            "",
            "ENGINEERING REVIEW:",
            "ROBOT REACHABILITY: NOT CHECKED",
            "COLLISION: NOT CHECKED",
            "FIXTURE CLEARANCE: NOT CHECKED",
            "TCP CALIBRATION: NOT VERIFIED",
            "SPRAY COVERAGE: NOT VERIFIED",
        ])

        self.analysis_summary.setText(
            "\n".join(summary_lines)
        )

        self.analysis_summary.setStyleSheet(
            "color: #00E676;"
            if str(overall_status).upper() == "PASS"
            else "color: #FFD740;"
        )

        stage_directory = self.current_stage3_result.get(
            "stage_directory"
        )

        if stage_directory:
            self.results_path_label.setText(
                f"STAGE 3 RESULTS:\n{stage_directory}"
            )

    # ========================================================
    # STAGE 1 SUMMARY
    # ========================================================

    def _show_analysis_summary(self, report):
        mesh_data = report["mesh"]
        segmentation = report["surface_segmentation"]

        unit = report.get(
            "unit_configuration",
            {},
        ).get(
            "coordinate_unit",
            self.coordinate_unit,
        )

        self.summary_section.setText("STAGE 1 SUMMARY")
        self.analysis_summary.setStyleSheet("")

        self.analysis_summary.setText(
            f"COORDINATE UNIT    {str(unit).upper()}\n\n"
            f"SURFACE PATCHES   {segmentation['patch_count']}\n"
            f"PLANAR PATCHES    {segmentation['planar_patch_count']}\n"
            f"CURVED PATCHES    {segmentation['curved_patch_count']}\n"
            f"INVALID PATCHES   {segmentation['invalid_patch_count']}\n\n"
            f"CONNECTED PARTS   {mesh_data['connected_components']}\n"
            f"DEGENERATE FACES  {mesh_data['degenerate_faces']}\n"
            f"WATERTIGHT        {mesh_data['watertight']}\n"
            f"WINDING           {mesh_data['winding_consistent']}\n\n"
            "NEXT: SURFACE IDENTIFICATION"
        )

    # ========================================================
    # STAGE 2 SUMMARY
    # ========================================================

    def _show_surface_identification_summary(self):
        if self.current_stage2_result is None:
            if self.current_stage1_result is not None:
                self.summary_section.setText("STAGE 2 - READY")
                self.analysis_summary.setStyleSheet("")

                self.analysis_summary.setText(
                    "GEOMETRY INSPECTION COMPLETE\n\n"
                    "READY TO CLASSIFY SURFACE PATCHES\n\n"
                    "ACTION:\n"
                    "IDENTIFY PAINTABLE SURFACES"
                )
            return

        report = self.current_stage2_result.get(
            "report",
            {},
        )

        summary = report.get(
            "summary",
            {},
        )

        total = len(self.patch_inspection)

        inspected = sum(
            1
            for item in self.patch_inspection.values()
            if item["inspected"]
        )

        # ----------------------------------------------------
        # Only count actual completed manual decisions.
        # ----------------------------------------------------

        paint_count = sum(
            1
            for item in self.patch_inspection.values()
            if item["inspected"]
            and item["decision"] == DECISION_PAINT
        )

        no_paint_count = sum(
            1
            for item in self.patch_inspection.values()
            if item["inspected"]
            and item["decision"] == DECISION_NO_PAINT
        )

        requires_count = sum(
            1
            for item in self.patch_inspection.values()
            if item["inspected"]
            and item["decision"]
            == DECISION_REQUIRES_INSPECTION
        )

        uninspected_count = total - inspected

        candidate_area_mm2 = summary.get(
            "total_candidate_area_mm2"
        )

        area_text = (
            f"{candidate_area_mm2:,.2f} mm²"
            if isinstance(
                candidate_area_mm2,
                (int, float),
            )
            else "NOT AVAILABLE"
        )

        self.summary_section.setText(
            "STAGE 2 - MANUAL PATCH INSPECTION"
        )

        self.analysis_summary.setStyleSheet(
            "color: #00E676;"
            if inspected == total and total > 0
            else "color: #FFD740;"
        )

        self.analysis_summary.setText(
            f"PATCH INSPECTION: {inspected} / {total}\n\n"
            f"PAINT PATCHES: {paint_count}\n"
            f"NO PAINT PATCHES: {no_paint_count}\n"
            f"REQUIRES INSPECTION: {requires_count}\n"
            f"NOT YET INSPECTED: {uninspected_count}\n\n"
            f"PAINTABLE FACE AREA:\n{area_text}\n\n"
            "DECISION RULES:\n"
            "PAINT: INCLUDED IN STAGE 3\n"
            "NO PAINT: EXCLUDED FROM STAGE 3\n"
            "REQUIRES INSPECTION: EXCLUDED FROM STAGE 3\n\n"
            + (
                "STAGE 2 COMPLETE\n"
                "STAGE 3 IS NOW AVAILABLE."
                if inspected == total and total > 0
                else "INSPECT EVERY PATCH TO CONTINUE."
            )
        )

        if self.patch_inspection_path:
            self.results_path_label.setText(
                "PATCH INSPECTION RESULTS:\n"
                f"{self.patch_inspection_path}"
            )

    # ========================================================
    # WORKER CLEANUP
    # ========================================================

    def _on_worker_thread_finished(self):
        self._worker = None

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def closeEvent(self, event):
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.wait()

        self.viewer.close()
        event.accept()