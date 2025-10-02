# pip install nidaqmx matplotlib
import sys
import time
import threading
import queue
from collections import deque
from datetime import datetime
from pathlib import Path

from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from nidaqmx.errors import DaqError
from nidaqmx.system.storage import PersistedTask
from nidaqmx.constants import LoggingMode, LoggingOperation

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import colormaps as mpl_colormaps
from matplotlib import colors as mcolors

# ========= USER: set your saved DAQmx task name =========
TASK_NAME = "RT-1081 SLB"   # must match the task in NI-MAX exactly
# ========================================================

# ---- user-tunable display defaults ----
WINDOW_SECONDS = 60        # plot ~this much history
UPDATE_BLOCKS_PER_SEC = 1  # set to 5 for snappier ~200 ms updates if your rate allows

def _ensure_list(x):
    return x if isinstance(x, list) else [x]

def _channel_limits(channel):
    try:
        return float(channel.ai_min), float(channel.ai_max)
    except AttributeError:
        return float(channel["ai_min"]), float(channel["ai_max"])

def _apply_axis_limits(axis, channels, idxs):
    if axis is None or not idxs:
        return
    lows = []
    highs = []
    for idx in idxs:
        try:
            lo, hi = _channel_limits(channels[idx])
        except Exception:
            continue
        lows.append(lo)
        highs.append(hi)
    if not lows:
        return
    ymin = min(lows)
    ymax = max(highs)
    if ymin == ymax:
        span = abs(ymin) * 0.05
        if span == 0:
            span = 1.0
        ymin -= span
        ymax += span
    axis.set_ylim(ymin, ymax)

UNIT_ABBREVIATIONS: Dict[str, str] = {
    "VOLTS": "V",
    "V": "V",
    "AMPS": "A",
    "AMPERES": "A",
    "A": "A",
    "DEG C": "°C",
    "DEG F": "°F",
    "DEG K": "K",
    "DEGREES CELSIUS": "°C",
    "DEGREES FAHRENHEIT": "°F",
    "DEGREES KELVIN": "K",
    "KELVIN": "K",
    "CELSIUS": "°C",
    "FAHRENHEIT": "°F",
    "PSI": "psi",
    "PASCALS": "Pa",
    "PA": "Pa",
    "BAR": "bar",
    "METERS": "m",
    "METER": "m",
    "INCHES": "in",
    "INCH": "in",
    "MM": "mm",
    "MILLIMETERS": "mm",
    "RPM": "rpm",
    "PERCENT": "%",
    "NEWTONS": "N",
    "NEWTON_METERS": "N·m",
    "METERS PER SECOND": "m/s",
    "METERS PER SECOND SQUARED": "m/s²",
    "VOLTS PER VOLT": "V/V",
    "COULOMBS": "C",
    "OHMS": "Ω",
    "HZ": "Hz",
    "G": "g",
    "STRAIN": "strain",
}

UNIT_ALIASES: Dict[str, str] = {
    "POUNDS PER SQUARE INCH": "PSI",
    "POUNDS PER SQ INCH": "PSI",
    "POUND FORCE PER SQUARE INCH": "PSI",
    "POUND/SQ IN": "PSI",
    "POUND-FORCE/SQ IN": "PSI",
    "POUND FORCE/SQUARE INCH": "PSI",
    "POUND FORCE PER SQ IN": "PSI",
    "POUND FORCE PER SQUARE IN": "PSI",
    "POUNDS PER SQUARE IN": "PSI",
    "POUNDS PER SQUARE-INCH": "PSI",
    "METERS/SECOND": "METERS PER SECOND",
    "METERS/SECOND^2": "METERS PER SECOND SQUARED",
    "METERS PER SECOND^2": "METERS PER SECOND SQUARED",
}

PREFERRED_UNIT_ATTRS: List[str] = [
    "ai_pressure_units",
    "ai_force_units",
    "ai_torque_units",
    "ai_displacement_units",
    "ai_velocity_units",
    "ai_accel_units",
    "ai_strain_units",
    "ai_sound_pressure_units",
    "ai_voltage_units",
    "ai_current_units",
    "ai_resistance_units",
    "ai_units",
    "ai_temperature_units",
    "ai_temp_units",
]

MEASUREMENT_UNIT_ATTR: Dict[str, str] = {
    "PRESSURE": "ai_pressure_units",
    "FORCE": "ai_force_units",
    "TORQUE": "ai_torque_units",
    "STRAIN": "ai_strain_units",
    "ACCEL": "ai_accel_units",
    "VELOCITY": "ai_velocity_units",
    "DISPLACEMENT": "ai_displacement_units",
    "POSITION": "ai_displacement_units",
    "TEMPERATURE": "ai_temp_units",
    "VOLTAGE": "ai_voltage_units",
    "CURRENT": "ai_current_units",
    "FREQ": "ai_freq_units",
    "SOUND": "ai_sound_pressure_units",
    "RESISTANCE": "ai_resistance_units",
}

def _normalize_unit_name(raw: str) -> str:
    if not raw:
        return ""
    cleaned = str(raw).replace("_", " ").replace("-", " ")
    with_spaces: List[str] = []
    prev_is_lower = False
    for ch in cleaned:
        if prev_is_lower and ch.isupper():
            with_spaces.append(" ")
        with_spaces.append(ch)
        prev_is_lower = ch.islower()
    normalized = " ".join("".join(with_spaces).split())
    return normalized.upper()


def _extract_scaled_units(channel) -> str:
    try:
        scale = getattr(channel, 'ai_custom_scale')
    except Exception:
        return ''
    try:
        scaled_units = getattr(scale, 'scaled_units', '')
    except Exception:
        scaled_units = ''
    return str(scaled_units) if scaled_units else ''

def _measurement_hint(channel) -> str:
    meas = getattr(channel, 'ai_meas_type', None)
    if meas is None:
        return ''
    return str(meas).upper()

def _channel_unit_label(channel) -> str:
    """Extract and format the unit label for a channel.
    
    Tries multiple strategies in order:
    1. Custom scale units
    2. Measurement-specific unit attributes
    3. Generic preferred unit attributes
    """
    # Strategy 1: Try custom scale units first
    scaled_units = _extract_scaled_units(channel)
    if scaled_units:
        normalized = _normalize_unit_name(scaled_units)
        mapped_key = UNIT_ALIASES.get(normalized, normalized)
        label = UNIT_ABBREVIATIONS.get(mapped_key)
        if label:
            return label
        if mapped_key:
            return mapped_key.title()

    # Strategy 2: Try measurement-specific unit attributes
    meas_hint = _measurement_hint(channel)
    for key, attr in MEASUREMENT_UNIT_ATTR.items():
        if key in meas_hint:
            try:
                value = getattr(channel, attr)
            except AttributeError:
                value = None
            if value:
                raw = value if isinstance(value, str) else getattr(value, 'name', str(value))
                normalized = _normalize_unit_name(raw)
                mapped_key = UNIT_ALIASES.get(normalized, normalized)
                label = UNIT_ABBREVIATIONS.get(mapped_key)
                if label:
                    return label
                if mapped_key.startswith('DEG '):
                    suffix = mapped_key.split()[-1]
                    if suffix == 'C':
                        return '°C'
                    if suffix == 'F':
                        return '°F'
                    if suffix == 'K':
                        return 'K'
                if len(mapped_key) <= 4:
                    return mapped_key.capitalize()
                return mapped_key.title()
            break

    # Strategy 3: Try generic preferred unit attributes
    for attr in PREFERRED_UNIT_ATTRS:
        try:
            value = getattr(channel, attr)
        except AttributeError:
            continue
        if value in (None, ''):
            continue
        raw = value if isinstance(value, str) else getattr(value, 'name', str(value))
        if not raw:
            continue
        key_upper = str(raw).upper()
        if key_upper in {'FROM_CUSTOM_SCALE', 'FROM_TEDS'}:
            continue
        normalized = _normalize_unit_name(raw)
        mapped_key = UNIT_ALIASES.get(normalized, normalized)
        label = UNIT_ABBREVIATIONS.get(mapped_key)
        if label:
            return label
        if mapped_key.startswith('DEG '):
            suffix = mapped_key.split()[-1]
            if suffix == 'C':
                return '°C'
            if suffix == 'F':
                return '°F'
            if suffix == 'K':
                return 'K'
        if len(mapped_key) <= 4:
            return mapped_key.capitalize()
        return mapped_key.title()

    return ''

class AcquisitionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"NI-DAQmx Task Runner - {TASK_NAME}")
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.data_queue: queue.Queue = queue.Queue()
        self.default_rate = 5.0
        self.channels = self._load_channel_metadata()
        if not self.channels:
            messagebox.showerror("Task Error", "The task does not contain any AI channels.")
            self.root.destroy()
            return
        self.channel_names = [ch["name"] for ch in self.channels]
        self.axis_count_var = tk.StringVar(value="2")
        self.channel_axis_vars: Dict[int, tk.StringVar] = {}
        self.axis_colormaps: Dict[int, str] = {
            1: 'Blues',
            2: 'Oranges',
            3: 'Greens',
            4: 'Purples',
            5: 'Reds',
        }
        self.table: Optional[ttk.Treeview] = None
        self.table_rows: Dict[int, str] = {}
        self.latest_values: Dict[int, Optional[float]] = {}
        self.channel_axis_widgets: Dict[int, ttk.Combobox] = {}
        self.current_assignments: Dict[int, List[int]] = {}
        self.channel_axis_lookup: Dict[int, int] = {}
        self.lines: Dict[int, Any] = {}
        self.axes: List[Any] = []
        self.extra_axes: List[Any] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._poll_queue)

    def _load_channel_metadata(self) -> List[Dict[str, Any]]:
        try:
            with PersistedTask(TASK_NAME).load() as task:
                channels = []
                for ch in task.ai_channels:
                    unit_label = _channel_unit_label(ch)
                    channels.append({
                        "name": ch.name,
                        "physical": ch.physical_channel,
                        "ai_min": float(ch.ai_min),
                        "ai_max": float(ch.ai_max),
                        "unit_label": unit_label,
                    })
                try:
                    rate = float(task.timing.samp_clk_rate)
                    if rate > 0:
                        self.default_rate = rate
                except Exception:
                    pass
                return channels
        except DaqError as exc:
            messagebox.showerror(
                "Task Error",
                f"Failed to load saved task '{TASK_NAME}':\n{exc}"
            )
            sys.exit(1)

    def _build_ui(self) -> None:
        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main_pane, padding=10)
        main_pane.add(controls, weight=0)

        plot_holder = ttk.Frame(main_pane, padding=5)
        main_pane.add(plot_holder, weight=1)

        logging_frame = ttk.LabelFrame(controls, text="Logging")
        logging_frame.pack(fill=tk.X, pady=(0, 10))

        self.override_logging_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            logging_frame,
            text="Override TDMS logging file",
            variable=self.override_logging_var,
            command=self._toggle_logging_controls,
        ).pack(anchor="w")

        self.log_path_var = tk.StringVar()
        self.log_entry = ttk.Entry(logging_frame, textvariable=self.log_path_var, state="disabled", width=34)
        self.log_entry.pack(fill=tk.X, pady=3)
        self.browse_button = ttk.Button(logging_frame, text="Browse.", command=self._browse_log_path, state=tk.DISABLED)
        self.browse_button.pack(anchor="e")

        channels_frame = ttk.LabelFrame(controls, text="Channels")
        channels_frame.pack(fill=tk.BOTH, expand=True)

        axis_controls = ttk.Frame(channels_frame)
        axis_controls.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(axis_controls, text="Number of Y-axes:").pack(side=tk.LEFT)
        self.axis_count_combo = ttk.Combobox(
            axis_controls,
            state="readonly",
            width=5,
            values=("1", "2", "3", "4"),
            textvariable=self.axis_count_var,
        )
        self.axis_count_combo.pack(side=tk.LEFT, padx=(8, 0))
        self.axis_count_combo.bind("<<ComboboxSelected>>", self._on_axis_count_change)

        ttk.Label(channels_frame, text="Assign channels to axes:").pack(anchor="w")

        assignments_frame = ttk.Frame(channels_frame)
        assignments_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        for idx, ch in enumerate(self.channels):
            row = ttk.Frame(assignments_frame)
            row.pack(fill=tk.X, pady=1)
            row.columnconfigure(1, weight=1)
            ttk.Label(row, text=f"{ch['name']} ({ch['physical']})").grid(row=0, column=0, sticky="w")
            var = tk.StringVar(value="Axis 1")
            combo = ttk.Combobox(
                row,
                width=18,
                state="readonly",
                values=self._axis_choice_labels(int(self.axis_count_var.get())),
                textvariable=var,
            )
            combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
            self.channel_axis_vars[idx] = var
            self.channel_axis_widgets[idx] = combo

        self._on_axis_count_change()

        ttk.Separator(controls).pack(fill=tk.X, pady=8)

        buttons = ttk.Frame(controls)
        buttons.pack(fill=tk.X)

        self.start_button = ttk.Button(buttons, text="Start", command=self.start_acquisition)
        self.start_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop_acquisition, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(controls, textvariable=self.status_var).pack(anchor="w", pady=(6, 0))

        self.figure = Figure(figsize=(7, 4), dpi=100)
        self.ax1 = self.figure.add_subplot(111)
        self.ax1.set_xlabel("Time (s)")
        self.ax1.grid(True)
        self.axes = [self.ax1]
        self.figure.subplots_adjust(right=0.78)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_holder)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.LabelFrame(plot_holder, text="Live readings")
        table_frame.pack(fill=tk.X, pady=(6, 0))
        columns = ("axis", "value")
        self.table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="tree headings",
            height=min(8, len(self.channels)),
        )
        self.table.heading("#0", text="Channel")
        self.table.column("#0", anchor="w", width=220, stretch=True)
        self.table.heading("axis", text="Axis")
        self.table.column("axis", anchor="center", width=100, stretch=False)
        self.table.heading("value", text="Latest value")
        self.table.column("value", anchor="e", width=140, stretch=False)
        self.table.pack(fill=tk.X, expand=False)

    def _toggle_logging_controls(self) -> None:
        state = tk.NORMAL if self.override_logging_var.get() else tk.DISABLED
        self.log_entry.configure(state=state)
        self.browse_button.configure(state=state)

    def _browse_log_path(self) -> None:
        default_path = self._default_log_path()
        filename = filedialog.asksaveasfilename(
            title="Select TDMS log file",
            defaultextension=".tdms",
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            filetypes=[("TDMS Files", "*.tdms"), ("All Files", "*.*")],
        )
        if filename:
            self.log_path_var.set(filename)

    def _axis_choice_labels(self, count: int) -> List[str]:
        return ["None"] + [f"Axis {i}" for i in range(1, count + 1)]

    def _on_axis_count_change(self, event: Optional[object] = None) -> None:
        try:
            count = max(1, int(self.axis_count_var.get()))
        except (TypeError, ValueError):
            count = 1
            self.axis_count_var.set("1")
        choices = self._axis_choice_labels(count)
        for idx, combo in self.channel_axis_widgets.items():
            combo["values"] = choices
            current = self.channel_axis_vars[idx].get()
            if current not in choices:
                fallback = "Axis 1" if count >= 1 else "None"
                self.channel_axis_vars[idx].set(fallback)
        if self.axis_count_combo.get() != str(count):
            self.axis_count_combo.set(str(count))

    def _collect_axis_assignments(self) -> Dict[int, List[int]]:
        try:
            count = max(1, int(self.axis_count_var.get()))
        except (TypeError, ValueError):
            count = 1
            self.axis_count_var.set("1")
        assignments: Dict[int, List[int]] = {i: [] for i in range(1, count + 1)}
        for idx in range(len(self.channels)):
            selection = self.channel_axis_vars[idx].get()
            if not selection or selection == "None":
                continue
            try:
                axis_idx = int(selection.split()[-1])
            except (IndexError, ValueError):
                continue
            if axis_idx > count:
                continue
            assignments[axis_idx].append(idx)
        return assignments

    def _channel_label(self, idx: int) -> str:
        channel = self.channels[idx]
        name = channel.get('name', f'Channel {idx}')
        unit_label = (channel.get('unit_label') or '').strip()
        if unit_label:
            return f"{name} ({unit_label})"
        return name

    def _format_value(self, value: Optional[float]) -> str:
        if value is None:
            return "-"
        magnitude = abs(value)
        if magnitude >= 10000:
            formatted = f"{value:,.0f}"
        elif magnitude >= 100:
            formatted = f"{value:,.1f}"
        elif magnitude >= 1:
            formatted = f"{value:,.3f}"
        elif magnitude >= 1e-2:
            formatted = f"{value:.4f}"
        elif magnitude >= 1e-4:
            formatted = f"{value:.6f}"
        else:
            formatted = f"{value:.8f}"
        return formatted.rstrip("0").rstrip(".")

    def _refresh_live_table(self, axis_count: int, assignments: Dict[int, List[int]]) -> None:
        if self.table is None:
            return
        for item in self.table.get_children():
            self.table.delete(item)
        self.table_rows.clear()
        self.latest_values.clear()
        for axis_index in range(1, axis_count + 1):
            axis_name = f"Axis {axis_index}"
            for ch_idx in assignments.get(axis_index, []):
                label = self._channel_label(ch_idx)
                iid = f"ch{ch_idx}"
                self.table.insert("", "end", iid=iid, text=label, values=(axis_name, "-"))
                self.table_rows[ch_idx] = iid
                self.latest_values[ch_idx] = None

    def _update_table_value(self, ch_idx: int, value: Optional[float]) -> None:
        if self.table is None:
            return
        iid = self.table_rows.get(ch_idx)
        if iid is None:
            return
        self.table.set(iid, "value", self._format_value(value))

    def _axis_colors(self, axis_index: int, count: int) -> List[str]:
        cmap_name = self.axis_colormaps.get(axis_index, 'tab10')
        cmap = mpl_colormaps.get_cmap(cmap_name)
        if count <= 0:
            return []
        start_stop_map = {
            1: (0.45, 0.85),
            2: (0.45, 0.85),
            3: (0.45, 0.85),
            4: (0.45, 0.85),
        }
        start, stop = start_stop_map.get(axis_index, (0.25, 0.85))
        if count == 1:
            sample = (start + stop) / 2.0
            sample = min(max(sample, 0.0), 1.0)
            return [mcolors.to_hex(cmap(sample))]
        if count == 2:
            samples = [start, stop]
        else:
            step = (stop - start) / max(1, count - 1)
            samples = [start + step * i for i in range(count)]
        return [mcolors.to_hex(cmap(min(max(s, 0.0), 1.0))) for s in samples]

    def _configure_axis_style(self, axis: Any, color: str, axis_index: int) -> None:
        axis.yaxis.label.set_color(color)
        axis.tick_params(axis='y', colors=color)
        spine_key = 'left' if axis_index == 1 else 'right'
        if spine_key in axis.spines:
            axis.spines[spine_key].set_color(color)

    def _reset_axes(self, axis_count: int) -> None:
        for ax in self.extra_axes:
            ax.remove()
        self.extra_axes = []
        self.axes = [self.ax1]

        self.ax1.clear()
        self.ax1.set_xlabel("Time (s)")
        self.ax1.grid(True)

        for axis_index in range(2, axis_count + 1):
            new_axis = self.ax1.twinx()
            self._style_extra_axis(new_axis)
            if axis_index >= 3:
                offset = 1.0 + 0.20 * (axis_index - 2)
                new_axis.spines["right"].set_position(("axes", offset))
            new_axis.grid(False)
            self.extra_axes.append(new_axis)
            self.axes.append(new_axis)

        right_margin = max(0.9 - 0.11 * max(0, axis_count - 1), 0.65)
        self.figure.subplots_adjust(right=right_margin)

    def _style_extra_axis(self, axis: Any) -> None:
        axis.set_frame_on(True)
        axis.patch.set_visible(False)
        axis.yaxis.set_ticks_position("right")
        axis.yaxis.set_label_position("right")
        axis.spines["right"].set_visible(True)

    def _default_log_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path.cwd() / "logs" / f"{TASK_NAME}_{timestamp}.tdms"

    def _drain_queue(self) -> None:
        try:
            while True:
                self.data_queue.get_nowait()
        except queue.Empty:
            pass

    def start_acquisition(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return

        assignments = self._collect_axis_assignments()
        selected = sorted({idx for channels in assignments.values() for idx in channels})
        if not selected:
            messagebox.showwarning("Selection Required", "Assign at least one channel to a Y-axis.")
            return

        log_path: Optional[Path] = None
        if self.override_logging_var.get():
            path_text = self.log_path_var.get().strip()
            if not path_text:
                default_path = self._default_log_path()
                self.log_path_var.set(str(default_path))
                path_text = str(default_path)
            log_path = Path(path_text)
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                messagebox.showerror("Path Error", f"Unable to create log directory:\n{exc}")
                return

        axis_count = max(1, int(self.axis_count_var.get()))
        self._prepare_plot(axis_count, assignments)

        self.stop_event.clear()
        self._drain_queue()
        self.worker_thread = threading.Thread(
            target=self._acquisition_worker,
            args=(selected, log_path),
            daemon=True,
        )
        self.worker_thread.start()

        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set("Running...")

    def _prepare_plot(self, axis_count: int, assignments: Dict[int, List[int]]) -> None:
        self.current_assignments = {idx: list(assignments.get(idx, [])) for idx in range(1, axis_count + 1)}
        self.channel_axis_lookup = {}
        self.lines.clear()

        self._reset_axes(axis_count)

        self._refresh_live_table(axis_count, self.current_assignments)

        handles: List[Any] = []
        labels: List[str] = []
        for axis_index in range(1, axis_count + 1):
            axis = self.axes[axis_index - 1]
            channels = self.current_assignments.get(axis_index, [])
            palette = self._axis_colors(axis_index, len(channels))
            if palette:
                self._configure_axis_style(axis, palette[0], axis_index)
            for ch_idx, color in zip(channels, palette):
                label = self._channel_label(ch_idx)
                (line,) = axis.plot([], [], label=label, color=color)
                self.lines[ch_idx] = line
                self.channel_axis_lookup[ch_idx] = axis_index
                handles.append(line)
                labels.append(label)
                self.latest_values[ch_idx] = None
            _apply_axis_limits(axis, self.channels, channels)

        legend = self.ax1.get_legend()
        if handles:
            self.ax1.legend(handles, labels, loc="upper right")
        elif legend is not None:
            legend.remove()

        self.canvas.draw_idle()

    def stop_acquisition(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.status_var.set("Stopping...")
            self.stop_event.set()
            self.stop_button.configure(state=tk.DISABLED)

    def _acquisition_worker(self, selected: List[int], log_path: Optional[Path]) -> None:
        selected = sorted(selected)
        try:
            with PersistedTask(TASK_NAME).load() as task:
                rate = self._extract_rate(task)
                read_n = max(1, int(round(rate / max(1, UPDATE_BLOCKS_PER_SEC))))
                window_len = max(read_n, int(rate * WINDOW_SECONDS))

                tbuf = deque(maxlen=window_len)
                ybufs = {idx: deque(maxlen=window_len) for idx in selected}

                if log_path is not None:
                    try:
                        task.in_stream.configure_logging(
                            str(log_path),
                            logging_mode=LoggingMode.LOG_AND_READ,
                            group_name=log_path.stem,
                            operation=LoggingOperation.CREATE_OR_REPLACE,
                        )
                    except DaqError as exc:
                        self.data_queue.put({"type": "error", "message": f"Logging configuration failed: {exc}"})

                task.start()
                start_time = time.time()

                try:
                    while not self.stop_event.is_set():
                        data = task.read(number_of_samples_per_channel=read_n, timeout=5.0)
                        if len(task.ai_channels) == 1:
                            per_channel = [_ensure_list(data)]
                        else:
                            per_channel = [_ensure_list(ch_data) for ch_data in data]

                        if not per_channel or not per_channel[0]:
                            continue

                        block_len = len(per_channel[0])
                        now = time.time() - start_time
                        for k in range(block_len):
                            sample_time = now - (block_len - 1 - k) / rate
                            tbuf.append(sample_time)
                            for idx in selected:
                                ybufs[idx].append(per_channel[idx][k])

                        self.data_queue.put({
                            "type": "data",
                            "time": list(tbuf),
                            "series": {idx: list(ybufs[idx]) for idx in selected},
                        })
                finally:
                    task.stop()
        except DaqError as exc:
            self.data_queue.put({"type": "error", "message": str(exc)})
        except Exception as exc:
            self.data_queue.put({"type": "error", "message": f"Unexpected error: {exc}"})
        finally:
            self.data_queue.put({"type": "stopped"})

    def _extract_rate(self, task) -> float:
        try:
            rate = float(task.timing.samp_clk_rate)
            if rate > 0:
                return rate
        except Exception:
            pass
        return max(1.0, self.default_rate)

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.data_queue.get_nowait()
                kind = item.get("type")
                if kind == "data":
                    self._update_plot(item)
                elif kind == "error":
                    messagebox.showerror("Acquisition Error", item.get("message", "Unknown error"))
                    self.stop_event.set()
                elif kind == "stopped":
                    self._on_worker_stopped()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_queue)

    def _update_plot(self, payload: Dict[str, Any]) -> None:
        times = payload.get("time", [])
        if not times:
            return
        series = payload.get("series", {})  # type: Dict[int, List[float]]
        for idx, data in series.items():
            line = self.lines.get(idx)
            if line is not None:
                line.set_data(times, data)
                last_value = data[-1] if data else None
                self.latest_values[idx] = last_value
                self._update_table_value(idx, last_value)

        left = times[0]
        right = times[-1]
        if right - left < 1e-6:
            half = 0.5
            left -= half
            right += half
        for axis in self.axes:
            axis.set_xlim(left, right)

        self.canvas.draw_idle()
        self.status_var.set(f"Running. t = {right:.1f} s")

    def _on_worker_stopped(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.worker_thread = None
        self.stop_event.clear()
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        if self.status_var.get().startswith("Running") or self.status_var.get() == "Stopping...":
            self.status_var.set("Idle")

    def on_close(self) -> None:
        self.stop_acquisition()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

def main() -> None:
    root = tk.Tk()
    app = AcquisitionApp(root)
    app.run()

if __name__ == "__main__":
    main()







