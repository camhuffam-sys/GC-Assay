import csv
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


def clean_assay_data(df):
    """Keep valid well measurements before any assay metadata section."""
    required_columns = {"Well", "MeasA:Result"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "The CSV must contain these columns: "
            + ", ".join(sorted(required_columns))
            + f". Missing: {', '.join(sorted(missing_columns))}"
        )

    marker_rows = df.iloc[:, 0].astype(str).str.contains(
        "Basic assay information", case=False, na=False
    )
    if marker_rows.any():
        first_marker_position = np.flatnonzero(marker_rows.to_numpy())[0]
        df = df.iloc[:first_marker_position]

    cleaned_df = df.loc[:, ["Well", "MeasA:Result"]].copy()
    cleaned_df["Well"] = cleaned_df["Well"].astype(str).str.strip().str.upper()
    is_well = cleaned_df["Well"].str.fullmatch(r"[A-H](?:0?[1-9]|1[0-2])")
    cleaned_df = cleaned_df.loc[is_well].copy()
    if cleaned_df.empty:
        raise ValueError("No plate well measurements were found in the selected CSV.")

    well_parts = cleaned_df["Well"].str.extract(r"^([A-H])0?([1-9]|1[0-2])$")
    cleaned_df["Well"] = (
        well_parts[0] + well_parts[1].astype(int).astype(str).str.zfill(2)
    )
    cleaned_df["Measurement"] = pd.to_numeric(
        cleaned_df["MeasA:Result"],
        errors="coerce",
    )
    invalid_measurements = cleaned_df["Measurement"].isna()
    if invalid_measurements.any():
        invalid_wells = cleaned_df.loc[invalid_measurements, "Well"].unique().tolist()
        raise ValueError(
            f"Non-numeric measurements were found for wells: {invalid_wells}"
        )
    return cleaned_df[["Well", "Measurement"]]


def read_assay_csv(file_path):
    """Read an assay CSV by locating its Well/MeasA:Result header row."""
    file_path = Path(file_path)
    header_line = None
    data_end_line = None
    with file_path.open("r", encoding="ISO-8859-1", newline="") as csv_file:
        for line_number, row in enumerate(csv.reader(csv_file)):
            normalized_header = {column.strip() for column in row}
            if header_line is None and {"Well", "MeasA:Result"} <= normalized_header:
                header_line = line_number
            elif (
                header_line is not None
                and row
                and "Basic assay information" in row[0]
            ):
                data_end_line = line_number
                break

    if header_line is None:
        raise ValueError(
            "Could not find a CSV header containing 'Well' and 'MeasA:Result'."
        )

    read_options = {
        "sep": ",",
        "encoding": "ISO-8859-1",
        "skiprows": header_line,
    }
    if data_end_line is not None:
        read_options["nrows"] = data_end_line - header_line - 1
    raw_df = pd.read_csv(file_path, **read_options)
    raw_df.columns = raw_df.columns.astype(str).str.strip()
    return clean_assay_data(raw_df)


def reshape_assay_data(df, interval_minutes=15):
    """Return measurements grouped by well, with time in minutes as the index."""
    if interval_minutes <= 0:
        raise ValueError("The measurement interval must be greater than zero.")
    well_data = {
        well: group["Measurement"].reset_index(drop=True)
        for well, group in df.groupby("Well")
    }

    reshaped_df = pd.DataFrame(well_data)
    reshaped_df.index = [i * interval_minutes for i in range(len(reshaped_df))]
    return reshaped_df


def calculate_control_f50(reshaped_df, bac_conc_controls):
    """Calculate the mean control F50 for each bacteria and concentration.

    ``bac_conc_controls`` is nested as bacteria -> concentrations -> replicate wells.
    The result has the same bacteria/concentration nesting, with one mean F50 per group.
    """
    bac_conc_f50s = []
    for bac in bac_conc_controls:
        conc_f50s = []
        for conc in bac:
            f50s = []
            for replicate in conc:
                measurements = reshaped_df[replicate]
                start = measurements.iloc[0]
                f50s.append((measurements.max() - start) * 0.5 + start)
            conc_f50s.append(np.mean(f50s))
        bac_conc_f50s.append(conc_f50s)

    return bac_conc_f50s


def calculate_control_t50(reshaped_df, bac_cols, bac_conc_f50s):
    """Interpolate each well's time to its bacteria group's control F50.

    ``bac_cols`` is nested as bacteria -> wells, while ``bac_conc_f50s`` is
    nested as bacteria -> concentrations -> F50 values.
    Returns the same bacteria grouping, with (well, t50) pairs. A well that
    does not cross its threshold has a ``None`` t50.
    """
    bac_t50s = []
    for bacteria, wells in enumerate(bac_cols):
        f50 = bac_conc_f50s[bacteria][0]
        bacteria_t50s = []

        for well in wells:
            measurements = reshaped_df[well]
            above_positions = np.flatnonzero((measurements > f50).to_numpy())

            if len(above_positions) == 0 or above_positions[0] == 0:
                bacteria_t50s.append((well, None))
                continue

            after_pos = above_positions[0]
            before_pos = after_pos - 1
            t1 = reshaped_df.index[before_pos]
            t2 = reshaped_df.index[after_pos]
            f1 = measurements.iloc[before_pos]
            f2 = measurements.iloc[after_pos]

            if pd.isna(f1) or pd.isna(f2) or f2 == f1:
                t50 = None
            else:
                t50 = t1 + ((f50 - f1) / (f2 - f1)) * (t2 - t1)

            bacteria_t50s.append((well, t50))

        bac_t50s.append(bacteria_t50s)

    return bac_t50s


def calculate_delta_t50s(bac_t50s, bac_conc_controls):
    """Return each well's T50 relative to its bacteria group's mean control T50.

    ``bac_t50s`` is nested as bacteria -> (well, t50) pairs.
    ``bac_conc_controls`` is nested as bacteria -> concentrations -> control wells;
    the first concentration's wells are used as controls.
    Wells without a T50 crossing are returned with a ``None`` delta.
    """
    if len(bac_t50s) != len(bac_conc_controls):
        raise ValueError("T50 results and control wells must have the same bacteria groups.")

    bac_delta_t50s = []
    for bacteria, bacteria_t50s in enumerate(bac_t50s):
        t50_by_well = dict(bacteria_t50s)
        control_groups = bac_conc_controls[bacteria]
        if not control_groups:
            raise ValueError(f"No control wells provided for bacteria group {bacteria}.")

        missing_controls = [
            well for well in control_groups[0] if well not in t50_by_well
        ]
        if missing_controls:
            raise ValueError(
                f"Control wells missing T50 results for bacteria group {bacteria}: "
                f"{missing_controls}"
            )

        control_t50s = [
            t50_by_well[well]
            for well in control_groups[0]
            if t50_by_well[well] is not None
            and not pd.isna(t50_by_well[well])
        ]
        if not control_t50s:
            raise ValueError(
                f"No control wells crossed the F50 threshold for bacteria group {bacteria}."
            )

        mean_control_t50 = np.mean(control_t50s)
        bacteria_delta_t50s = [
            (
                well,
                None if t50 is None or pd.isna(t50) else t50 - mean_control_t50,
            )
            for well, t50 in bacteria_t50s
        ]
        bac_delta_t50s.append(bacteria_delta_t50s)

    return bac_delta_t50s


def delta_t50s_to_plate_dataframe(bac_delta_t50s):
    """Combine bacteria groups' delta-T50 values into one 8x12 plate DataFrame."""
    all_well_results = [
        well_result
        for bacteria_delta_t50s in bac_delta_t50s
        for well_result in bacteria_delta_t50s
    ]
    well_df = pd.DataFrame(
        all_well_results,
        columns=["Well", "Delta_T50"],
    )
    well_df["Row"] = well_df["Well"].str[0]
    well_df["Column"] = pd.to_numeric(
        well_df["Well"].str[1:],
        errors="raise",
    )
    plate = well_df.pivot(
        index="Row",
        columns="Column",
        values="Delta_T50",
    )
    plate = plate.reindex(index=list("ABCDEFGH"), columns=range(1, 13))
    plate.columns.name = None
    plate.index.name = None
    return plate


def analyze_assay(file_path, bacteria_groups, interval_minutes=15):
    """Run F50/T50 normalization for user-defined bacteria control and test wells."""
    if not bacteria_groups:
        raise ValueError("Add at least one bacteria group.")

    names = [group["name"].strip() for group in bacteria_groups]
    if any(not name for name in names):
        raise ValueError("Every bacteria group must have a name.")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Bacteria group names must be unique.")

    all_wells = []
    controls_by_group = []
    wells_by_group = []
    for group in bacteria_groups:
        controls = group["controls"]
        tests = group["tests"]
        if not controls:
            raise ValueError(f"{group['name']}: enter at least one control well.")
        if not tests:
            raise ValueError(f"{group['name']}: enter at least one test well.")
        group_wells = controls + tests
        if len(set(group_wells)) != len(group_wells):
            raise ValueError(
                f"{group['name']}: a well cannot be listed as both a control and a test "
                "or repeated."
            )
        all_wells.extend(group_wells)
        controls_by_group.append([controls])
        wells_by_group.append(group_wells)

    if len(set(all_wells)) != len(all_wells):
        raise ValueError(
            "Each plate well can only belong to one bacteria group and one role."
        )

    cleaned_df = read_assay_csv(file_path)
    available_wells = set(cleaned_df["Well"].unique())
    missing_wells = sorted(set(all_wells) - available_wells)
    if missing_wells:
        raise ValueError(
            "These selected wells were not found in the CSV: "
            + ", ".join(missing_wells)
        )

    reshaped_df = reshape_assay_data(cleaned_df, interval_minutes)
    control_f50s = calculate_control_f50(reshaped_df, controls_by_group)
    t50s = calculate_control_t50(reshaped_df, wells_by_group, control_f50s)
    delta_t50s = calculate_delta_t50s(t50s, controls_by_group)
    plate_df = delta_t50s_to_plate_dataframe(delta_t50s)
    return plate_df


def save_delta_t50_heatmap(
    plate_df,
    bacteria_labels,
    png_path,
    csv_path,
    title="Delta T50 Heatmap",
    show=True,
    bacteria_wells=None,
):
    """Display and save a plate heatmap and CSV with optional group outlines."""
    if plate_df.shape != (8, 12):
        raise ValueError("plate_df must have 8 rows (A-H) and 12 columns (1-12).")
    if not bacteria_labels:
        raise ValueError("Provide at least one bacteria label.")
    if bacteria_wells is not None and len(bacteria_wells) != len(bacteria_labels):
        raise ValueError("Provide well locations for each bacteria label.")

    png_path = Path(png_path)
    csv_path = Path(csv_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plate_df.to_csv(csv_path, index_label="Row")

    colormap = LinearSegmentedColormap.from_list(
        "white_yellow_red",
        ["white", "yellow", "red"],
    ).with_extremes(bad="black")

    fig = plt.figure(figsize=(11, 4.5), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[12, 4, 0.8])
    ax = fig.add_subplot(grid[0, 0])
    label_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    colorbar_ax = fig.add_subplot(grid[0, 2])
    image = ax.imshow(plate_df.to_numpy(dtype=float), aspect="auto", cmap=colormap)
    fig.colorbar(image, cax=colorbar_ax, label="Delta T50 (minutes)")
    ax.set_xticks(np.arange(plate_df.shape[1]), labels=plate_df.columns)
    ax.set_yticks(np.arange(plate_df.shape[0]), labels=plate_df.index)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title)

    if bacteria_wells is not None:
        colors = plt.get_cmap("tab10")
        for group_index, wells in enumerate(bacteria_wells):
            color = colors(group_index % colors.N)
            for well in wells:
                row_label = well[0].upper()
                column_number = int(well[1:])
                if row_label not in plate_df.index or column_number not in plate_df.columns:
                    raise ValueError(f"Invalid well location: {well}")
                ax.add_patch(
                    Rectangle(
                        (column_number - 1.5, plate_df.index.get_loc(row_label) - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.5,
                    )
                )
            label_ax.plot([], [], color=color, linewidth=3, label=bacteria_labels[group_index])
        label_ax.legend(
            title="Bacteria groups",
            loc="center left",
            frameon=False,
        )
    else:
        label_ax.axis("off")
        label_ax.text(
            0,
            0.5,
            "\n".join(bacteria_labels),
            transform=label_ax.transAxes,
            va="center",
            wrap=True,
        )
    fig.savefig(png_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return png_path, csv_path


def parse_well_list(text):
    """Parse comma-, semicolon-, space-, or newline-separated plate well IDs."""
    wells = []
    for value in re.split(r"[,;\s]+", text.strip()):
        if not value:
            continue
        match = re.fullmatch(r"([A-H])0?([1-9]|1[0-2])", value.upper())
        if not match:
            raise ValueError(f"Invalid well location '{value}'. Use A01 through H12.")
        wells.append(f"{match.group(1)}{int(match.group(2)):02d}")
    if len(set(wells)) != len(wells):
        raise ValueError("A well was entered more than once in the same list.")
    return wells


class AssayAnalysisApp:
    """Tkinter interface for configuring and analyzing an assay plate."""

    def __init__(self, root):
        self.root = root
        self.root.title("Resazurin Assay T50 Analysis")
        self.root.geometry("1100x760")
        self.root.minsize(850, 600)
        self.group_fields = []

        main = ttk.Frame(root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(5, weight=1)

        ttk.Label(main, text="Assay CSV:").grid(row=0, column=0, sticky="w", pady=4)
        self.csv_path = tk.StringVar()
        ttk.Entry(main, textvariable=self.csv_path).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(main, text="Browse...", command=self._choose_csv).grid(
            row=0, column=2, sticky="ew"
        )

        ttk.Label(main, text="Output folder:").grid(row=1, column=0, sticky="w", pady=4)
        self.output_folder = tk.StringVar()
        ttk.Entry(main, textvariable=self.output_folder).grid(
            row=1, column=1, sticky="ew", padx=6
        )
        ttk.Button(main, text="Browse...", command=self._choose_output_folder).grid(
            row=1, column=2, sticky="ew"
        )

        settings = ttk.Frame(main)
        settings.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        ttk.Label(settings, text="Measurement interval (minutes):").pack(side="left")
        self.interval = tk.StringVar(value="15")
        ttk.Entry(settings, textvariable=self.interval, width=8).pack(
            side="left", padx=(6, 18)
        )
        ttk.Label(settings, text="Output file name:").pack(side="left")
        self.output_prefix = tk.StringVar(value="delta_t50")
        ttk.Entry(settings, textvariable=self.output_prefix, width=28).pack(
            side="left", padx=6
        )

        group_controls = ttk.Frame(main)
        group_controls.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        ttk.Label(group_controls, text="Number of bacteria groups:").pack(side="left")
        self.group_count = tk.StringVar(value="2")
        ttk.Spinbox(
            group_controls,
            from_=1,
            to=96,
            textvariable=self.group_count,
            width=5,
        ).pack(side="left", padx=6)
        ttk.Button(
            group_controls,
            text="Set groups",
            command=self._resize_groups,
        ).pack(side="left")
        ttk.Label(
            main,
            text=(
                "CSV must include Well and MeasA:Result columns. Enter well IDs "
                "separated by commas, spaces, semicolons, or new lines."
            ),
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 4))

        groups_container = ttk.Frame(main)
        groups_container.grid(row=5, column=0, columnspan=3, sticky="nsew")
        groups_container.columnconfigure(0, weight=1)
        groups_container.rowconfigure(0, weight=1)
        self.groups_canvas = tk.Canvas(groups_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            groups_container,
            orient="vertical",
            command=self.groups_canvas.yview,
        )
        self.groups_canvas.configure(yscrollcommand=scrollbar.set)
        self.groups_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.groups_frame = ttk.Frame(self.groups_canvas)
        self.groups_window = self.groups_canvas.create_window(
            (0, 0),
            window=self.groups_frame,
            anchor="nw",
        )
        self.groups_frame.bind(
            "<Configure>",
            lambda event: self.groups_canvas.configure(
                scrollregion=(0, 0, event.width, event.height)
            ),
        )
        self.groups_canvas.bind(
            "<Configure>",
            lambda event: self.groups_canvas.itemconfigure(
                self.groups_window,
                width=event.width,
            ),
        )

        actions = ttk.Frame(main)
        actions.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(12, 4))
        ttk.Button(actions, text="Analyze and save", command=self._analyze).pack(
            side="left"
        )
        self.status = tk.StringVar(value="Choose a CSV and configure the bacteria groups.")
        ttk.Label(actions, textvariable=self.status, wraplength=800).pack(
            side="left",
            padx=12,
        )
        self._resize_groups()

    def _choose_csv(self):
        selected = filedialog.askopenfilename(
            title="Select assay CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            path = Path(selected)
            self.csv_path.set(str(path))
            self.output_folder.set(str(path.parent))
            self.output_prefix.set(path.stem + "_delta_t50")

    def _choose_output_folder(self):
        selected = filedialog.askdirectory(title="Select output folder")
        if selected:
            self.output_folder.set(selected)

    def _resize_groups(self):
        try:
            count = int(self.group_count.get())
        except ValueError:
            messagebox.showerror("Invalid group count", "Enter a whole number from 1 to 96.")
            return
        if not 1 <= count <= 96:
            messagebox.showerror("Invalid group count", "Enter a number from 1 to 96.")
            return

        saved = [
            {
                "name": fields["name"].get(),
                "controls": self._field_value(fields["controls"]),
                "tests": self._field_value(fields["tests"]),
            }
            for fields in self.group_fields
        ]
        for child in self.groups_frame.winfo_children():
            child.destroy()
        self.group_fields = []

        for group_index in range(count):
            values = (
                saved[group_index]
                if group_index < len(saved)
                else {
                    "name": f"Bacteria {group_index + 1}",
                    "controls": "",
                    "tests": "",
                }
            )
            group_frame = ttk.LabelFrame(
                self.groups_frame,
                text=f"Bacteria group {group_index + 1}",
                padding=8,
            )
            group_frame.pack(fill="x", expand=True, padx=4, pady=4)
            group_frame.columnconfigure(0, weight=1)
            group_frame.columnconfigure(1, weight=2)
            group_frame.columnconfigure(2, weight=2)
            ttk.Label(group_frame, text="Bacteria name").grid(
                row=0, column=0, sticky="w"
            )
            ttk.Label(group_frame, text="Control wells").grid(
                row=0, column=1, sticky="w", padx=6
            )
            ttk.Label(group_frame, text="Test wells").grid(
                row=0, column=2, sticky="w", padx=6
            )

            name_var = tk.StringVar(value=values["name"])
            ttk.Entry(group_frame, textvariable=name_var).grid(
                row=1, column=0, sticky="ew"
            )
            controls_text = tk.Text(group_frame, height=3, wrap="word")
            controls_text.grid(
                row=1, column=1, sticky="ew", padx=6
            )
            controls_text.insert("1.0", values["controls"])
            tests_text = tk.Text(group_frame, height=3, wrap="word")
            tests_text.grid(
                row=1, column=2, sticky="ew", padx=6
            )
            tests_text.insert("1.0", values["tests"])
            self.group_fields.append(
                {
                    "name": name_var,
                    "controls": controls_text,
                    "tests": tests_text,
                }
            )

    @staticmethod
    def _field_value(field):
        if isinstance(field, tk.Text):
            return field.get("1.0", "end-1c")
        return field.get()

    def _analyze(self):
        try:
            csv_path_text = self.csv_path.get().strip()
            output_folder_text = self.output_folder.get().strip()
            if not csv_path_text:
                raise ValueError("Select an assay CSV file.")
            if not output_folder_text:
                raise ValueError("Select an output folder.")
            csv_path = Path(csv_path_text)
            output_folder = Path(output_folder_text)
            prefix = self.output_prefix.get().strip()
            interval = float(self.interval.get())
            if not csv_path.is_file():
                raise ValueError("Select an existing assay CSV file.")
            if not output_folder.is_dir():
                raise ValueError("Select an existing output folder.")
            if not prefix or not re.fullmatch(r"[\w .()-]+", prefix):
                raise ValueError(
                    "Output file name can contain letters, numbers, spaces, dots, "
                    "underscores, parentheses, and hyphens."
                )
            if not np.isfinite(interval) or interval <= 0:
                raise ValueError("Measurement interval must be a positive number.")

            groups = []
            for fields in self.group_fields:
                name = fields["name"].get().strip()
                groups.append(
                    {
                        "name": name,
                        "controls": parse_well_list(
                            self._field_value(fields["controls"])
                        ),
                        "tests": parse_well_list(
                            self._field_value(fields["tests"])
                        ),
                    }
                )

            self.status.set("Analyzing assay data...")
            self.root.update_idletasks()
            plate_df = analyze_assay(csv_path, groups, interval)
            csv_output = output_folder / f"{prefix}_plate.csv"
            png_output = output_folder / f"{prefix}_heatmap.png"
            existing_outputs = [
                path for path in (csv_output, png_output) if path.exists()
            ]
            if existing_outputs and not messagebox.askyesno(
                "Overwrite files?",
                "These output files already exist:\n"
                + "\n".join(str(path) for path in existing_outputs)
                + "\n\nReplace them?",
            ):
                self.status.set("Analysis canceled.")
                return
            save_delta_t50_heatmap(
                plate_df,
                bacteria_labels=[group["name"] for group in groups],
                png_path=png_output,
                csv_path=csv_output,
                title=f"Delta T50 Heatmap - {csv_path.stem}",
                show=False,
                bacteria_wells=[
                    group["controls"] + group["tests"] for group in groups
                ],
            )
            self.status.set(f"Saved outputs to {output_folder}")
            messagebox.showinfo(
                "Analysis complete",
                f"Plate CSV:\n{csv_output}\n\nHeatmap PNG:\n{png_output}",
            )
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            self.status.set("Analysis did not complete.")
            messagebox.showerror("Analysis failed", str(error))


def main():
    root = tk.Tk()
    AssayAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()