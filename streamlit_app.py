from __future__ import annotations

import hashlib
import tempfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import main as assay


ROWS = "ABCDEFGH"
WELLS = [f"{row}{column:02d}" for row in ROWS for column in range(1, 13)]


@st.cache_data(show_spinner=False)
def load_uploaded_assay(file_contents: bytes) -> pd.DataFrame:
    """Parse an uploaded instrument CSV using the core assay reader."""
    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "uploaded_assay.csv"
        csv_path.write_bytes(file_contents)
        return assay.read_assay_csv(csv_path)


def build_bacteria_groups(names: list[str], assignments: dict) -> list[dict]:
    """Convert per-well assignments to the core analysis input format."""
    groups = [
        {"name": name.strip(), "controls": [], "tests": []}
        for name in names
    ]
    for well, assignment in assignments.items():
        group_index = assignment["group"]
        role = assignment["role"]
        if 0 <= group_index < len(groups) and role in {"controls", "tests"}:
            groups[group_index][role].append(well)
    for group in groups:
        group["controls"].sort()
        group["tests"].sort()
    return groups


def normalize_group_index(value: object, group_count: int) -> int:
    """Return a valid bacteria-group index for a widget's persisted state."""
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < group_count
    ):
        return value
    return 0


def summarize_replicates(
    reshaped_df: pd.DataFrame,
    wells: list[str],
    error_mode: str = "SD",
    baseline_subtract: bool = False,
) -> pd.DataFrame:
    """Calculate replicate mean, sample count, and SD/SEM for each time point."""
    if not wells:
        raise ValueError("Select at least one replicate well.")
    if error_mode not in {"SD", "SEM"}:
        raise ValueError("Error mode must be 'SD' or 'SEM'.")
    missing_wells = sorted(set(wells) - set(reshaped_df.columns))
    if missing_wells:
        raise ValueError(f"Wells not found in the data: {', '.join(missing_wells)}")

    replicates = reshaped_df.loc[:, wells].astype(float)
    if baseline_subtract:
        replicates = replicates.subtract(replicates.iloc[0], axis="columns")

    count = replicates.count(axis=1)
    standard_deviation = replicates.std(axis=1, ddof=1)
    error = standard_deviation
    if error_mode == "SEM":
        error = standard_deviation / np.sqrt(count)
    return pd.DataFrame(
        {
            "Mean": replicates.mean(axis=1),
            "Error": error,
            "N": count,
        },
        index=reshaped_df.index,
    )


def make_plate_heatmap(
    plate_df: pd.DataFrame,
    bacteria_groups: list[dict],
) -> plt.Figure:
    """Render a heatmap with wells outlined by bacteria group."""
    cmap = LinearSegmentedColormap.from_list(
        "white_yellow_red",
        ["white", "yellow", "red"],
    ).with_extremes(bad="black")

    fig = plt.figure(figsize=(11, 4.5), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[12, 4, 0.8])
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    colorbar_ax = fig.add_subplot(grid[0, 2])
    image = ax.imshow(plate_df.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    fig.colorbar(image, cax=colorbar_ax, label="Delta T50 (minutes)")
    ax.set_xticks(np.arange(12), labels=range(1, 13))
    ax.set_yticks(np.arange(8), labels=list(ROWS))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Delta T50 plate heatmap")

    colors = plt.get_cmap("tab10")
    for group_index, group in enumerate(bacteria_groups):
        color = colors(group_index % colors.N)
        for well in group["controls"] + group["tests"]:
            row = ROWS.index(well[0])
            column = int(well[1:]) - 1
            ax.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.5,
                )
            )
        legend_ax.plot([], [], color=color, linewidth=3, label=group["name"])
    legend_ax.legend(title="Bacteria groups", loc="center left", frameon=False)
    return fig


def figure_png_bytes(fig: plt.Figure) -> bytes:
    """Serialize a Matplotlib figure as PNG bytes."""
    image = BytesIO()
    fig.savefig(image, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    image.seek(0)
    return image.getvalue()


def _well_label(well: str, assignment: dict | None) -> str:
    if assignment is None:
        return well
    icon = "🔵" if assignment["role"] == "controls" else "🟠"
    return f"{icon} {well}"


def _render_well_picker(names: list[str], reshaped_df: pd.DataFrame | None) -> None:
    st.subheader("Assign wells on the plate")
    if not names:
        st.info("Add at least one bacteria group to assign wells.")
        return

    group_options = list(range(len(names)))
    selected_group = normalize_group_index(
        st.session_state.get("well_picker_group_index", 0),
        len(names),
    )
    if st.session_state.get("well_picker_group_index", 0) != selected_group:
        st.session_state["well_picker_group_index"] = selected_group
    active_group = st.selectbox(
        "Assign wells to bacteria",
        options=group_options,
        format_func=lambda index: f"{index + 1}. {names[index] or 'Unnamed group'}",
        key="well_picker_group_index",
    )
    active_role = st.radio(
        "Well role",
        options=["controls", "tests"],
        format_func=lambda role: "Control" if role == "controls" else "Test",
        horizontal=True,
        key="active_role",
    )
    st.caption(
        "Click an available well to assign it to the selected group and role. "
        "Click it again with the same selection to clear it. 🔵 Control  🟠 Test"
    )
    assignments = st.session_state.setdefault("well_assignments", {})
    available = set(reshaped_df.columns) if reshaped_df is not None else set()

    for row in ROWS:
        columns = st.columns(12, gap="small")
        for column_number, column_container in enumerate(columns, start=1):
            well = f"{row}{column_number:02d}"
            assignment = assignments.get(well)
            label = _well_label(well, assignment)
            with column_container:
                if st.button(
                    label,
                    key=f"plate_{well}",
                    help=(
                        f"{well}: {names[assignment['group']]} "
                        f"{assignment['role']}"
                        if assignment is not None
                        else f"Assign {well} as a {active_role[:-1]} "
                        f"for {names[active_group]}"
                    ),
                    disabled=(
                        reshaped_df is not None
                        and well not in available
                        and assignment is None
                    ),
                    use_container_width=True,
                ):
                    if assignment is not None and well not in available:
                        assignments.pop(well)
                    elif (
                        assignment is not None
                        and assignment["group"] == active_group
                        and assignment["role"] == active_role
                    ):
                        assignments.pop(well)
                    else:
                        assignments[well] = {
                            "group": active_group,
                            "role": active_role,
                        }
                    st.session_state.pop("analysis_result", None)
                    st.rerun()

    if assignments:
        st.caption(
            "Assigned wells: "
            + ", ".join(
                f"{well} ({names[value['group']]}, {value['role'][:-1]})"
                for well, value in sorted(assignments.items())
                if value["group"] < len(names)
            )
        )
    if st.button("Clear all well assignments", key="clear_assignments"):
        assignments.clear()
        st.session_state.pop("analysis_result", None)
        st.rerun()


def _render_analysis_tab(
    cleaned_df: pd.DataFrame | None,
    reshaped_df: pd.DataFrame | None,
    uploaded_name: str | None,
    interval_minutes: float,
    file_hash: str | None,
) -> None:
    st.header("Plate analysis")
    count = st.number_input(
        "Number of bacteria groups",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
        key="bacteria_group_count",
    )
    names = [
        st.text_input(
            f"Bacteria group {index + 1} name",
            value=f"Bacteria {index + 1}",
            key=f"bacteria_name_{index}",
        ).strip()
        for index in range(int(count))
    ]

    assignments = st.session_state.setdefault("well_assignments", {})
    for well, assignment in list(assignments.items()):
        if assignment["group"] >= count:
            del assignments[well]

    _render_well_picker(names, reshaped_df)

    if cleaned_df is None or uploaded_name is None:
        st.info("Upload an assay CSV in the sidebar to enable plate selection and analysis.")
        return

    groups = build_bacteria_groups(names, assignments)
    fingerprint = (
        file_hash,
        uploaded_name,
        interval_minutes,
        tuple(
            (group["name"], tuple(group["controls"]), tuple(group["tests"]))
            for group in groups
        ),
    )
    if st.button(
        "Calculate delta T50 plate",
        type="primary",
        disabled=cleaned_df is None,
    ):
        try:
            result = assay.analyze_assay_data(
                cleaned_df,
                groups,
                interval_minutes,
            )
            st.session_state["analysis_result"] = {
                "plate": result,
                "groups": groups,
                "uploaded_name": uploaded_name,
                "fingerprint": fingerprint,
            }
        except (ValueError, KeyError) as error:
            st.error(str(error))

    result = st.session_state.get("analysis_result")
    if result is None:
        return
    if result["fingerprint"] != fingerprint:
        st.info("Your file or well assignments changed. Recalculate to refresh results.")
        return

    fig = make_plate_heatmap(result["plate"], result["groups"])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.dataframe(result["plate"], use_container_width=True)

    csv_bytes = result["plate"].to_csv(index_label="Row").encode("utf-8")
    png_bytes = figure_png_bytes(make_plate_heatmap(result["plate"], result["groups"]))
    base_name = Path(result["uploaded_name"]).stem
    st.download_button(
        "Download delta T50 CSV",
        data=csv_bytes,
        file_name=f"{base_name}_delta_t50_plate.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download heatmap PNG",
        data=png_bytes,
        file_name=f"{base_name}_delta_t50_heatmap.png",
        mime="image/png",
    )
    with st.expander("Delta T50 values"):
        details = [
            {"Bacteria": group["name"], "Role": role[:-1], "Well": well,
             "Delta T50 (minutes)": result["plate"].loc[well[0], int(well[1:])]}
            for group in result["groups"]
            for role in ("controls", "tests")
            for well in group[role]
        ]
        st.dataframe(pd.DataFrame(details), use_container_width=True)


def _render_curves_tab(reshaped_df: pd.DataFrame | None) -> None:
    st.header("Fluorescence curves")
    if reshaped_df is None:
        st.info("Upload an assay CSV in the sidebar to view fluorescence curves.")
        return

    well_options = sorted(reshaped_df.columns)
    mode = st.radio(
        "Plot mode",
        ["Compare individual wells", "Replicate mean with error bars"],
        horizontal=True,
        key="curve_mode",
    )
    baseline_subtract = st.checkbox(
        "Subtract each well's initial fluorescence",
        value=False,
        help="Plots change from each well's starting signal instead of raw fluorescence.",
    )

    if mode == "Compare individual wells":
        selected_wells = st.multiselect(
            "Wells to compare",
            options=well_options,
            default=well_options[: min(4, len(well_options))],
            key="compare_wells",
        )
        if not selected_wells:
            st.info("Select one or more wells to plot.")
            return
        curve_data = reshaped_df[selected_wells].astype(float)
        if baseline_subtract:
            curve_data = curve_data.subtract(curve_data.iloc[0], axis="columns")

        fig, ax = plt.subplots(figsize=(10, 5))
        for well in selected_wells:
            ax.plot(curve_data.index, curve_data[well], label=well)
        ax.set(
            xlabel="Time (minutes)",
            ylabel=(
                "Change in fluorescence from start"
                if baseline_subtract
                else "Fluorescence"
            ),
            title="Fluorescence by well",
        )
        ax.grid(True, alpha=0.25)
        ax.legend(title="Well", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.download_button(
            "Download plotted fluorescence data",
            data=curve_data.rename_axis("Time (minutes)").to_csv().encode("utf-8"),
            file_name="fluorescence_curves.csv",
            mime="text/csv",
        )
    else:
        selected_wells = st.multiselect(
            "Replicate wells to combine",
            options=well_options,
            default=[],
            key="replicate_wells",
        )
        error_mode = st.radio(
            "Error bars",
            ["SD", "SEM"],
            horizontal=True,
            help="SD shows replicate spread; SEM shows uncertainty in the mean.",
            key="replicate_error_mode",
        )
        if not selected_wells:
            st.info("Select replicate wells to calculate their mean and error bars.")
            return
        summary = summarize_replicates(
            reshaped_df,
            selected_wells,
            error_mode=error_mode,
            baseline_subtract=baseline_subtract,
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(
            summary.index,
            summary["Mean"],
            yerr=summary["Error"],
            fmt="o-",
            capsize=3,
            label=f"Mean ± {error_mode}",
        )
        ax.set(
            xlabel="Time (minutes)",
            ylabel=(
                "Change in fluorescence from start"
                if baseline_subtract
                else "Fluorescence"
            ),
            title=f"Replicate mean ± {error_mode} (n={len(selected_wells)})",
        )
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.dataframe(summary, use_container_width=True)
        st.download_button(
            "Download replicate summary",
            data=summary.rename_axis("Time (minutes)").to_csv().encode("utf-8"),
            file_name="fluorescence_replicate_summary.csv",
            mime="text/csv",
        )


def _render_data_tab(
    cleaned_df: pd.DataFrame | None,
    reshaped_df: pd.DataFrame | None,
) -> None:
    st.header("Data overview")
    if cleaned_df is None or reshaped_df is None:
        st.info("Upload an assay CSV in the sidebar to inspect its data.")
        return

    left, middle, right = st.columns(3)
    left.metric("Wells with data", reshaped_df.shape[1])
    middle.metric("Time points", reshaped_df.shape[0])
    right.metric("Measurement rows", len(cleaned_df))
    st.caption("Preview of the cleaned long-form measurements")
    st.dataframe(cleaned_df.head(30), use_container_width=True)
    st.caption("Measurements arranged by well and time")
    st.dataframe(reshaped_df.head(20), use_container_width=True)
    st.download_button(
        "Download cleaned measurements",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_assay_measurements.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="Resazurin Assay Analysis",
        page_icon="🧪",
        layout="wide",
    )
    st.title("Resazurin Assay Analysis")
    st.caption(
        "Select controls and test wells visually, calculate delta T50, "
        "and explore fluorescence curves."
    )

    with st.sidebar:
        st.header("Assay input")
        uploaded_file = st.file_uploader(
            "Upload instrument CSV",
            type=["csv"],
            help="The CSV must include Well and MeasA:Result columns.",
        )
        interval = st.number_input(
            "Time between measurements (minutes)",
            min_value=0.001,
            value=15.0,
            step=1.0,
        )
        if uploaded_file is not None:
            st.caption(f"Loaded: {uploaded_file.name}")

    cleaned_df = None
    reshaped_df = None
    uploaded_name = None
    file_hash = None
    if uploaded_file is not None:
        try:
            file_contents = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_contents).hexdigest()
            cleaned_df = load_uploaded_assay(file_contents)
            reshaped_df = assay.reshape_assay_data(cleaned_df, interval)
            uploaded_name = uploaded_file.name
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.error(f"Could not load assay CSV: {error}")

    analysis_tab, curves_tab, data_tab = st.tabs(
        ["Plate analysis", "Fluorescence curves", "Data overview"]
    )
    with analysis_tab:
        _render_analysis_tab(
            cleaned_df,
            reshaped_df,
            uploaded_name,
            interval,
            file_hash,
        )
    with curves_tab:
        _render_curves_tab(reshaped_df)
    with data_tab:
        _render_data_tab(cleaned_df, reshaped_df)


if __name__ == "__main__":
    main()
