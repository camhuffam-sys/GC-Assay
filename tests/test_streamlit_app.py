import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

import streamlit_app


def test_build_bacteria_groups_collects_wells_by_group_and_role():
    groups = streamlit_app.build_bacteria_groups(
        [" Alpha ", "Beta"],
        {
            "A01": {"group": 0, "role": "controls"},
            "A02": {"group": 0, "role": "tests"},
            "B01": {"group": 1, "role": "tests"},
        },
    )

    assert groups == [
        {"name": "Alpha", "controls": ["A01"], "tests": ["A02"]},
        {"name": "Beta", "controls": [], "tests": ["B01"]},
    ]


@pytest.mark.parametrize("value", ["P.a", "Bacteria 2", -1, 2, True, None])
def test_normalize_group_index_recovers_invalid_session_values(value):
    assert streamlit_app.normalize_group_index(value, group_count=2) == 0


def test_normalize_group_index_keeps_valid_index():
    assert streamlit_app.normalize_group_index(1, group_count=2) == 1


def test_load_uploaded_assay_uses_assay_csv_parser():
    contents = (
        b"metadata\nWell,Sample,MeasA:Result\nA1,-,3\n"
        b"A1,-,6\nBasic assay information\n"
    )

    loaded = streamlit_app.load_uploaded_assay(contents)

    assert loaded.to_dict("records") == [
        {"Well": "A01", "Measurement": 3},
        {"Well": "A01", "Measurement": 6},
    ]


def test_summarize_replicates_returns_mean_sd_and_count():
    data = pd.DataFrame(
        {"A01": [1.0, 3.0], "A02": [3.0, 7.0]},
        index=[0, 15],
    )

    summary = streamlit_app.summarize_replicates(data, ["A01", "A02"])

    assert summary["Mean"].tolist() == [2.0, 5.0]
    assert summary["Error"].tolist() == pytest.approx([2**0.5, 2**0.5 * 2])
    assert summary["N"].tolist() == [2, 2]


def test_summarize_replicates_sem_is_sd_divided_by_sqrt_n():
    data = pd.DataFrame({"A01": [0.0], "A02": [2.0], "A03": [4.0]})

    summary = streamlit_app.summarize_replicates(
        data,
        ["A01", "A02", "A03"],
        error_mode="SEM",
    )

    assert summary.loc[0, "Mean"] == 2.0
    assert summary.loc[0, "Error"] == pytest.approx(2 / 3**0.5)


def test_summarize_replicates_can_subtract_each_baseline():
    data = pd.DataFrame(
        {"A01": [10.0, 14.0], "A02": [30.0, 38.0]},
        index=[0, 15],
    )

    summary = streamlit_app.summarize_replicates(
        data,
        ["A01", "A02"],
        baseline_subtract=True,
    )

    assert summary["Mean"].tolist() == [0.0, 6.0]


def test_summarize_replicates_rejects_missing_wells():
    with pytest.raises(ValueError, match="Wells not found"):
        streamlit_app.summarize_replicates(
            pd.DataFrame({"A01": [1.0]}),
            ["A02"],
        )


def test_summarize_replicates_rejects_unknown_error_mode():
    with pytest.raises(ValueError, match="Error mode"):
        streamlit_app.summarize_replicates(
            pd.DataFrame({"A01": [1.0]}),
            ["A01"],
            error_mode="range",
        )


def test_streamlit_app_renders_all_tabs():
    app_path = Path(streamlit_app.__file__)

    app_test = AppTest.from_file(str(app_path)).run()

    assert not app_test.exception
    assert [tab.label for tab in app_test.tabs] == [
        "Plate analysis",
        "Fluorescence curves",
        "Data overview",
    ]


def test_streamlit_plate_picker_assigns_control_and_test_wells():
    app_test = AppTest.from_file(str(Path(streamlit_app.__file__))).run()

    app_test.button(key="plate_A01").click().run()
    app_test.radio(key="active_role").set_value("tests").run()
    app_test.button(key="plate_A02").click().run()

    assignments = app_test.session_state["well_assignments"]
    assert assignments == {
        "A01": {"group": 0, "role": "controls"},
        "A02": {"group": 0, "role": "tests"},
    }


def test_streamlit_name_entry_recovers_from_invalid_group_selector_state():
    app_test = AppTest.from_file(str(Path(streamlit_app.__file__))).run()
    app_test.session_state["active_group"] = "P.a"

    app_test.text_input(key="bacteria_name_1").set_value("P.a").run()

    assert not app_test.exception
    assert app_test.session_state["well_picker_group_index"] == 0
