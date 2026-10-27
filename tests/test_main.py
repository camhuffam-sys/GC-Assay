import csv

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import main


def write_assay_csv(path, measurements):
    with path.open("w", encoding="ISO-8859-1", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["assay export metadata"])
        writer.writerow(["Well", "Sample", "MeasA:Result"])
        for well, values in measurements.items():
            for value in values:
                writer.writerow([well, "-", value])
        writer.writerow(["Basic assay information", "", ""])
        writer.writerow(["malformed metadata", "", "", "", "extra"])
    return path


def example_groups():
    return [
        {
            "name": "Bacteria Alpha",
            "controls": ["A01", "A02"],
            "tests": ["A03"],
        },
        {
            "name": "Bacteria Beta",
            "controls": ["B01", "B02"],
            "tests": ["B03"],
        },
    ]


def test_clean_assay_data_normalizes_wells_and_keeps_numeric_measurements():
    raw = pd.DataFrame(
        {
            "Well": [" a1 ", "H12", "metadata"],
            "MeasA:Result": ["4.5", "8", "not a measurement"],
        }
    )

    cleaned = main.clean_assay_data(raw)

    assert cleaned.to_dict("records") == [
        {"Well": "A01", "Measurement": 4.5},
        {"Well": "H12", "Measurement": 8},
    ]


def test_clean_assay_data_rejects_missing_columns():
    with pytest.raises(ValueError, match="MeasA:Result"):
        main.clean_assay_data(pd.DataFrame({"Well": ["A01"]}))


def test_clean_assay_data_rejects_non_numeric_well_measurement():
    raw = pd.DataFrame({"Well": ["A01"], "MeasA:Result": ["invalid"]})

    with pytest.raises(ValueError, match="Non-numeric measurements"):
        main.clean_assay_data(raw)


def test_read_assay_csv_finds_header_and_stops_before_metadata(tmp_path):
    csv_path = write_assay_csv(
        tmp_path / "assay.csv",
        {"A01": [1, 2], "H12": [3]},
    )

    result = main.read_assay_csv(csv_path)

    assert result.to_dict("records") == [
        {"Well": "A01", "Measurement": 1},
        {"Well": "A01", "Measurement": 2},
        {"Well": "H12", "Measurement": 3},
    ]


def test_read_assay_csv_rejects_missing_header(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("metadata\nNo assay header\n", encoding="ISO-8859-1")

    with pytest.raises(ValueError, match="Could not find a CSV header"):
        main.read_assay_csv(csv_path)


def test_reshape_assay_data_groups_wells_and_uses_configured_interval():
    cleaned = pd.DataFrame(
        {
            "Well": ["A01", "B01", "A01", "B01"],
            "Measurement": [1, 10, 2, 20],
        }
    )

    reshaped = main.reshape_assay_data(cleaned, interval_minutes=5)

    assert reshaped.index.tolist() == [0, 5]
    assert reshaped["A01"].tolist() == [1, 2]
    assert reshaped["B01"].tolist() == [10, 20]


@pytest.mark.parametrize("interval", [0, -1])
def test_reshape_assay_data_rejects_non_positive_interval(interval):
    cleaned = pd.DataFrame({"Well": ["A01"], "Measurement": [1]})

    with pytest.raises(ValueError, match="greater than zero"):
        main.reshape_assay_data(cleaned, interval)


def test_calculate_control_f50_averages_replicate_thresholds():
    reshaped = pd.DataFrame({"A01": [0, 10, 20], "A02": [2, 12, 22]})

    result = main.calculate_control_f50(reshaped, [[["A01", "A02"]]])

    assert result == [[11.0]]


def test_calculate_control_t50_interpolates_and_returns_none_without_crossing():
    reshaped = pd.DataFrame(
        {"A01": [0, 10, 20], "A02": [0, 2, 4], "A03": [11, 12, 13]},
        index=[0, 15, 30],
    )

    result = main.calculate_control_t50(reshaped, [["A01", "A02", "A03"]], [[10]])

    assert result[0][0][0] == "A01"
    assert result[0][0][1] == pytest.approx(15)
    assert result[0][1] == ("A02", None)
    assert result[0][2] == ("A03", None)


def test_calculate_delta_t50s_normalizes_to_control_mean():
    t50s = [[("A01", 10), ("A02", 20), ("A03", 35), ("A04", None)]]

    result = main.calculate_delta_t50s(t50s, [[["A01", "A02"]]])

    assert result == [[("A01", -5), ("A02", 5), ("A03", 20), ("A04", None)]]


def test_calculate_delta_t50s_rejects_missing_control_results():
    with pytest.raises(ValueError, match="missing T50 results"):
        main.calculate_delta_t50s([[("A01", 10)]], [[["A01", "A02"]]])


def test_calculate_delta_t50s_rejects_controls_without_crossings():
    with pytest.raises(ValueError, match="No control wells crossed"):
        main.calculate_delta_t50s(
            [[("A01", None), ("A02", None)]],
            [[["A01", "A02"]]],
        )


def test_delta_t50s_to_plate_dataframe_combines_groups():
    delta_t50s = [
        [("A01", -2.5), ("D12", 1.0)],
        [("E02", 4.0), ("H12", None)],
    ]

    plate = main.delta_t50s_to_plate_dataframe(delta_t50s)

    assert plate.shape == (8, 12)
    assert plate.index.tolist() == list("ABCDEFGH")
    assert plate.columns.tolist() == list(range(1, 13))
    assert plate.loc["A", 1] == -2.5
    assert plate.loc["D", 12] == 1.0
    assert plate.loc["E", 2] == 4.0
    assert pd.isna(plate.loc["H", 12])


def test_analyze_assay_processes_multiple_bacteria_groups(tmp_path):
    csv_path = write_assay_csv(
        tmp_path / "assay.csv",
        {
            "A01": [0, 10, 20],
            "A02": [0, 10, 20],
            "A03": [0, 5, 20],
            "B01": [0, 20, 40],
            "B02": [0, 20, 40],
            "B03": [0, 10, 40],
        },
    )

    plate = main.analyze_assay(csv_path, example_groups(), interval_minutes=15)

    assert plate.loc["A", 3] == pytest.approx(5)
    assert plate.loc["B", 3] == pytest.approx(5)
    assert plate.shape == (8, 12)


def test_analyze_assay_rejects_duplicate_well_across_groups(tmp_path):
    groups = example_groups()
    groups[1]["tests"] = ["A03"]

    with pytest.raises(ValueError, match="only belong to one bacteria group"):
        main.analyze_assay(tmp_path / "unused.csv", groups)


def test_analyze_assay_rejects_duplicate_group_names(tmp_path):
    groups = example_groups()
    groups[1]["name"] = groups[0]["name"].lower()

    with pytest.raises(ValueError, match="must be unique"):
        main.analyze_assay(tmp_path / "unused.csv", groups)


def test_parse_well_list_normalizes_valid_wells():
    assert main.parse_well_list("a1, A02;\nH12") == ["A01", "A02", "H12"]


@pytest.mark.parametrize("value", ["I01", "A13", "A00"])
def test_parse_well_list_rejects_invalid_wells(value):
    with pytest.raises(ValueError):
        main.parse_well_list(value)


def test_parse_well_list_allows_empty_input():
    assert main.parse_well_list("") == []


def test_parse_well_list_rejects_duplicates():
    with pytest.raises(ValueError, match="more than once"):
        main.parse_well_list("A01, a1")


def test_save_delta_t50_heatmap_writes_png_and_csv(tmp_path):
    plate = pd.DataFrame(
        np.nan,
        index=list("ABCDEFGH"),
        columns=range(1, 13),
    )
    plate.loc["A", 1] = -2.5
    plate.loc["H", 12] = 4.0
    csv_path = tmp_path / "nested" / "plate.csv"
    png_path = tmp_path / "nested" / "heatmap.png"

    returned_paths = main.save_delta_t50_heatmap(
        plate,
        bacteria_labels=["Alpha", "Beta"],
        png_path=png_path,
        csv_path=csv_path,
        show=False,
        bacteria_wells=[["A01"], ["H12"]],
    )

    saved_plate = pd.read_csv(csv_path, index_col="Row")
    assert returned_paths == (png_path, csv_path)
    assert png_path.is_file() and png_path.stat().st_size > 0
    assert saved_plate.shape == (8, 12)
    assert saved_plate.loc["A", "1"] == -2.5
    assert saved_plate.loc["H", "12"] == 4.0


def test_save_delta_t50_heatmap_rejects_wrong_plate_shape(tmp_path):
    with pytest.raises(ValueError, match="8 rows"):
        main.save_delta_t50_heatmap(
            pd.DataFrame([[1]]),
            bacteria_labels=["Alpha"],
            png_path=tmp_path / "heatmap.png",
            csv_path=tmp_path / "plate.csv",
            show=False,
        )
