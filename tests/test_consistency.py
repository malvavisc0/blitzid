"""Tests for the printed-vs-code-strip consistency check (no models)."""

from __future__ import annotations

from datetime import date

import pytest

from blitzid import (
    ConsistencyReport,
    ExtractedField,
    FaceAttributes,
    MRZRecord,
    StructuredOCR,
    cross_check,
)

_TODAY = date(2026, 10, 1)


def _mrz(**overrides: str) -> MRZRecord:
    base: dict[str, str] = {
        "mrz_type": "TD3",
        "document_code": "P<",
        "issuer": "UTO",
        "document_number": "SPECI2014",
        "birth_date": "650310",
        "sex": "F",
        "expiry_date": "260309",
        "nationality": "UTO",
        "surname": "DE BRUIJN",
        "given_names": "WILLEKE LISELOTTE",
        "optional_data1": "",
        "optional_data2": "",
    }
    return MRZRecord(**(base | overrides))


def _structured(**values: object) -> StructuredOCR:
    return StructuredOCR(
        document_type="id",
        fields=[
            ExtractedField(name=name, value=value, confidence=0.9)  # type: ignore[arg-type]
            for name, value in values.items()
        ],
        raw_text="ID",
    )


_MATCHING: dict[str, object] = {
    "document_number": "SPECI2014",
    "date_of_birth": date(1965, 3, 10),
    "date_of_expiry": date(2026, 3, 9),
    "surname": "DE BRUIJN",
    "given_names": "WILLEKE LISELOTTE",
    "sex": "F",
}


def _verdicts(report: ConsistencyReport) -> dict[str, str]:
    return {row.field: row.verdict for row in report.comparisons}


def _photo(age: str = "60-69", gender: str = "Female") -> FaceAttributes:
    return FaceAttributes(
        bbox=(0, 0, 10, 10),
        confidence=0.95,
        age=age,
        gender=gender,
        race="White",
        age_confidence=0.5,
        gender_confidence=0.9,
        race_confidence=0.7,
    )


class TestVerdicts:
    def test_every_field_gets_a_row(self) -> None:
        report = cross_check(_structured(**_MATCHING), _mrz(), today=_TODAY)
        assert [row.field for row in report.comparisons] == [
            "document_number",
            "date_of_birth",
            "date_of_expiry",
            "surname",
            "given_names",
            "sex",
        ]
        assert _verdicts(report) == dict.fromkeys(_verdicts(report), "match")
        assert report.consistent is True

    def test_row_keeps_both_sides_and_confidence(self) -> None:
        report = cross_check(_structured(**_MATCHING), _mrz(), today=_TODAY)
        birth = report.comparisons[1]
        assert birth.mrz_value == "650310"
        assert birth.printed_value == "1965-03-10"
        assert birth.printed_confidence == pytest.approx(0.9)

    def test_mismatch_is_flagged_and_breaks_consistency(self) -> None:
        values = dict(_MATCHING, document_number="SPECI2015")
        report = cross_check(_structured(**values), _mrz(), today=_TODAY)
        assert _verdicts(report)["document_number"] == "mismatch"
        assert report.consistent is False

    def test_document_number_ignores_case_and_fillers(self) -> None:
        values = dict(_MATCHING, document_number="speci2014<<<")
        report = cross_check(_structured(**values), _mrz(), today=_TODAY)
        assert _verdicts(report)["document_number"] == "match"

    def test_names_match_ignoring_word_order(self) -> None:
        values = dict(_MATCHING, given_names="LISELOTTE WILLEKE")
        report = cross_check(_structured(**values), _mrz(), today=_TODAY)
        assert _verdicts(report)["given_names"] == "match"

    def test_name_subset_is_partial_but_consistent(self) -> None:
        values = dict(_MATCHING, surname="BRUIJN")
        report = cross_check(_structured(**values), _mrz(), today=_TODAY)
        assert _verdicts(report)["surname"] == "partial"
        assert report.consistent is True

    def test_disjoint_names_mismatch(self) -> None:
        values = dict(_MATCHING, surname="JANSEN")
        report = cross_check(_structured(**values), _mrz(), today=_TODAY)
        assert _verdicts(report)["surname"] == "mismatch"

    def test_missing_printed_field_is_unavailable(self) -> None:
        values = {k: v for k, v in _MATCHING.items() if k != "sex"}
        report = cross_check(_structured(**values), _mrz(), today=_TODAY)
        row = report.comparisons[-1]
        assert row.verdict == "unavailable"
        assert row.mrz_value == "F"
        assert row.printed_value is None
        assert report.consistent is True

    def test_missing_on_both_sides_is_unavailable(self) -> None:
        values = {k: v for k, v in _MATCHING.items() if k != "given_names"}
        report = cross_check(_structured(**values), _mrz(given_names=""), today=_TODAY)
        assert _verdicts(report)["given_names"] == "unavailable"

    def test_sex_compares_ignoring_case(self) -> None:
        values = dict(_MATCHING, sex="f")
        report = cross_check(_structured(**values), _mrz(), today=_TODAY)
        assert _verdicts(report)["sex"] == "match"


class TestDates:
    def test_iso_and_dotted_printed_dates_match(self) -> None:
        for printed in ("1965-03-10", "10.03.1965", date(1965, 3, 10)):
            values = dict(_MATCHING, date_of_birth=printed)
            report = cross_check(_structured(**values), _mrz(), today=_TODAY)
            assert _verdicts(report)["date_of_birth"] == "match", printed

    def test_mismatched_date_is_flagged(self) -> None:
        values = dict(_MATCHING, date_of_birth=date(1966, 3, 10))
        report = cross_check(_structured(**values), _mrz(), today=_TODAY)
        assert _verdicts(report)["date_of_birth"] == "mismatch"

    def test_birth_century_slides_to_the_past(self) -> None:
        values = dict(_MATCHING, date_of_birth=date(1927, 1, 1))
        report = cross_check(
            _structured(**values), _mrz(birth_date="270101"), today=_TODAY
        )
        assert _verdicts(report)["date_of_birth"] == "match"

    def test_birth_century_slides_to_the_2000s(self) -> None:
        values = dict(_MATCHING, date_of_birth=date(2024, 3, 15))
        report = cross_check(
            _structured(**values), _mrz(birth_date="240315"), today=_TODAY
        )
        assert _verdicts(report)["date_of_birth"] == "match"

    def test_expiry_always_lands_in_the_2000s(self) -> None:
        values = dict(_MATCHING, date_of_expiry=date(2024, 3, 9))
        report = cross_check(
            _structured(**values), _mrz(expiry_date="240309"), today=_TODAY
        )
        assert _verdicts(report)["date_of_expiry"] == "match"

    def test_unparseable_printed_date_is_unavailable(self) -> None:
        values = dict(_MATCHING, date_of_birth="sometime in spring")
        report = cross_check(_structured(**values), _mrz(), today=_TODAY)
        assert _verdicts(report)["date_of_birth"] == "unavailable"

    def test_birth_date_alias_is_compared(self) -> None:
        values = {k: v for k, v in _MATCHING.items() if k != "date_of_birth"}
        values["birth_date"] = date(1965, 3, 10)
        report = cross_check(_structured(**values), _mrz(), today=_TODAY)
        assert _verdicts(report)["date_of_birth"] == "match"


class TestPhotoCrossCheck:
    def test_no_photo_rows_are_unavailable(self) -> None:
        report = cross_check(_structured(**_MATCHING), _mrz(), today=_TODAY)
        rows = [row.verdict for row in report.photo_comparisons]
        assert rows == ["unavailable", "unavailable"]
        assert report.consistent is True

    def test_age_band_match(self) -> None:
        report = cross_check(
            _structured(**_MATCHING), _mrz(), photo=_photo(age="60-69"), today=_TODAY
        )
        age = report.photo_comparisons[0]
        assert age.verdict == "match"
        assert age.document_value == "61"
        assert age.photo_value == "60-69"
        assert age.photo_confidence == pytest.approx(0.5)

    def test_age_within_one_band_is_partial(self) -> None:
        report = cross_check(
            _structured(**_MATCHING), _mrz(), photo=_photo(age="50-59"), today=_TODAY
        )
        assert report.photo_comparisons[0].verdict == "partial"
        assert report.consistent is True

    def test_age_far_off_is_mismatch_and_flips_consistent(self) -> None:
        report = cross_check(
            _structured(**_MATCHING), _mrz(), photo=_photo(age="30-39"), today=_TODAY
        )
        assert report.photo_comparisons[0].verdict == "mismatch"
        assert report.consistent is False

    def test_open_ended_band_matches_old_ages(self) -> None:
        values = dict(_MATCHING, date_of_birth=date(1940, 1, 1))
        report = cross_check(
            _structured(**values),
            _mrz(birth_date="400101"),
            photo=_photo(age="70+"),
            today=_TODAY,
        )
        assert report.photo_comparisons[0].verdict == "match"

    def test_gender_match_and_mismatch(self) -> None:
        match = cross_check(
            _structured(**_MATCHING),
            _mrz(),
            photo=_photo(gender="Female"),
            today=_TODAY,
        )
        assert match.photo_comparisons[1].verdict == "match"
        clash = cross_check(
            _structured(**_MATCHING),
            _mrz(),
            photo=_photo(gender="Male"),
            today=_TODAY,
        )
        assert clash.photo_comparisons[1].verdict == "mismatch"
        assert clash.consistent is False

    def test_document_sex_x_is_partial(self) -> None:
        values = dict(_MATCHING, sex="X")
        report = cross_check(
            _structured(**values), _mrz(sex="X"), photo=_photo(), today=_TODAY
        )
        sex = report.photo_comparisons[1]
        assert sex.verdict == "partial"
        assert sex.document_value == "X"
        assert report.consistent is True

    def test_birth_falls_back_to_the_printed_side(self) -> None:
        report = cross_check(
            _structured(**_MATCHING),
            _mrz(birth_date="ABCDEF"),
            photo=_photo(age="60-69"),
            today=_TODAY,
        )
        age = report.photo_comparisons[0]
        assert age.verdict == "match"
        assert age.document_value == "61"
