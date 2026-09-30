"""Tests for blitzid.reading.mrz — ICAO 9303 check digits, parsing, line selection.

All tests are pure-function / fast: no OCR engine, models, or network.
Test vectors are the public ICAO 9303 specimen strings.
"""

from __future__ import annotations

import sys

import pytest

from blitzid import MRZReader
from blitzid.exceptions import BlitzIDError, MRZError
from blitzid.reading.mrz import (
    _check_digit,
    _is_mrz_line,
    _parse,
    _parse_name,
    _read_mrz,
    _select_mrz_lines,
)
from blitzid.reading.ocr import OCRText

TD1 = [
    "I<UTOD231458907<<<<<<<<<<<<<<<",
    "7408122F1204159UTO<<<<<<<<<<<6",
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
]
TD2 = [
    "I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<",
    "D231458907UTO7408122F1204159<<<<<<<6",
]
TD3 = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
]


def _texts(*lines: str) -> list[OCRText]:
    return [
        OCRText(bbox=(0, 30 * idx, 100, 20), text=text, confidence=0.9)
        for idx, text in enumerate(lines)
    ]


class TestCheckDigit:
    def test_known_values(self) -> None:
        assert _check_digit("L898902C3") == 6
        assert _check_digit("740812") == 2
        assert _check_digit("120415") == 9
        assert _check_digit("ZE184226B<<<<<") == 1

    def test_filler_is_zero(self) -> None:
        assert _check_digit("<") == 0
        assert _check_digit("<<<") == 0

    def test_letter_values(self) -> None:
        assert _check_digit("D") == 1  # 13 * 7 = 91
        assert _check_digit("B") == 7  # 11 * 7 = 77
        assert _check_digit("DB") == 4  # 91 + 33 = 124


class TestIsMrzLine:
    def test_accepts_valid_lines(self) -> None:
        for line in TD1 + TD2 + TD3:
            assert _is_mrz_line(line)

    def test_rejects_wrong_length(self) -> None:
        assert not _is_mrz_line("I<UTO" + "<" * 24)

    def test_rejects_all_filler_lines(self) -> None:
        assert not _is_mrz_line("<" * 30)
        assert not _is_mrz_line("<" * 44)

    def test_rejects_lowercase_and_spaces(self) -> None:
        assert not _is_mrz_line(TD1[0].lower())
        assert not _is_mrz_line("I<UTO D231458907<<<<<<<<<<<<")


class TestSelectMrzLines:
    def test_td1_bottom_most_with_noise(self) -> None:
        noise = [
            "BUNDESREPUBLIK DEUTSCHLAND",
            "LZ6311T4777111111111111111",
        ]
        groups = list(_select_mrz_lines(_texts(*noise, *TD1)))
        assert groups == [TD1]

    def test_reassembles_line_order_by_y(self) -> None:
        """Confidence-sorted input must be re-assembled by bbox y."""
        swapped = [
            OCRText(bbox=(0, 30, 100, 20), text=TD3[1], confidence=0.5),
            OCRText(bbox=(0, 0, 100, 20), text=TD3[0], confidence=0.9),
        ]
        assert list(_select_mrz_lines(swapped)) == [TD3]

    def test_strips_ocr_spaces(self) -> None:
        spaced = [line.replace("<", "< ", 3) for line in TD2]
        assert list(_select_mrz_lines(_texts(*spaced))) == [TD2]

    def test_noise_line_interleaved_between_td1_lines(self) -> None:
        """A 30-char noise line between MRZ lines must not displace them."""
        noise = "AB" * 15
        texts = _texts(TD1[0], noise, TD1[1], TD1[2])
        assert TD1 in list(_select_mrz_lines(texts))

    def test_td3_with_three_30_char_noise_lines(self) -> None:
        """30-char noise must not shadow a valid 44-char TD3 group."""
        noise = ["AB" * 15, "CD" * 15, "EF" * 15]
        texts = _texts(*noise, *TD3)
        assert _read_mrz(texts).mrz_type == "TD3"

    def test_no_mrz_yields_no_candidates(self) -> None:
        assert not list(_select_mrz_lines(_texts("NOT AN MRZ LINE", "12345")))


class TestReadMrz:
    def test_reads_td3(self) -> None:
        assert _read_mrz(_texts(*TD3)).document_number == "L898902C3"

    def test_incomplete_group_raises(self) -> None:
        with pytest.raises(MRZError, match="no MRZ"):
            _read_mrz(_texts(TD3[0]))

    def test_no_valid_group_raises(self) -> None:
        """Candidate groups exist but every one fails validation."""
        bad_digit = TD1[1].replace("740812", "740813", 1)
        with pytest.raises(MRZError, match=r"no valid MRZ: .*birth date"):
            _read_mrz(_texts(TD1[0], bad_digit, TD1[2]))

    def test_invalid_group_reports_field(self) -> None:
        bad = [TD3[0], TD3[1][:-1] + "1"]
        with pytest.raises(MRZError, match="TD3 composite"):
            _read_mrz(_texts(*bad))


class TestParse:
    def test_all_filler_group_rejected(self) -> None:
        """All-'<' lines must never parse as a valid MRZ."""
        with pytest.raises(MRZError, match="no MRZ"):
            _read_mrz(_texts("<" * 30, "<" * 30, "<" * 30))

    def test_filler_document_code_rejected(self) -> None:
        bad = ["<<" + TD1[0][2:], TD1[1], TD1[2]]
        with pytest.raises(MRZError, match="TD1 document code"):
            _parse(bad)

    def test_filler_issuer_rejected(self) -> None:
        bad = [TD1[0][:2] + "<<<" + TD1[0][5:], TD1[1], TD1[2]]
        with pytest.raises(MRZError, match="TD1 issuer"):
            _parse(bad)

    def test_trailing_filler_in_issuer_tolerated(self) -> None:
        """Trailing '<' fillers are ignored, but a letter must remain."""
        padded = [TD1[0][:2] + "UT<" + TD1[0][5:], TD1[1], TD1[2]]
        assert _parse(padded).issuer == "UT<"

    def test_filler_nationality_rejected(self) -> None:
        bad = [TD1[0], TD1[1][:15] + "<<<" + TD1[1][18:], TD1[2]]
        with pytest.raises(MRZError, match="TD1 nationality"):
            _parse(bad)

    def test_impossible_birth_date_rejected(self) -> None:
        """991239 (month 99, day 12) has valid check digits but no date."""
        l1 = TD1[0]
        l2 = (
            "991239"
            + str(_check_digit("991239"))
            + "F"
            + "120415"
            + str(_check_digit("120415"))
            + "UTO"
            + "<" * 11
        )
        composite = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
        l2 += str(_check_digit(composite))
        with pytest.raises(MRZError, match=r"TD1 birth date.*not a real date"):
            _parse([l1, l2, "<" * 30])

    def test_impossible_expiry_date_rejected(self) -> None:
        bad = [TD1[0], TD1[1][:8] + "990231" + TD1[1][14:], TD1[2]]
        with pytest.raises(MRZError, match="TD1 expiry date"):
            _parse(bad)

    def test_leap_day_accepted(self) -> None:
        """Feb 29 on a leap century (2000) is a real date and must pass."""
        birth = "000229"
        l1 = TD1[0]
        l2 = (
            birth
            + str(_check_digit(birth))
            + "F"
            + "120415"
            + str(_check_digit("120415"))
            + "UTO"
            + "<" * 11
        )
        composite = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
        l2 += str(_check_digit(composite))
        assert _parse([l1, l2, TD1[2]]).birth_date == "000229"

    def test_td1(self) -> None:
        record = _parse(TD1)
        assert (record.mrz_type, record.document_code, record.issuer) == (
            "TD1",
            "I<",
            "UTO",
        )
        assert (record.document_number, record.nationality) == ("D23145890", "UTO")
        assert (record.birth_date, record.sex, record.expiry_date) == (
            "740812",
            "F",
            "120415",
        )
        assert (record.surname, record.given_names) == ("ERIKSSON", "ANNA MARIA")
        assert (record.optional_data1, record.optional_data2) == ("<" * 15, "<" * 11)

    def test_td2(self) -> None:
        record = _parse(TD2)
        assert (record.mrz_type, record.document_code, record.issuer) == (
            "TD2",
            "I<",
            "UTO",
        )
        assert (record.document_number, record.nationality) == ("D23145890", "UTO")
        assert (record.birth_date, record.sex, record.expiry_date) == (
            "740812",
            "F",
            "120415",
        )
        assert (record.surname, record.given_names) == ("ERIKSSON", "ANNA MARIA")
        assert (record.optional_data1, record.optional_data2) == ("<<<<<<<", "")

    def test_td3(self) -> None:
        record = _parse(TD3)
        assert (record.mrz_type, record.document_code, record.issuer) == (
            "TD3",
            "P<",
            "UTO",
        )
        assert (record.document_number, record.nationality) == ("L898902C3", "UTO")
        assert (record.birth_date, record.sex, record.expiry_date) == (
            "740812",
            "F",
            "120415",
        )
        assert (record.surname, record.given_names) == ("ERIKSSON", "ANNA MARIA")
        assert (record.optional_data1, record.optional_data2) == (
            "ZE184226B<<<<<",
            "",
        )

    def test_bad_document_number_check_digit(self) -> None:
        bad = [TD3[0], "L898902C30UTO7408122F1204159ZE184226B<<<<<10"]
        with pytest.raises(MRZError, match="TD3 document number"):
            _parse(bad)

    def test_bad_composite_check_digit(self) -> None:
        bad = [TD3[0], TD3[1][:-1] + "1"]
        with pytest.raises(MRZError, match="TD3 composite"):
            _parse(bad)

    def test_bad_sex_raises(self) -> None:
        bad = [TD1[0], TD1[1][:7] + "2" + TD1[1][8:], TD1[2]]
        with pytest.raises(MRZError, match="sex"):
            _parse(bad)

    def test_ocr_digit_in_issuer_raises(self) -> None:
        """O→0 confusion in a field without a check digit must fail loudly."""
        bad = ["P<UT0ERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", TD3[1]]
        with pytest.raises(MRZError, match="TD3 issuer"):
            _parse(bad)

    def test_ocr_digit_in_document_code_raises(self) -> None:
        bad = [TD1[0].replace("I<", "1<"), TD1[1], TD1[2]]
        with pytest.raises(MRZError, match="TD1 document code"):
            _parse(bad)

    def test_filler_document_number_stripped(self) -> None:
        """Trailing '<' fillers in the document number are stripped.

        '<' and '0' share check-digit value 0, so swapping the trailing
        digit for a filler keeps both document-number and composite
        check digits valid.
        """
        l1 = TD1[0][:5] + "D2314589<" + "7" + "<" * 15
        record = _parse([l1, TD1[1], TD1[2]])
        assert record.document_number == "D2314589"

    def test_unspecified_sex_normalized(self) -> None:
        lines = [TD1[0], TD1[1][:7] + "<" + TD1[1][8:], TD1[2]]
        assert _parse(lines).sex == "X"

    def test_unsupported_layout_raises(self) -> None:
        with pytest.raises(MRZError, match="unsupported"):
            _parse(["AB", "CD"])

    def test_td3_empty_personal_number_allows_filler_check(self) -> None:
        lines = [TD3[0], "L898902C36UTO7408122F1204159" + "<" * 15 + "8"]
        record = _parse(lines)
        assert record.optional_data1 == "<" * 14

    def test_td3_filler_check_with_data_raises(self) -> None:
        bad = [TD3[0], TD3[1][:42] + "<" + TD3[1][43]]
        with pytest.raises(MRZError, match="TD3 personal number"):
            _parse(bad)


class TestParseName:
    def test_primary_and_secondary(self) -> None:
        assert _parse_name("ERIKSSON<<ANNA<MARIA<<<<<<<<<<") == (
            "ERIKSSON",
            "ANNA MARIA",
        )

    def test_surname_only(self) -> None:
        assert _parse_name("SVEN<<<<<<<<<<<<<<<<<<<<<<") == ("SVEN", "")

    def test_empty(self) -> None:
        assert _parse_name("<<<<<<<<<<<<<<<<<<<<<<<<<<") == ("", "")


class TestMissingExtra:
    def test_init_raises_without_rapidocr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "rapidocr", None)
        with pytest.raises(BlitzIDError, match=r"blitzid\[ocr\]"):
            MRZReader()
