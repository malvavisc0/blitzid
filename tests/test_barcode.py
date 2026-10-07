"""Tests for blitzid.reading.barcode — AAMVA PDF417 parsing and decoding.

Parser tests are pure-function over synthetic payloads plus the
standard's own published sample (no PII: every name, number, and date
is invented or the AAMVA fictitious specimen). Decode tests round-trip
synthetic payloads through zxing-cpp's PDF417 encoder/decoder and are
skipped when the ``barcode`` extra is absent.
"""

from __future__ import annotations

import logging
import sys
from datetime import date

import numpy as np
import pytest

from blitzid import BarcodeError, BarcodeReader, BlitzIDError
from blitzid.reading.barcode import _parse_aamva, _record_dict

_VERSIONS = tuple(f"{number:02d}" for number in range(2, 15))

_AAMVA_2020_SAMPLE = (
    "@\n\x1e\rANSI 636000100102DL00410278ZV03190008DLDAQT64235789\n"
    "DCSSAMPLE\nDDEN\nDACMICHAEL\nDDFN\nDADJOHN\nDDGN\nDCUJR\nDCAD\n"
    "DCBK\nDCDPH\nDBD06062019\nDBB06061986\nDBA12102024\nDBC1\n"
    "DAU068 in\nDAYBRO\nDAG2300 WEST BROAD STREET\nDAIRICHMOND\nDAJVA\n"
    "DAK232690000  \nDCF2424244747474786102204\nDCGUSA\nDCK123456789\n"
    "DDAF\nDDB06062018\nDDC06062020\nDDD1\rZVZVA01\r"
)


def _assemble(
    version: str,
    iin: str,
    jurisdiction: str,
    subfiles: list[tuple[str, str]],
) -> str:
    """Assemble a synthetic payload, computing offsets and lengths."""
    count = len(subfiles)
    header = f"@\n\x1e\rANSI {iin}{version}{jurisdiction}{count:02d}"
    directory = ""
    bodies: list[str] = []
    offset = 21 + 10 * count
    for kind, body in subfiles:
        full = kind + body + "\r"
        directory += f"{kind}{offset:04d}{len(full):04d}"
        bodies.append(full)
        offset += len(full)
    return header + directory + "".join(bodies)


def _elements(version: str, **overrides: str) -> list[tuple[str, str]]:
    """The standard element set for one edition, with overrides."""
    given_tag = "DCT" if version in ("02", "03") else "DAC"
    given = overrides.pop("given", "JOHN")
    tags = [
        ("DCS", overrides.pop("family", "PUBLIC")),
        (given_tag, given),
        ("DAD", overrides.pop("middle", "QUINCY")),
        ("DBB", overrides.pop("birth", "01241989")),
        ("DBD", overrides.pop("issue", "06042013")),
        ("DBA", overrides.pop("expiry", "01312035")),
        ("DBC", overrides.pop("sex", "1")),
        ("DAQ", overrides.pop("license", "D12345678")),
        ("DCG", overrides.pop("country", "USA")),
        ("DAU", overrides.pop("height", "069 in")),
        ("DCA", "D"),
    ]
    tags.extend((tag, value) for tag, value in overrides.items())
    return tags


def _payload(
    version: str,
    *,
    iin: str = "636014",
    jurisdiction: str = "00",
    elements: list[tuple[str, str]] | None = None,
    z_elements: list[tuple[str, str]] | None = None,
) -> str:
    """A synthetic DL payload with an optional jurisdiction Z subfile."""
    if elements is None:
        elements = _elements(version)
    dl = "\n".join(f"{tag}{value}" for tag, value in elements)
    subfiles: list[tuple[str, str]] = [("DL", dl)]
    if z_elements is not None:
        subfiles.append(("ZC", "\n".join(f"{tag}{value}" for tag, value in z_elements)))
    return _assemble(version, iin, jurisdiction, subfiles)


def _entry(text: str, index: int) -> tuple[str, int, int]:
    """Read one subfile directory entry back out of a payload."""
    start = 21 + index * 10
    return (
        text[start : start + 2],
        int(text[start + 2 : start + 6]),
        int(text[start + 6 : start + 10]),
    )


def _on_canvas(*symbols: np.ndarray) -> np.ndarray:
    """The encoded symbols side by side on one white canvas."""
    height = max(symbol.shape[0] for symbol in symbols)
    columns = []
    for symbol in symbols:
        column = np.full((height, symbol.shape[1]), 255, dtype=np.uint8)
        column[: symbol.shape[0], : symbol.shape[1]] = symbol
        columns.append(column)
    return np.concatenate(columns, axis=1)


class TestParseHeader:
    def test_rejects_missing_prefix(self) -> None:
        with pytest.raises(BarcodeError, match="not an AAMVA payload"):
            _parse_aamva("ANSI 63601404" + "0" * 40)

    def test_rejects_unsupported_versions(self) -> None:
        for version in ("01", "15", "99"):
            payload = _payload(version)
            with pytest.raises(BarcodeError, match="unsupported AAMVA version"):
                _parse_aamva(payload)

    def test_rejects_non_digit_iin(self) -> None:
        payload = _payload("08")
        with pytest.raises(BarcodeError, match="issuer identification number"):
            _parse_aamva(payload[:9] + "X36014" + payload[15:])

    def test_rejects_non_digit_jurisdiction(self) -> None:
        payload = _payload("08")
        with pytest.raises(BarcodeError, match="jurisdiction version"):
            _parse_aamva(payload[:17] + "AB" + payload[19:])

    def test_rejects_non_digit_count(self) -> None:
        payload = _payload("08")
        with pytest.raises(BarcodeError, match="number of entries"):
            _parse_aamva(payload[:19] + "0A" + payload[21:])

    def test_rejects_zero_count(self) -> None:
        payload = _payload("08")
        with pytest.raises(BarcodeError, match="number of entries: 0"):
            _parse_aamva(payload[:19] + "00" + payload[21:])

    def test_rejects_unicode_digits_in_count(self) -> None:
        """``isdigit`` admits Unicode digits; ASCII-only must gate the int()."""
        payload = _payload("08")
        with pytest.raises(BarcodeError, match="number of entries"):
            _parse_aamva(payload[:19] + "²²" + payload[21:])

    def test_rejects_unicode_digits_in_directory(self) -> None:
        payload = _payload("08")
        with pytest.raises(BarcodeError, match="malformed subfile directory entry"):
            _parse_aamva(payload[:23] + "²²²²" + payload[27:])

    def test_count_must_match_the_directory(self) -> None:
        """A count claiming two subfiles with one entry present must fail."""
        body = "\n".join(f"{tag}{value}" for tag, value in _elements("08"))
        subfile = "DL" + body + "\r"
        payload = (
            "@\n\x1e\rANSI 636014"
            + "08"
            + "00"
            + "02"
            + f"DL{41:04d}{len(subfile):04d}"
            + "X" * 10
            + subfile
        )
        with pytest.raises(BarcodeError, match="malformed subfile directory entry"):
            _parse_aamva(payload)

    def test_rejects_offset_inside_header(self) -> None:
        payload = _payload("08")
        with pytest.raises(BarcodeError, match="lies inside the header"):
            _parse_aamva(payload[:23] + "0000" + payload[27:])

    def test_rejects_zero_length_subfile(self) -> None:
        payload = _payload("08")
        with pytest.raises(BarcodeError, match=r"invalid subfile 'DL' length: 0"):
            _parse_aamva(payload[:27] + "0000" + payload[31:])

    def test_rejects_non_letter_designator(self) -> None:
        payload = _payload("08")
        with pytest.raises(BarcodeError, match="malformed subfile directory entry"):
            _parse_aamva(payload[:21] + "1A" + payload[23:])

    def test_rejects_subfile_past_payload(self) -> None:
        payload = _payload("08")
        with pytest.raises(BarcodeError, match="exceeds the payload"):
            _parse_aamva(payload[:27] + "9999" + payload[31:])

    def test_rejects_offset_mismatch(self) -> None:
        payload = _payload("08")
        _, offset, _ = _entry(payload, 0)
        corrupted = payload[:offset] + "XY" + payload[offset + 2 :]
        with pytest.raises(BarcodeError, match="offset does not match"):
            _parse_aamva(corrupted)

    def test_rejects_missing_segment_terminator(self) -> None:
        payload = _payload("08")
        _, offset, length = _entry(payload, 0)
        end = offset + length - 1
        corrupted = payload[:end] + "X" + payload[end + 1 :]
        with pytest.raises(BarcodeError, match="segment terminator"):
            _parse_aamva(corrupted)

    def test_rejects_without_data_subfile(self) -> None:
        payload = _assemble("08", "636014", "00", [("ZC", "ZCZEXTRA")])
        with pytest.raises(BarcodeError, match="no DL/ID data subfile"):
            _parse_aamva(payload)


class TestParseRecord:
    @pytest.mark.parametrize("version", _VERSIONS)
    def test_parses_every_edition(self, version: str) -> None:
        record = _parse_aamva(_payload(version))
        assert (
            record.jurisdiction,
            record.aamva_version,
            record.jurisdiction_version,
        ) == ("636014", version, "00")
        assert (record.family_name, record.given_name, record.middle_name) == (
            "PUBLIC",
            "JOHN",
            "QUINCY",
        )
        assert (record.birth_date, record.issue_date, record.expiry_date) == (
            date(1989, 1, 24),
            date(2013, 6, 4),
            date(2035, 1, 31),
        )
        assert (record.sex, record.license_number, record.height) == (
            "M",
            "D12345678",
            "069 in",
        )

    def test_canadian_date_format(self) -> None:
        payload = _payload(
            "10",
            elements=_elements("10", country="CAN", birth="19890124"),
        )
        assert _parse_aamva(payload).birth_date == date(1989, 1, 24)

    @pytest.mark.parametrize("middle", ["QUINCY", "QUINCY,MAXIMUS"])
    def test_legacy_dct_splits_given_and_middle(self, middle: str) -> None:
        elements = [
            ("DCS", "PUBLIC"),
            ("DCT", f"JOHN,{middle}"),
            ("DBB", "01241989"),
            ("DBD", "06042013"),
            ("DBA", "01312035"),
            ("DBC", "1"),
            ("DAQ", "D12345678"),
        ]
        record = _parse_aamva(_payload("03", elements=elements))
        assert record.given_name == "JOHN"
        assert record.middle_name == middle

    def test_legacy_dad_wins_over_dct_middle(self) -> None:
        elements = [
            ("DCS", "PUBLIC"),
            ("DCT", "JOHN,QUINCY"),
            ("DAD", "MAXIMUS"),
            ("DBB", "01241989"),
            ("DBD", "06042013"),
            ("DBA", "01312035"),
            ("DBC", "1"),
            ("DAQ", "D12345678"),
        ]
        record = _parse_aamva(_payload("02", elements=elements))
        assert record.given_name == "JOHN"
        assert record.middle_name == "MAXIMUS"

    def test_middle_name_and_height_optional(self) -> None:
        elements = [
            ("DCS", "PUBLIC"),
            ("DAC", "JOHN"),
            ("DBB", "01241989"),
            ("DBD", "06042013"),
            ("DBA", "01312035"),
            ("DBC", "1"),
            ("DAQ", "D12345678"),
        ]
        record = _parse_aamva(_payload("14", elements=elements))
        assert record.middle_name == ""
        assert record.height is None

    def test_strips_values_and_treats_blank_optional_fields_as_absent(self) -> None:
        elements = _elements(
            "08", DCS="  PUBLIC  ", DAC="  JOHN  ", DAD=" \t ", DAU=" \t "
        )
        record = _parse_aamva(_payload("08", elements=elements))
        assert (record.family_name, record.given_name) == ("PUBLIC", "JOHN")
        assert record.middle_name == ""
        assert record.height is None

    def test_unknown_tags_preserved_raw(self) -> None:
        record = _parse_aamva(
            _payload(
                "10",
                z_elements=[("ZCZ", "EXTRA"), ("ZCA", "RAW")],
            )
        )
        assert record.extra_tags["ZCZ"] == "EXTRA"
        assert record.extra_tags["ZCA"] == "RAW"
        assert record.extra_tags["DCA"] == "D"
        assert record.extra_tags["DCG"] == "USA"
        for tag in ("DCS", "DAC", "DAD", "DBB", "DBD", "DBA", "DBC", "DAQ", "DAU"):
            assert tag not in record.extra_tags

    def test_jurisdiction_daa_stays_raw(self) -> None:
        """A 2016+ jurisdiction re-using DAA must not touch the name fields."""
        record = _parse_aamva(
            _payload(
                "12",
                z_elements=[("DAA", "OTHER-NAME"), ("ZCD", "AA")],
            )
        )
        assert record.family_name == "PUBLIC"
        assert record.extra_tags["DAA"] == "OTHER-NAME"
        assert record.extra_tags["ZCD"] == "AA"

    def test_unmapped_dct_in_late_editions_stays_raw(self) -> None:
        """DCT is only a mapped given-name tag in 02-03; later editions keep it raw."""
        record = _parse_aamva(
            _payload(
                "10",
                elements=_elements("10", DCT="JOHN,QUINCY"),
            )
        )
        assert record.given_name == "JOHN"
        assert record.extra_tags["DCT"] == "JOHN,QUINCY"

    def test_sex_mapping(self) -> None:
        for code, expected in (("2", "F"), ("9", "X")):
            elements = _elements("08", sex=code)
            record = _parse_aamva(_payload("08", elements=elements))
            assert record.sex == expected


class TestPublishedSample:
    def test_parses_the_aamva_2020_sample_payload(self) -> None:
        """The DL/ID 2020 CDS Annex D sample credential (fictitious).

        A published byte stream, independent of the payload generator,
        so the tag map is checked against the standard's own example.
        """
        record = _parse_aamva(_AAMVA_2020_SAMPLE)
        assert (
            record.jurisdiction,
            record.aamva_version,
            record.jurisdiction_version,
        ) == ("636000", "10", "01")
        assert (record.family_name, record.given_name, record.middle_name) == (
            "SAMPLE",
            "MICHAEL",
            "JOHN",
        )
        assert (record.birth_date, record.issue_date, record.expiry_date) == (
            date(1986, 6, 6),
            date(2019, 6, 6),
            date(2024, 12, 10),
        )
        assert (record.sex, record.license_number, record.height) == (
            "M",
            "T64235789",
            "068 in",
        )
        assert (
            record.extra_tags["DCU"],
            record.extra_tags["DDF"],
            record.extra_tags["ZVA"],
        ) == ("JR", "N", "01")


class TestSectionBody:
    def test_record_dict_serializes_a_parsed_record(self) -> None:
        body = _record_dict(_parse_aamva(_payload("10")))
        assert body["birth_date"] == "1989-01-24"
        assert body["issue_date"] == "2013-06-04"
        assert body["expiry_date"] == "2035-01-31"
        assert body["extra_tags"] == {"DCA": "D", "DCG": "USA"}
        assert body["symbology"] is None


class TestParseValidation:
    @pytest.mark.parametrize("name", [None, " \t "])
    def test_missing_family_name(self, name: str | None) -> None:
        elements = [tag for tag in _elements("08") if tag[0] != "DCS"]
        if name is not None:
            elements.append(("DCS", name))
        with pytest.raises(BarcodeError, match="missing family name"):
            _parse_aamva(_payload("08", elements=elements))

    def test_missing_given_name(self) -> None:
        elements = [tag for tag in _elements("08") if tag[0] != "DAC"]
        with pytest.raises(BarcodeError, match="missing given name"):
            _parse_aamva(_payload("08", elements=elements))

    def test_missing_birth_date(self) -> None:
        elements = [tag for tag in _elements("08") if tag[0] != "DBB"]
        with pytest.raises(BarcodeError, match="missing birth date"):
            _parse_aamva(_payload("08", elements=elements))

    def test_invalid_birth_date(self) -> None:
        elements = _elements("08", birth="31970")
        with pytest.raises(BarcodeError, match=r"birth date \(DBB\).*not a real date"):
            _parse_aamva(_payload("08", elements=elements))

    def test_impossible_birth_date(self) -> None:
        elements = _elements("08", birth="02991970")
        with pytest.raises(BarcodeError, match="not a real date"):
            _parse_aamva(_payload("08", elements=elements))

    def test_nonstandard_sex_maps_to_unspecified(self) -> None:
        elements = _elements("08", sex="3")
        record = _parse_aamva(_payload("08", elements=elements))
        assert record.sex == "X"
        assert record.extra_tags["DBC"] == "3"

    def test_missing_sex(self) -> None:
        elements = [tag for tag in _elements("08") if tag[0] != "DBC"]
        with pytest.raises(BarcodeError, match="missing sex"):
            _parse_aamva(_payload("08", elements=elements))

    def test_missing_license_number(self) -> None:
        elements = [tag for tag in _elements("08") if tag[0] != "DAQ"]
        with pytest.raises(BarcodeError, match="missing license number"):
            _parse_aamva(_payload("08", elements=elements))


class TestReader:
    def test_init_raises_without_zxingcpp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "zxingcpp", None)
        with pytest.raises(BlitzIDError, match=r"blitzid\[barcode\]"):
            BarcodeReader()


class TestDecodeRoundTrip:
    def _encode(self, text: str) -> np.ndarray:
        zxingcpp = pytest.importorskip("zxingcpp")
        return np.asarray(
            zxingcpp.create_barcode(
                text, format=zxingcpp.BarcodeFormat.PDF417
            ).to_image()
        )

    def test_round_trips_a_v08_payload(self) -> None:
        pytest.importorskip("zxingcpp")
        reader = BarcodeReader(log_level=logging.WARNING)
        payload = _payload("08")
        record = reader.read(self._encode(payload))
        assert record.aamva_version == "08"
        assert record.symbology == "PDF417"
        assert record.family_name == "PUBLIC"
        assert record.birth_date == date(1989, 1, 24)
        assert record.license_number == "D12345678"

    def test_round_trips_a_v02_payload(self) -> None:
        pytest.importorskip("zxingcpp")
        reader = BarcodeReader(log_level=logging.WARNING)
        record = reader.read(self._encode(_payload("02")))
        assert record.aamva_version == "02"
        assert record.given_name == "JOHN"

    def test_accepts_a_pil_image(self) -> None:
        pytest.importorskip("zxingcpp")
        pytest.importorskip("PIL")
        from PIL import Image

        reader = BarcodeReader(log_level=logging.WARNING)
        record = reader.read(Image.fromarray(self._encode(_payload("10"))))
        assert record.aamva_version == "10"

    def test_accepts_a_three_channel_array(self) -> None:
        pytest.importorskip("zxingcpp")
        reader = BarcodeReader(log_level=logging.WARNING)
        gray = self._encode(_payload("08"))
        record = reader.read(np.repeat(gray[:, :, None], 3, axis=2))
        assert record.aamva_version == "08"

    def test_falls_back_past_a_corrupt_candidate(self) -> None:
        pytest.importorskip("zxingcpp")
        reader = BarcodeReader(log_level=logging.WARNING)
        canvas = _on_canvas(self._encode(_payload("15")), self._encode(_payload("08")))
        record = reader.read(canvas)
        assert record.aamva_version == "08"

    def test_raises_the_candidate_error_when_nothing_parses(self) -> None:
        pytest.importorskip("zxingcpp")
        reader = BarcodeReader(log_level=logging.WARNING)
        with pytest.raises(BarcodeError, match="unsupported AAMVA version"):
            reader.read(self._encode(_payload("15")))

    def test_no_barcode_raises(self) -> None:
        pytest.importorskip("zxingcpp")
        reader = BarcodeReader(log_level=logging.WARNING)
        blank = np.zeros((200, 400, 3), dtype=np.uint8)
        with pytest.raises(BarcodeError, match="no PDF417 barcode"):
            reader.read(blank)

    def test_non_aamva_payload_raises(self) -> None:
        pytest.importorskip("zxingcpp")
        reader = BarcodeReader(log_level=logging.WARNING)
        with pytest.raises(BarcodeError, match="no AAMVA payload"):
            reader.read(self._encode("garbage, not a license"))
