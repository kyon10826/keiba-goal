"""netkeiba パーサ共通の部品 (例外・文字列の正規化・HTML の読み込み)。"""

from __future__ import annotations

import re
import unicodedata

from lxml import html as lxml_html

_WS = re.compile(r"\s+")
_INT = re.compile(r"[0-9]+")
_FLOAT = re.compile(r"[0-9]+(?:\.[0-9]+)?")


class ParseError(ValueError):
    """ページに必要な部分が無い / 解釈できない。0 や '' で埋めずにこれを投げる。"""


def to_int(text: str | None, what: str) -> int:
    """'1,234' → 1234。半角数字以外 (空・'特払'・丸数字など) は ParseError。"""
    t = (text or "").strip().replace(",", "")
    if not _INT.fullmatch(t):
        raise ParseError(f"{what} が数字でない: {text!r}")
    return int(t)


def to_float(text: str | None, what: str) -> float:
    t = (text or "").strip().replace(",", "")
    if not _FLOAT.fullmatch(t):
        raise ParseError(f"{what} が数値でない: {text!r}")
    return float(t)


def norm(text: str | None) -> str:
    """空白類 (改行・全角空白を含む) を 1 つの半角空白にまとめて前後を削る。"""
    return _WS.sub(" ", (text or "").replace("　", " ")).strip()


def nfkc(text: str | None) -> str:
    """全角英数字を半角に (クラス名・距離などの解釈用。馬名には使わない)。"""
    return norm(unicodedata.normalize("NFKC", text or ""))


def parse_doc(text: str) -> lxml_html.HtmlElement:
    if not text or not text.strip():
        raise ParseError("空のページ")
    return lxml_html.fromstring(text)


def text_of(el) -> str:
    return norm(el.text_content()) if el is not None else ""


def first(doc, xpath: str):
    found = doc.xpath(xpath)
    return found[0] if found else None


def has_class(cls: str) -> str:
    """XPath の述語: class 属性に単語 cls を含む。"""
    return f'contains(concat(" ", normalize-space(@class), " "), " {cls} ")'
