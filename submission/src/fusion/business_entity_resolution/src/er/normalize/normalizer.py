'record normaliser: raw (entity_id, business_name, business_address, country) ->'
import re

import polars as pl
from unidecode import unidecode

from er.normalize.rules import (ADDR_CANON, ADDR_FILLER, ADDR_NULL_COMPONENTS, ALIAS_MARKERS, IN_STATES,
                                LEET, LEGAL, LEGAL_SKEL, LEGAL_TRANSLIT, NAME_STOP, US_STATES, skeleton)

NON_ASCII = r"[^\x00-\x7F]"


def ascii_fold(s: pl.Series) -> pl.Series:
    'unidecode only the (unique) non-ascii strings, then lowercase'
    s = s.fill_null("")
    uniq = s.filter(s.str.contains(NON_ASCII)).unique()
    if len(uniq):
        s = s.replace(uniq, [unidecode(x) for x in uniq.to_list()])
    return s.str.to_lowercase()



_STATE_EXACT, _STATE_SKEL = {}, {}
for _d in (US_STATES, IN_STATES):
    for _full, _code in _d.items():
        _STATE_EXACT[_full] = _code
        _STATE_EXACT[_code] = _code
        _sk = skeleton(_full.replace(" ", ""))
        if len(_sk) >= 2:
            _STATE_SKEL.setdefault(_sk, _code)


def _state_of(comp: str):
    c = comp.strip()
    if c in _STATE_EXACT:
        return _STATE_EXACT[c]
    c2 = re.sub(r"[^a-z]", "", c)
    if len(c2) >= 4 and skeleton(c2) in _STATE_SKEL:
        return _STATE_SKEL[skeleton(c2)]
    return None



def _name_tok_map(tok: str):
    '-> (canonical token, kind) with kind 0 = core, 1 = legal form, 2 = stop word'
    if not tok.isdigit() and not tok.isalpha() and sum(c.isalpha() for c in tok) >= 2:
        tok = tok.translate(LEET)
    if tok in LEGAL:
        return LEGAL[tok], 1
    if tok in NAME_STOP:
        return tok, 2
    if tok in LEGAL_TRANSLIT:
        return LEGAL_TRANSLIT[tok], 1
    if not tok.isdigit() and len(tok) >= 5:
        sk = skeleton(tok)
        if len(sk) >= 3 and sk in LEGAL_SKEL:
            return LEGAL_SKEL[sk], 1
    return tok, 0


def _addr_tok_map(tok: str) -> str:
    m = re.fullmatch(r"0*(\d+)(?:st|nd|rd|th)?", tok)
    if m:
        return m.group(1) or "0"
    return ADDR_CANON.get(tok, tok)



_DOMAIN_FULL = r"^(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|biz|co|io|us|info)$"
_DOMAIN_ANY = r"(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|co\.in|in|fr|biz|co|io|us|info)(?:[^a-z0-9]|$)"


def _clean_name_text(s: pl.Series) -> pl.Series:
    s = s.fill_null("").str.strip_chars()
    is_dom = s.str.contains(_DOMAIN_FULL)
    s = pl.select(pl.when(is_dom).then(s.str.extract(_DOMAIN_FULL, 1)).otherwise(s)).to_series()
    s = s.str.replace_all(_DOMAIN_ANY, "$1 ")
    return (s.str.replace_all("&", " and ")
             .str.replace_all(r"(^|[^a-z0-9])([a-z])\.", "$1$2")
             .str.replace_all(r"(^|[^a-z0-9])([a-z]{1,2})\.", "$1$2")
             .str.replace_all("'", "")
             .str.replace_all(r"[^a-z0-9]+", " ")
             .str.strip_chars())


def clean_names(name: pl.Series) -> pl.DataFrame:
    n = ascii_fold(name)
    n = (n.str.replace_all(r"\(\s*id:?\s*\d+\s*\)", " ")
          .str.replace_all(r"\bid:\s*\d+", " ")
          .str.replace_all(r"\s-\s*\+?\d[\d\s]{5,}", " ")
          .str.replace_all(r"#\d{3,}", " ")
          .str.replace_all(r"\d{7,}", " "))
    alias_re = rf"^(.*?)\s*{ALIAS_MARKERS}\s*(.*)$"
    has_alias = n.str.contains(ALIAS_MARKERS)
    alt = pl.select(pl.when(has_alias).then(n.str.extract(alias_re, 1)).otherwise(pl.lit(""))).to_series()
    core = pl.select(pl.when(has_alias).then(n.str.extract(alias_re, 2)).otherwise(n)).to_series()
    return pl.DataFrame({"n_core_raw": _clean_name_text(core), "n_alt_raw": _clean_name_text(alt),
                         "n_is_dom": core.fill_null("").str.contains(r"\.(com|net|org|in|fr|biz|co|io)$")})


def name_fields(df: pl.DataFrame) -> pl.DataFrame:
    toks = df.select(pl.col("n_core_raw").str.split(" ").alias("t"), pl.col("n_alt_raw").str.split(" ").alias("ta"))
    uniq = pl.concat([toks["t"], toks["ta"]]).explode().unique().drop_nulls().to_list()
    mp = {t: _name_tok_map(t) for t in uniq}
    lut = pl.DataFrame({"tok": list(mp), "canon": [v[0] for v in mp.values()],
                        "kind": pl.Series([v[1] for v in mp.values()], dtype=pl.Int8),
                        "skel": [skeleton(v[0]) for v in mp.values()]})

    def agg(col):
        e = (toks.with_row_index("rid").select("rid", pl.col(col).alias("tok")).explode("tok")
             .filter(pl.col("tok") != "").join(lut, on="tok", how="left", maintain_order="left"))
        return e.group_by("rid", maintain_order=True).agg(
            pl.col("canon").filter(pl.col("kind") == 0).str.join(" ").alias("core"),
            pl.col("skel").filter(pl.col("kind") == 0).str.join(" ").alias("skel"),
            pl.col("canon").filter(pl.col("kind") == 1).unique().sort().str.join(" ").alias("legal"),
            pl.col("canon").filter(pl.col("kind") != 2).str.join(" ").alias("full"))

    base = pl.DataFrame({"rid": pl.arange(0, df.height, eager=True).cast(pl.UInt32)})
    out = (base.join(agg("t").with_columns(pl.col("rid").cast(pl.UInt32)), on="rid", how="left")
               .join(agg("ta").select(pl.col("rid").cast(pl.UInt32), pl.col("core").alias("alt")), on="rid", how="left")
               .with_columns([pl.col(c).fill_null("") for c in ["core", "skel", "legal", "full", "alt"]]))
    return out.select(pl.col("core").alias("name_core"), pl.col("skel").alias("name_skel"),
                      pl.col("legal").alias("name_legal"), pl.col("full").alias("name_full"),
                      pl.col("alt").alias("name_alt"), pl.col("core").str.replace_all(" ", "").alias("name_compact"))



def address_fields(addr: pl.Series) -> pl.DataFrame:
    a = ascii_fold(addr)
    comps = (pl.DataFrame({"c": a.str.split(",")}).with_row_index("rid")
             .explode("c").with_columns(pl.col("c").str.strip_chars()))
    comps = comps.filter(pl.col("c").is_not_null() & ~pl.col("c").is_in(ADDR_NULL_COMPONENTS))

    cand = comps.filter(~pl.col("c").str.contains(r"\d") & (pl.col("c").str.len_chars() <= 30))["c"].unique()
    smap = {k: v for k, v in ((c, _state_of(c)) for c in cand.to_list()) if v}
    comps = comps.with_columns(pl.col("c").replace_strict(smap, default=None).alias("state"))
    state = comps.filter(pl.col("state").is_not_null()).group_by("rid").agg(pl.col("state").first())
    tk = (comps.filter(pl.col("state").is_null())
               .with_columns(pl.col("c").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().str.split(" ").alias("tok"))
               .select("rid", "tok").explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")))
    mp = {t: _addr_tok_map(t) for t in tk["tok"].unique().to_list()}
    lut = pl.DataFrame({"tok": list(mp), "canon": list(mp.values()), "skel": [skeleton(v) for v in mp.values()]})
    lut = lut.with_columns(pl.col("canon").str.contains(r"^\d+$").alias("isnum"),
                           pl.col("canon").is_in(list(ADDR_FILLER)).alias("filler"))
    g = tk.join(lut, on="tok", how="left", maintain_order="left").group_by("rid", maintain_order=True).agg(
        pl.col("canon").filter(~pl.col("filler")).str.join(" ").alias("addr_can"),
        pl.col("skel").filter(~pl.col("filler")).str.join(" ").alias("addr_skel"),
        pl.col("canon").filter(pl.col("isnum")).str.join(" ").alias("addr_nums"))
    base = pl.DataFrame({"rid": pl.arange(0, len(a), eager=True).cast(pl.UInt32)})
    out = (base.join(g.with_columns(pl.col("rid").cast(pl.UInt32)), on="rid", how="left")
               .join(state.with_columns(pl.col("rid").cast(pl.UInt32)), on="rid", how="left"))
    return out.select(pl.col("addr_can").fill_null(""), pl.col("addr_skel").fill_null(""),
                      pl.col("addr_nums").fill_null(""), pl.col("state").fill_null("").alias("addr_state"))


def normalize_records(df: pl.DataFrame) -> pl.DataFrame:
    'raw records -> normalised fields (see module docstring)'
    nm = clean_names(df["business_name"])
    return pl.concat([
        df.select("entity_id", pl.col("country").fill_null("").str.strip_chars(),
                  pl.col("business_name").fill_null("").str.contains(NON_ASCII).alias("name_nonascii"),
                  pl.col("business_address").is_null().alias("addr_missing")),
        nm.select("n_is_dom"), name_fields(nm), address_fields(df["business_address"])], how="horizontal")
