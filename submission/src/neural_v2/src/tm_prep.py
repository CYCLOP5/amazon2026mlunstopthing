'team preprocessing with explicit empty-table dtypes'
import re

import polars as pl
from unidecode import unidecode

from tm_rules import (addr_canon, addr_filler, alias_markers, fr_regions, in_states,
                      legal, legal_skel, name_stop, skeleton, us_states)

non_ascii = r"[^\x00-\x7f]"
state_exact = {}
state_skel = {}
for d in (us_states, in_states, fr_regions):
    for full, code in d.items():
        state_exact[full] = code
        state_exact[code] = code
        sk = skeleton(full.replace(" ", ""))
        if len(sk) >= 2:
            state_skel.setdefault(sk, code)
leet = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a"})


def ascii_fold(s):
    s = s.fill_null("")
    uniq = s.filter(s.str.contains(non_ascii)).unique()
    if len(uniq):
        s = s.replace(uniq, [unidecode(x) for x in uniq.to_list()])
    return s.str.to_lowercase()


def state_of(comp):
    c = comp.strip()
    if c in state_exact:
        return state_exact[c]
    c2 = re.sub(r"[^a-z]", "", c)
    sk = skeleton(c2)
    return state_skel.get(sk) if len(c2) >= 4 else None


def name_tok_map(tok):
    if not tok.isdigit() and not tok.isalpha() and sum(c.isalpha() for c in tok) >= 2:
        tok = tok.translate(leet)
    if tok in legal:
        return legal[tok], 1
    if tok in name_stop:
        return tok, 2
    if tok in ("elelpi", "elelpii", "ellpi"):
        return "llp", 1
    if not tok.isdigit() and len(tok) >= 5:
        sk = skeleton(tok)
        if len(sk) >= 3 and sk in legal_skel:
            return legal_skel[sk], 1
    return tok, 0


def addr_tok_map(tok):
    m = re.fullmatch(r"0*(\d+)(?:st|nd|rd|th)?", tok)
    return m.group(1) or "0" if m else addr_canon.get(tok, tok)


def clean_names(name):
    n = ascii_fold(name)
    n = (n.str.replace_all(r"\(\s*id:?\s*\d+\s*\)", " ")
          .str.replace_all(r"\bid:\s*\d+", " ")
          .str.replace_all(r"\s-\s*\+?\d[\d\s]{5,}", " ")
          .str.replace_all(r"#\d{3,}", " ")
          .str.replace_all(r"\d{7,}", " "))
    ar = rf"^(.*?)\s*{alias_markers}\s*(.*)$"
    has = n.str.contains(alias_markers)
    alt = pl.select(pl.when(has).then(n.str.extract(ar, 1)).otherwise(pl.lit(""))).to_series()
    core = pl.select(pl.when(has).then(n.str.extract(ar, 2)).otherwise(n)).to_series()

    def base(s):
        s = s.fill_null("").str.strip_chars()
        dom = r"^(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|biz|co|io|us|info)$"
        s = pl.select(pl.when(s.str.contains(dom)).then(s.str.extract(dom, 1)).otherwise(s)).to_series()
        s = s.str.replace_all(r"(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|co\.in|in|fr|biz|co|io|us|info)(?:[^a-z0-9]|$)", "$1 ")
        return (s.str.replace_all("&", " and ")
                 .str.replace_all(r"(^|[^a-z0-9])([a-z])\.", "$1$2")
                 .str.replace_all(r"(^|[^a-z0-9])([a-z]{1,2})\.", "$1$2")
                 .str.replace_all("'", "").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())

    return pl.DataFrame({"n_core_raw": base(core), "n_alt_raw": base(alt),
                         "n_is_dom": core.fill_null("").str.contains(r"\.(com|net|org|in|fr|biz|co|io)$")})


def name_features(df):
    toks = df.select(pl.col("n_core_raw").str.split(" ").alias("t"),
                     pl.col("n_alt_raw").str.split(" ").alias("ta"))
    keys = pl.DataFrame({"x": pl.concat([toks["t"], toks["ta"]])}).select(
        pl.col("x").explode().unique().drop_nulls()).to_series().to_list()
    mp = {t: name_tok_map(t) for t in keys}
    lut = pl.DataFrame({"tok": pl.Series(keys, dtype=pl.String),
                        "canon": pl.Series([mp[k][0] for k in keys], dtype=pl.String),
                        "kind": pl.Series([mp[k][1] for k in keys], dtype=pl.Int8),
                        "skel": pl.Series([skeleton(mp[k][0]) for k in keys], dtype=pl.String)})

    def agg(col):
        e = (toks.with_row_index("rid").select("rid", pl.col(col).alias("tok")).explode("tok")
             .filter(pl.col("tok") != "").join(lut, on="tok", how="left", maintain_order="left"))
        return e.group_by("rid", maintain_order=True).agg(
            pl.col("canon").filter(pl.col("kind") == 0).str.join(" ").alias("core"),
            pl.col("skel").filter(pl.col("kind") == 0).str.join(" ").alias("skel"),
            pl.col("canon").filter(pl.col("kind") == 1).unique().sort().str.join(" ").alias("legal"),
            pl.col("canon").filter(pl.col("kind") != 2).str.join(" ").alias("full"))

    a, b = agg("t"), agg("ta").select("rid", pl.col("core").alias("alt"))
    base = pl.DataFrame({"rid": pl.arange(0, df.height, eager=True).cast(pl.UInt32)})
    out = (base.join(a.with_columns(pl.col("rid").cast(pl.UInt32)), on="rid", how="left")
               .join(b.with_columns(pl.col("rid").cast(pl.UInt32)), on="rid", how="left"))
    out = out.with_columns([pl.col(c).fill_null("") for c in ["core", "skel", "legal", "full", "alt"]])
    return out.select(pl.col("core").alias("name_core"), pl.col("skel").alias("name_skel"),
                      pl.col("legal").alias("name_legal"), pl.col("full").alias("name_full"),
                      pl.col("alt").alias("name_alt"), pl.col("core").str.replace_all(" ", "").alias("name_compact"))


def address_features(addr):
    a = ascii_fold(addr)
    comps = (pl.DataFrame({"c": a.str.split(",")}).with_row_index("rid").explode("c")
             .with_columns(pl.col("c").str.strip_chars()).filter(~pl.col("c").is_in(["", "null", "none", "nan"])))
    cand = comps.filter(~pl.col("c").str.contains(r"\d") & (pl.col("c").str.len_chars() <= 30))["c"].unique()
    smap = {c: state_of(c) for c in cand.to_list()}
    smap = {k: v for k, v in smap.items() if v}
    comps = comps.with_columns(pl.col("c").replace_strict(smap, default=None, return_dtype=pl.String).alias("state"))
    state = comps.filter(pl.col("state").is_not_null()).group_by("rid").agg(pl.col("state").first())
    tk = (comps.filter(pl.col("state").is_null())
          .with_columns(pl.col("c").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().str.split(" ").alias("tok"))
          .select("rid", "tok").explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")))
    keys = tk["tok"].unique().to_list()
    mp = {t: addr_tok_map(t) for t in keys}
    lut = pl.DataFrame({"tok": pl.Series(keys, dtype=pl.String),
                        "canon": pl.Series([mp[k] for k in keys], dtype=pl.String),
                        "skel": pl.Series([skeleton(mp[k]) for k in keys], dtype=pl.String)})
    lut = lut.with_columns(pl.col("canon").str.contains(r"^\d+$").alias("isnum"),
                           pl.col("canon").is_in(list(addr_filler)).alias("filler"))
    tk = tk.join(lut, on="tok", how="left", maintain_order="left")
    g = tk.group_by("rid", maintain_order=True).agg(
        pl.col("canon").filter(~pl.col("filler")).str.join(" ").alias("addr_can"),
        pl.col("skel").filter(~pl.col("filler")).str.join(" ").alias("addr_skel"),
        pl.col("canon").filter(pl.col("isnum")).str.join(" ").alias("addr_nums"))
    base = pl.DataFrame({"rid": pl.arange(0, len(a), eager=True).cast(pl.UInt32)})
    out = (base.join(g.with_columns(pl.col("rid").cast(pl.UInt32)), on="rid", how="left")
               .join(state.with_columns(pl.col("rid").cast(pl.UInt32)), on="rid", how="left"))
    return out.select(pl.col("addr_can").fill_null(""), pl.col("addr_skel").fill_null(""),
                      pl.col("addr_nums").fill_null(""), pl.col("state").fill_null("").alias("addr_state"))


def process(df):
    nm = clean_names(df["business_name"])
    nf = name_features(nm)
    af = address_features(df["business_address"])
    return pl.concat([df.select("entity_id", pl.col("country").fill_null("").str.strip_chars(),
                      pl.col("business_name").fill_null("").str.contains(non_ascii).alias("name_nonascii"),
                      pl.col("business_address").is_null().alias("addr_missing")),
                      nm.select("n_is_dom"), nf, af], how="horizontal")
