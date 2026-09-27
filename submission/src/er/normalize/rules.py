'token-level normalisation rules (hand-written general knowledge; no external data)'
import re


_DIGRAPHS = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("dh", "d"),
             ("bh", "b"), ("kh", "k"), ("gh", "g"), ("ck", "k")]
_CONS_MAP = str.maketrans({"c": "k", "q": "k", "g": "k", "d": "t", "b": "p",
                           "w": "v", "z": "s", "x": "ks"})
_DROP = re.compile(r"[aeiouyh]")


def skeleton(tok: str) -> str:
    'consonant skeleton: robust to vowel typos, transliteration and letter doubling'
    if not tok:
        return ""
    if tok.isdigit():
        return tok
    t = tok
    for a, b in _DIGRAPHS:
        t = t.replace(a, b)
    t = t.translate(_CONS_MAP)
    t = _DROP.sub("", t)
    if not t:
        return tok[:1]
    out = [t[0]]
    for ch in t[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)



LEGAL = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "priv": "pvt",
    "limited": "ltd", "ltd": "ltd", "limitee": "ltd",
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc",
    "plc": "plc", "ltda": "ltd", "gmbh": "gmbh", "ag": "ag", "bv": "bv",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "scop": "scop", "selarl": "selarl",
    "opc": "opc", "public": "public",
}

LEGAL_SKEL = {skeleton(w): v for w, v in [("limited", "ltd"), ("private", "pvt"), ("llp", "llp"), ("pvt", "pvt")]}
LEGAL_SKEL.update({"lmtt": "ltd", "lmt": "ltd", "prvt": "pvt", "lp": "llp", "ll": "llp"})
LEGAL_TRANSLIT = {"elelpi": "llp", "elelpii": "llp", "ellpi": "llp"}

NAME_STOP = {"the", "and", "of", "a", "an", "mr", "mrs", "ms", "dr", "smt", "shri",
             "sri", "shree", "m", "s", "et", "de", "des", "du", "la", "le", "les",
             "l", "d", "for", "in", "on", "at", "by", "to", "esq"}

ALIAS_MARKERS = (r"\b(?:doing business as|d/b/a|dba|t/a|trading as|formerly|f/k/a|fka|a\.k\.a\.|aka|nee"
                 r"|o/a|also known as|operating as)\b:?")


LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a"})



ADDR_NULL_COMPONENTS = ["", "null", "<null>", "none", "nan", "n/a", "na", "<na>", "nil", "-", "--"]

ADDR_CANON = {

    "street": "st", "str": "st", "saint": "st", "st": "st",
    "road": "rd", "rd": "rd", "rod": "rd", "avenue": "ave", "av": "ave", "ave": "ave", "aven": "ave",
    "drive": "dr", "dr": "dr", "drv": "dr", "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd", "boul": "blvd",
    "circle": "cir", "cir": "cir", "place": "pl", "pl": "pl", "trail": "trl", "trl": "trl",
    "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy", "terrace": "ter", "ter": "ter",
    "square": "sq", "sq": "sq", "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt",
    "building": "bldg", "bldg": "bldg", "floor": "fl", "fl": "fl", "flr": "fl",
    "route": "rte", "rte": "rte", "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "way": "way", "wy": "way", "expressway": "expy",
    "point": "pt", "pt": "pt", "heights": "hts", "hts": "hts", "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "junction": "jct", "crossing": "xing", "number": "no", "num": "no", "nr": "near", "near": "near",
    "opposite": "opp", "opp": "opp", "po": "po", "box": "box",

    "nagar": "ngr", "ngr": "ngr", "colony": "col", "col": "col", "sector": "sec", "sec": "sec",
    "block": "blk", "blk": "blk", "phase": "ph", "ph": "ph", "industrial": "indl", "indl": "indl",
    "district": "dist", "dist": "dist", "distt": "dist", "taluk": "tq", "taluka": "tq", "tq": "tq",
    "village": "vill", "vill": "vill", "vpo": "vpo", "complex": "cmplx", "cmplx": "cmplx",
    "marg": "marg", "chowk": "chowk", "bazar": "bazaar", "bazaar": "bazaar",
    "cross": "crs", "crs": "crs", "main": "main", "layout": "lyt", "extension": "extn", "extn": "extn",

    "bombay": "mumbai", "poona": "pune", "madras": "chennai", "calcutta": "kolkata",
    "bengaluru": "bangalore", "gurugram": "gurgaon", "gurgoan": "gurgaon", "trivandrum": "thiruvananthapuram",
    "baroda": "vadodara", "mysuru": "mysore", "mangaluru": "mangalore", "belagavi": "belgaum",
    "cawnpore": "kanpur", "banaras": "varanasi", "benares": "varanasi", "prayagraj": "allahabad",
    "kochi": "cochin", "keralam": "kerala", "tuticorin": "thoothukudi", "vizag": "visakhapatnam",
    "hubli": "hubballi", "gulbarga": "kalaburagi", "simla": "shimla", "pondicherry": "puducherry",

    "rue": "rue", "r": "rue", "chemin": "ch", "che": "ch", "ch": "ch", "impasse": "imp", "imp": "imp",
    "allee": "all", "allees": "all", "all": "all", "quai": "quai", "cours": "crs",
    "faubourg": "fbg", "fbg": "fbg", "sainte": "ste", "residence": "res", "res": "res",
    "batiment": "bat", "bat": "bat", "lieu": "lieu", "dit": "dit", "zone": "zone", "za": "za", "zi": "zi",
    "cedex": "cedex", "bis": "bis",
}
ADDR_FILLER = {"no", "h", "hno", "hn", "door", "unit", "apt", "ste", "fl", "pmb", "box", "po", "near",
               "opp", "null", "none", "nan", "house", "plot", "flat", "shop", "office", "room",
               "the", "of", "and", "de", "du", "des", "la", "le", "les", "d", "l", "et", "story"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts", "tg": "ts", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
