"""team supplied token rules used by the verified lightgbm checkpoint"""
import re

dg = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("dh", "d"),
      ("bh", "b"), ("kh", "k"), ("gh", "g"), ("ck", "k")]
cm = str.maketrans({"c": "k", "q": "k", "g": "k", "d": "t", "b": "p",
                    "w": "v", "z": "s", "x": "ks"})
drop = re.compile(r"[aeiouyh]")


def skeleton(tok):
    if not tok:
        return ""
    if tok.isdigit():
        return tok
    t = tok
    for a, b in dg:
        t = t.replace(a, b)
    t = drop.sub("", t.translate(cm))
    if not t:
        return tok[:1]
    out = [t[0]]
    for ch in t[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


legal = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "priv": "pvt",
    "limited": "ltd", "ltd": "ltd", "limitee": "ltd",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc",
    "plc": "plc", "ltda": "ltd", "gmbh": "gmbh", "ag": "ag", "bv": "bv",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "scop": "scop", "selarl": "selarl",
    "opc": "opc", "public": "public",
}
legal_skel = {skeleton(w): v for w, v in [("limited", "ltd"), ("private", "pvt"),
                                          ("llp", "llp"), ("pvt", "pvt")]}
legal_skel.update({"lmtt": "ltd", "lmt": "ltd", "prvt": "pvt", "lp": "llp", "ll": "llp"})
name_stop = {"the", "and", "of", "a", "an", "mr", "mrs", "ms", "dr", "smt", "shri",
             "sri", "shree", "m", "s", "et", "de", "des", "du", "la", "le", "les",
             "l", "d", "for", "in", "on", "at", "by", "to", "esq"}
alias_markers = r"\b(?:doing business as|d/b/a|dba|t/a|trading as|formerly|f/k/a|fka|a\.k\.a\.|aka|nee|o/a|also known as|operating as)\b:?"
addr_canon = {
    "street": "st", "str": "st", "saint": "st", "st": "st",
    "road": "rd", "rd": "rd", "avenue": "ave", "av": "ave", "ave": "ave", "aven": "ave",
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
    "rue": "rue", "r": "rue", "chemin": "ch", "che": "ch", "ch": "ch", "impasse": "imp", "imp": "imp",
    "allee": "all", "allees": "all", "all": "all", "quai": "quai", "cours": "crs",
    "faubourg": "fbg", "fbg": "fbg", "sainte": "ste", "residence": "res", "res": "res",
    "batiment": "bat", "bat": "bat", "lieu": "lieu", "dit": "dit", "zone": "zone", "za": "za", "zi": "zi",
    "cedex": "cedex", "bis": "bis",
}
addr_filler = {"no", "h", "hno", "door", "unit", "apt", "ste", "fl", "pmb", "box", "po", "near",
               "opp", "null", "none", "nan", "house", "plot", "flat", "shop", "office", "room",
               "the", "of", "and", "de", "du", "des", "la", "le", "les", "d", "l", "et", "story"}
us_states = {
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
in_states = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
fr_regions = {
    "auvergne-rhone-alpes": "ara", "auvergne rhone alpes": "ara",
    "bourgogne-franche-comte": "bfc", "bourgogne franche comte": "bfc", "bretagne": "bre",
    "centre-val de loire": "cvl", "centre val de loire": "cvl", "corse": "cor", "grand est": "ge",
    "hauts-de-france": "hdf", "hauts de france": "hdf",
    "ile-de-france": "idf", "ile de france": "idf", "idf": "idf", "normandie": "nor",
    "nouvelle-aquitaine": "na", "nouvelle aquitaine": "na", "occitanie": "occ", "pays de la loire": "pdl",
    "provence-alpes-cote d'azur": "paca", "provence alpes cote d'azur": "paca",
    "provence-alpes-cote dazur": "paca", "paca": "paca",
    "guadeloupe": "gp", "martinique": "mq", "guyane": "gf", "la reunion": "re", "mayotte": "yt",
    "aquitaine": "na", "limousin": "na", "poitou-charentes": "na", "poitou charentes": "na",
    "alsace": "ge", "champagne-ardenne": "ge", "champagne ardenne": "ge", "lorraine": "ge",
    "auvergne": "ara", "rhone-alpes": "ara", "rhone alpes": "ara",
    "bourgogne": "bfc", "franche-comte": "bfc", "franche comte": "bfc",
    "languedoc-roussillon": "occ", "languedoc roussillon": "occ", "midi-pyrenees": "occ", "midi pyrenees": "occ",
    "nord-pas-de-calais": "hdf", "nord pas de calais": "hdf", "picardie": "hdf",
    "basse-normandie": "nor", "basse normandie": "nor", "haute-normandie": "nor", "haute normandie": "nor",
}
