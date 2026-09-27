'synthetic checks for the country-scoped lexical retrieval preparation'
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final_hybrid.prepare import clean_address, clean_name, run


def _fixture(root, refs, s2):
    data, prepared = root / "data", root / "prepared"
    (data / "train").mkdir(parents=True)
    (prepared / "train").mkdir(parents=True)
    pl.DataFrame(refs).write_parquet(data / "train" / "ref.parquet")
    pl.DataFrame(s2).write_parquet(data / "train" / "s2.parquet")
    pl.DataFrame({"rid": [], "nm": [], "ad": [], "co": []},
                 schema={"rid": pl.UInt32, "nm": pl.String, "ad": pl.String, "co": pl.String}) \
        .write_parquet(data / "train" / "s3.parquet")
    pl.DataFrame({"tid": s2["rid"]}).write_parquet(prepared / "train" / "features.parquet")
    return data, prepared


def test_name_retrieval_is_country_scoped_and_normalizes_unicode_order(tmp_path):
    refs = {
        "rid": [10, 20], "nm": ["Lille Ecole SARL", "Unrelated US Bakery"],
        "ad": ["1 Rue du Port", "2 River Road"], "co": ["france", "us"],
    }
    queries = {
        "rid": [100, 200], "nm": ["S.A.R.L. École Lille", "S.A.R.L. École Lille"],
        "ad": ["9 Rue du Port", "9 Rue du Port"], "co": ["france", "us"],
    }
    data, prepared = _fixture(tmp_path, refs, queries)
    output = tmp_path / "out"
    run(data, prepared, output, split="train", topk=1, threads=1)
    lexical = pl.read_parquet(output / "train" / "lexical.parquet")
    name = lexical.filter(pl.col("lex_name_rank") == 1)
    assert name.select("qid", "tid").rows() == [(10, 100)]
    assert lexical.join(pl.DataFrame({"tid": [100, 200], "co": ["france", "us"]}), on="tid") \
        .join(pl.DataFrame({"qid": [10, 20], "ref_co": ["france", "us"]}), on="qid") \
        .filter(pl.col("co") != pl.col("ref_co")).is_empty()
    assert clean_name("S.A.R.L. École Lille") == clean_name("Lille Ecole SARL")
    report = json.loads((output / "preparation_train.json").read_text())
    assert {x["country"] for x in report["lanes"]["name"]["country_reports"]} == {"france", "us"}
    assert len(list((output / "train" / "index").glob("*.joblib"))) == 4


def test_address_lane_removes_numbers_but_keeps_street_context(tmp_path):
    refs = {
        "rid": [10, 11], "nm": ["Alpha LLC", "Completely Different Shop"],
        "ad": ["92 River Street", "14 Other Avenue"], "co": ["us", "us"],
    }
    queries = {"rid": [100], "nm": ["No Name Match"], "ad": ["14 River Street"], "co": ["us"]}
    data, prepared = _fixture(tmp_path, refs, queries)
    output = tmp_path / "out"
    run(data, prepared, output, split="train", topk=1, threads=1)
    lexical = pl.read_parquet(output / "train" / "lexical.parquet")
    assert lexical.filter(pl.col("lex_address_rank") == 1)["qid"].to_list() == [10]
    assert clean_address("Door 834 House 1958, 14 River Street") == \
        clean_address("Door 92 House 37 River Street")


def test_blank_lane_returns_no_scores_and_reports_empty_vocabulary(tmp_path):
    refs = {"rid": [10, 11], "nm": ["Alpha Company", "Beta Company"],
            "ad": ["", ""], "co": ["us", "us"]}
    queries = {"rid": [100], "nm": ["Alpha Company"], "ad": [""], "co": ["us"]}
    data, prepared = _fixture(tmp_path, refs, queries)
    output = tmp_path / "out"
    run(data, prepared, output, split="train", topk=1, threads=1)
    report = json.loads((output / "preparation_train.json").read_text())
    assert report["lanes"]["address"]["country_reports"][0]["empty_vocabulary"] is True
    lexical = pl.read_parquet(output / "train" / "lexical.parquet")
    assert lexical["lex_address_score"].to_list() == [0.0]
    assert clean_address("") == ""
