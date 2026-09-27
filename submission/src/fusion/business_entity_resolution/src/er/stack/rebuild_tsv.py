'rebuild the challenge tsvs from prepared parquet (azure jobs have only the parquet)'
import os
import sys

import polars as pl

COUNTRY = {"us": "US", "india": "India", "france": "France"}


def main(src, dst):
    for split in ("train", "test"):
        os.makedirs(os.path.join(dst, split), exist_ok=True)
        for i, name in ((1, "ref"), (2, "s2"), (3, "s3")):
            d = pl.read_parquet(os.path.join(src, split, f"{name}.parquet"), columns=["rid", "eid", "nm", "ad", "co"]).sort("rid")
            d = d.select(pl.col("eid").alias("entity_id"), pl.col("nm").alias("business_name"),
                         pl.col("ad").alias("business_address"),
                         pl.col("co").replace(COUNTRY).alias("country"))
            d.write_csv(os.path.join(dst, split, f"{split}_source{i}.tsv"), separator="\t", quote_style="never")
            print(split, i, d.height, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
