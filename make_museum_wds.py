import os 
import math
import tarfile
import io 
# pandas and torch necessary but assuming already installed

# paths 
CSV_PATH="" # path to dataset
IMAGES_ROOT="" # name of img folder 
OUTPUT_ROOT="museum_wds" # folder we create for webdataset
SHARD_SIZE="" # samples per .tar 'box'

# turn csv rows into text descriptions for retrieval

def build_caption(row):
    title = str(row.get("title", "")).strip()
    desc = str(row.get("description", "")).strip()

    parts = []
    
    if title:
        parts.append(f"Title: {title}")
    if desc:
        parts.append(f"Description: {desc}")
    
    for col in ["culture", "location", "date", "material and techniques", "category"]:
        if col in row and pd.notna(row[col]):
            parts.append(f"{col.capitalize()}: {row[col]}")

    if not parts: 
        return "Museum artifact."
    
    return ". ".join(parts) + "."

def main():
    df = pd.read_csv(""" csv path""")
    n = len(df)
    if n == 0:
        raise ValueError("CSV is empty, nothing to export.")
    
    # prepare output dirs
    train_dir = os.path.join(OUTPUT_ROOT, "train")
    os.makedirs(train_dir, exist_ok=True)

    # figure out how many shards are needed
    nshards = math.ceil(n / SHARD_SIZE)
    nshards_path = os.path.join(train_dir, "nshards.txt")
    with open(nshards_path, "w") as f:
        f.write(str(nshards) + "\n")
    print(f"Total samples: {n}, shards: {nshards}, shard size: {SHARD_SIZE}")

    # loop over shards
    for shard_id in range (nshards):
        start = shard_id * SHARD_SIZE
        end = min((shard_id + 1) * SHARD_SIZE, n)
        shard_df = df.iloc[start:end]

        shard_path = os.path.join(train_dir, f"{shard_id}.tar")
        print(f"Writing shard {shard_id}: rows {start}-{end-1} -> {shard_path}")

        with tarfile.open(shard_path, "w") as tar:
            for idx, row in shard_df.iterrows():
                # build base name for the files in this sample 
                # use id column if present, else use csv

                if "id" in row and not pd.isna(row["id"]):
                    base = str(row["id"])
                else: 
                    base = f"sample_{idx}"

                # img path

                image_rel = str(row.get("image_path", "")).strip()
                if not image_rel:
                    print(f" row {idx} has no image_path, skipping")
                    continue 

                image_path = os.path.join(IMAGES_ROOT, image_rel)
                if not os.path.exists(image_path):
                    print(f" Image file not found: {image_path}, skipping")
                    continue

                # txt caption
                caption = build_caption(row)
                caption_bytes = caption.encode("utf-8")

                # add image to tar as (base).jpg
                img_ext = os.path.splitext(image_path)[1] or ".jpg"
                img_arcname = f"{base}{img_ext}"

                tar.add(image_path, arcname=img_arcname)

                # add text to tar as (base).txt
                txt_arcname = f"{base}.txt"
                txt_info = tarfile.TarInfo(name=txt_arcname)
                txt_info.size = len(caption_bytes)
                # wraps data into file like object, and adds to tar archive
                tar.addfile(txt_info, io.BytesIO(caption_bytes))
            
            # so dataset_type.txt says "retrieval"
    dataset_type_path = os.path.join(OUTPUT_ROOT, "dataset_type.txt")
    with open(dataset_type_path, "w") as f:
        f.write("retrieval\n")
            
    print("Webdataset written to: ", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
        


    